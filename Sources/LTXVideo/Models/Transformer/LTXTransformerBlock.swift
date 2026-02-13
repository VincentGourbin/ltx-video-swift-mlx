// LTXTransformerBlock.swift - Transformer Block for LTX-2
// Copyright 2025

import Foundation
@preconcurrency import MLX
import MLXFast
import MLXNN

// MARK: - Transformer Arguments

/// Arguments passed to transformer blocks during forward pass
public struct TransformerArgs {
    /// Hidden states (B, T, D)
    public var x: MLXArray

    /// Text context for cross-attention (B, S, D_ctx)
    public var context: MLXArray

    /// Timestep embeddings for AdaLN (B, T, 6, D)
    public var timesteps: MLXArray

    /// RoPE position embeddings (cos, sin)
    public var positionalEmbeddings: (cos: MLXArray, sin: MLXArray)

    /// Optional attention mask for cross-attention
    public var contextMask: MLXArray?

    /// Raw embedded timestep for output projection
    public var embeddedTimestep: MLXArray?

    /// Whether this modality is enabled
    public var enabled: Bool

    public init(
        x: MLXArray,
        context: MLXArray,
        timesteps: MLXArray,
        positionalEmbeddings: (cos: MLXArray, sin: MLXArray),
        contextMask: MLXArray? = nil,
        embeddedTimestep: MLXArray? = nil,
        enabled: Bool = true
    ) {
        self.x = x
        self.context = context
        self.timesteps = timesteps
        self.positionalEmbeddings = positionalEmbeddings
        self.contextMask = contextMask
        self.embeddedTimestep = embeddedTimestep
        self.enabled = enabled
    }

    /// Return a new TransformerArgs with specified fields replaced
    public func replacing(x: MLXArray) -> TransformerArgs {
        return TransformerArgs(
            x: x,
            context: context,
            timesteps: timesteps,
            positionalEmbeddings: positionalEmbeddings,
            contextMask: contextMask,
            embeddedTimestep: embeddedTimestep,
            enabled: enabled
        )
    }
}

// MARK: - AdaLN Helpers

/// Apply AdaLN: RMSNorm + scale + shift
private func adaln(
    _ x: MLXArray,
    scale: MLXArray,
    shift: MLXArray,
    eps: Float = 1e-6
) -> MLXArray {
    // RMS normalization (use identity weight)
    let weight = MLXArray.ones([x.dim(-1)])
    let normed = MLXFast.rmsNorm(x, weight: weight, eps: eps)
    // Apply adaptive scale and shift
    return normed * (1 + scale) + shift
}

/// Apply residual with gating
private func residualGate(
    _ x: MLXArray,
    residual: MLXArray,
    gate: MLXArray
) -> MLXArray {
    return x + residual * gate
}

// MARK: - Basic Transformer Block

/// A basic transformer block with self-attention, cross-attention, and feed-forward
///
/// Uses AdaLN (Adaptive Layer Norm) for timestep conditioning:
/// - scale and shift parameters are computed from timestep embeddings
/// - applied to normalized hidden states before each sub-layer
///
/// Architecture:
///     1. Self-attention with RoPE and AdaLN
///     2. Cross-attention to text context
///     3. Feed-forward network with AdaLN
public class BasicTransformerBlock: Module {
    let normEps: Float

    @ModuleInfo(key: "attn1") var attn1: LTXAttention
    @ModuleInfo(key: "attn2") var attn2: LTXAttention
    @ModuleInfo(key: "ff") var ff: LTXFeedForward

    /// AdaLN scale-shift table: 6 values (scale, shift, gate) x 2 (attn, ff)
    @ParameterInfo(key: "scale_shift_table") var scaleShiftTable: MLXArray

    public init(
        dim: Int,
        numHeads: Int,
        headDim: Int,
        contextDim: Int,
        ropeType: LTXRopeType = .split,
        normEps: Float = 1e-6
    ) {
        self.normEps = normEps

        // Self-attention
        self._attn1.wrappedValue = LTXAttention(
            queryDim: dim,
            contextDim: nil,  // Self-attention
            heads: numHeads,
            dimHead: headDim,
            normEps: normEps,
            ropeType: ropeType
        )

        // Cross-attention
        self._attn2.wrappedValue = LTXAttention(
            queryDim: dim,
            contextDim: contextDim,
            heads: numHeads,
            dimHead: headDim,
            normEps: normEps,
            ropeType: ropeType
        )

        // Feed-forward
        self._ff.wrappedValue = LTXFeedForward(dim: dim, dimOut: dim)

        // AdaLN scale-shift table (kept as float32 for numerical stability)
        self._scaleShiftTable.wrappedValue = MLXArray.zeros([6, dim])
    }

    /// Get adaptive normalization values from timestep embedding
    private func getAdaValues(
        batchSize: Int,
        timestep: MLXArray,
        start: Int,
        end: Int
    ) -> (MLXArray, MLXArray, MLXArray) {
        // scale_shift_table: (6, D)
        // timestep: (B, T, 6, D) where T is the number of tokens
        let tableSlice = scaleShiftTable[start..<end]  // (num_values, D)

        // Broadcast and add
        // table_slice: (1, 1, num_values, D) + timestep: (B, T, num_values, D)
        let tableExpanded = tableSlice.reshaped([1, 1, end - start, -1])
        let timestepSlice = timestep[0..., 0..., start..<end, 0...]
        let adaValues = tableExpanded + timestepSlice

        // Split into individual values
        return (
            adaValues[0..., 0..., 0, 0...],  // shift or scale
            adaValues[0..., 0..., 1, 0...],  // scale or shift
            adaValues[0..., 0..., 2, 0...]   // gate
        )
    }

    public func callAsFunction(_ args: TransformerArgs) -> TransformerArgs {
        var x = args.x

        // Get AdaLN values for self-attention
        let (shiftMSA, scaleMSA, gateMSA) = getAdaValues(
            batchSize: x.dim(0),
            timestep: args.timesteps,
            start: 0,
            end: 3
        )

        // Self-attention with AdaLN
        let normX = adaln(x, scale: scaleMSA, shift: shiftMSA, eps: normEps)
        let attnOut = attn1(normX, pe: args.positionalEmbeddings)
        x = residualGate(x, residual: attnOut, gate: gateMSA)

        // Cross-attention (no AdaLN, just RMSNorm)
        let crossOut = attn2(
            rmsNorm(x, eps: normEps),
            context: args.context,
            mask: args.contextMask
        )
        x = x + crossOut

        // Get AdaLN values for FFN
        let (shiftMLP, scaleMLP, gateMLP) = getAdaValues(
            batchSize: x.dim(0),
            timestep: args.timesteps,
            start: 3,
            end: 6
        )

        // Feed-forward with AdaLN
        let xScaled = adaln(x, scale: scaleMLP, shift: shiftMLP, eps: normEps)
        let ffOut = ff(xScaled)
        x = residualGate(x, residual: ffOut, gate: gateMLP)

        return args.replacing(x: x)
    }
}

// MARK: - Transformer Blocks Stack

/// Stack of transformer blocks
public class TransformerBlocks: Module {
    @ModuleInfo(key: "blocks") var blocks: [BasicTransformerBlock]

    let evalFrequency: Int

    public init(
        numLayers: Int,
        dim: Int,
        numHeads: Int,
        headDim: Int,
        contextDim: Int,
        ropeType: LTXRopeType = .split,
        normEps: Float = 1e-6,
        evalFrequency: Int = 4
    ) {
        self.evalFrequency = evalFrequency

        self._blocks.wrappedValue = (0..<numLayers).map { _ in
            BasicTransformerBlock(
                dim: dim,
                numHeads: numHeads,
                headDim: headDim,
                contextDim: contextDim,
                ropeType: ropeType,
                normEps: normEps
            )
        }
    }

    public func callAsFunction(_ args: TransformerArgs) -> TransformerArgs {
        var current = args

        for (i, block) in blocks.enumerated() {
            current = block(current)

            // Periodic evaluation to manage memory
            if evalFrequency > 0 && (i + 1) % evalFrequency == 0 {
                eval(current.x)
            }
        }

        return current
    }
}
