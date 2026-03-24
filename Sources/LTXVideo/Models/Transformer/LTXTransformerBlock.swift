// LTXTransformerBlock.swift - Transformer Block for LTX-2
// Copyright 2025

import Foundation
@preconcurrency import MLX
import MLXFast
import MLXNN

// MARK: - Transformer Arguments

/// Arguments passed to transformer blocks during forward pass
struct TransformerArgs {
    /// Hidden states (B, T, D)
    var x: MLXArray

    /// Text context for cross-attention (B, S, D_ctx)
    var context: MLXArray

    /// Timestep embeddings for AdaLN (B, T, numSST, D)
    var timesteps: MLXArray

    /// RoPE position embeddings (cos, sin)
    var positionalEmbeddings: (cos: MLXArray, sin: MLXArray)

    /// Optional attention mask for cross-attention
    var contextMask: MLXArray?

    /// Raw embedded timestep for output projection
    var embeddedTimestep: MLXArray?

    /// Whether this modality is enabled
    var enabled: Bool

    /// Prompt timestep embeddings for cross-attention AdaLN (LTX-2.3)
    /// Shape: (B, 1, 2, D) — shift and scale for prompt modulation
    var promptTimesteps: MLXArray?

    /// Block indices where video self-attention should be skipped (STG perturbation).
    /// When a block's index is in this set, self-attention output is zeroed.
    var skipSelfAttentionBlocks: Set<Int>?

    init(
        x: MLXArray,
        context: MLXArray,
        timesteps: MLXArray,
        positionalEmbeddings: (cos: MLXArray, sin: MLXArray),
        contextMask: MLXArray? = nil,
        embeddedTimestep: MLXArray? = nil,
        enabled: Bool = true,
        promptTimesteps: MLXArray? = nil
    ) {
        self.x = x
        self.context = context
        self.timesteps = timesteps
        self.positionalEmbeddings = positionalEmbeddings
        self.contextMask = contextMask
        self.embeddedTimestep = embeddedTimestep
        self.enabled = enabled
        self.promptTimesteps = promptTimesteps
    }

    /// Return a new TransformerArgs with specified fields replaced
    func replacing(x: MLXArray) -> TransformerArgs {
        return TransformerArgs(
            x: x,
            context: context,
            timesteps: timesteps,
            positionalEmbeddings: positionalEmbeddings,
            contextMask: contextMask,
            embeddedTimestep: embeddedTimestep,
            enabled: enabled,
            promptTimesteps: promptTimesteps
        )
    }
}

// MARK: - AdaLN Helpers

/// Apply AdaLN: RMSNorm + scale + shift
///
/// Matches Python: `rms_norm(x) * (1 + scale) + shift`
/// where rms_norm creates `mx.ones(..., dtype=x.dtype)` weight.
private func adaln(
    _ x: MLXArray,
    scale: MLXArray,
    shift: MLXArray,
    eps: Float = 1e-6
) -> MLXArray {
    // RMS normalization (use identity weight in input dtype)
    let weight = MLXArray.ones([x.dim(-1)]).asType(x.dtype)
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
class BasicTransformerBlock: Module {
    let normEps: Float
    let numSSTValues: Int  // 6 for LTX-2, 9 for LTX-2.3

    @ModuleInfo(key: "attn1") var attn1: LTXAttention
    @ModuleInfo(key: "attn2") var attn2: LTXAttention
    @ModuleInfo(key: "ff") var ff: LTXFeedForward

    /// AdaLN scale-shift table: 6 or 9 values
    /// LTX-2: [6, dim] — (shift,scale,gate) x (self-attn, ffn)
    /// LTX-2.3: [9, dim] — (shift,scale,gate) x (self-attn, cross-attn, ffn)
    @ParameterInfo(key: "scale_shift_table") var scaleShiftTable: MLXArray

    /// Cross-attention prompt modulation (LTX-2.3)
    @ParameterInfo(key: "prompt_scale_shift_table") var promptScaleShiftTable: MLXArray?

    /// Cross-attention output scaling factor (1.0 = no change, >1.0 = stronger prompt adherence)
    var crossAttentionScale: Float = 1.0

    /// When true, self-attention is skipped (STG perturbation for guidance)
    var skipSelfAttention: Bool = false

    init(
        dim: Int,
        numHeads: Int,
        headDim: Int,
        contextDim: Int,
        ropeType: LTXRopeType = .split,
        normEps: Float = 1e-6,
        gatedAttention: Bool = false,
        crossAttentionAdaLN: Bool = false
    ) {
        self.normEps = normEps
        self.numSSTValues = crossAttentionAdaLN ? 9 : 6

        // Self-attention
        self._attn1.wrappedValue = LTXAttention(
            queryDim: dim,
            contextDim: nil,  // Self-attention
            heads: numHeads,
            dimHead: headDim,
            normEps: normEps,
            ropeType: ropeType,
            gatedAttention: gatedAttention
        )

        // Cross-attention
        self._attn2.wrappedValue = LTXAttention(
            queryDim: dim,
            contextDim: contextDim,
            heads: numHeads,
            dimHead: headDim,
            normEps: normEps,
            ropeType: ropeType,
            gatedAttention: gatedAttention
        )

        // Feed-forward
        self._ff.wrappedValue = LTXFeedForward(dim: dim, dimOut: dim)

        // AdaLN scale-shift table (kept as float32 for numerical stability)
        self._scaleShiftTable.wrappedValue = MLXArray.zeros([numSSTValues, dim])

        // Cross-attention prompt modulation (LTX-2.3)
        if crossAttentionAdaLN {
            self._promptScaleShiftTable.wrappedValue = MLXArray.zeros([2, dim])
        }
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

    func callAsFunction(_ args: TransformerArgs) -> TransformerArgs {
        var x = args.x

        // Get AdaLN values for self-attention (indices 0-2)
        let (shiftMSA, scaleMSA, gateMSA) = getAdaValues(
            batchSize: x.dim(0),
            timestep: args.timesteps,
            start: 0,
            end: 3
        )

        // Self-attention with AdaLN (skip for STG perturbation)
        let normX = adaln(x, scale: scaleMSA, shift: shiftMSA, eps: normEps)
        if skipSelfAttention {
            // STG: skip self-attention, just apply gate to zero → residual unchanged
        } else {
            let attnOut = attn1(normX, pe: args.positionalEmbeddings)
            x = residualGate(x, residual: attnOut, gate: gateMSA)
        }

        // Cross-attention
        if numSSTValues == 9 {
            // LTX-2.3: cross-attention with AdaLN on both query and context
            // Reference: slice(6, 9) — cross-attention uses SST indices 6,7,8
            let (shiftCA, scaleCA, gateCA) = getAdaValues(
                batchSize: x.dim(0),
                timestep: args.timesteps,
                start: 6,
                end: 9
            )

            // Modulate query with AdaLN
            let normXCA = adaln(x, scale: scaleCA, shift: shiftCA, eps: normEps)

            // Modulate context with prompt AdaLN
            var ctx = args.context
            if let promptSST = promptScaleShiftTable, let promptTs = args.promptTimesteps {
                let promptMod = promptSST.reshaped([1, 1, 2, -1]) + promptTs
                let promptShift = promptMod[0..., 0..., 0, 0...]
                let promptScale = promptMod[0..., 0..., 1, 0...]
                ctx = ctx * (1 + promptScale) + promptShift
            }

            var crossOut = attn2(normXCA, context: ctx, mask: args.contextMask)
            if crossAttentionScale != 1.0 {
                crossOut = crossOut * MLXArray(crossAttentionScale)
            }
            x = x + crossOut * gateCA
        } else {
            // LTX-2: cross-attention without AdaLN (qNorm inside attn2 handles Q normalization)
            var crossOut = attn2(
                x,
                context: args.context,
                mask: args.contextMask
            )
            if crossAttentionScale != 1.0 {
                crossOut = crossOut * MLXArray(crossAttentionScale)
            }
            x = x + crossOut
        }

        // Get AdaLN values for FFN
        // Reference: slice(3, 6) — FFN uses SST indices 3,4,5 (both LTX-2 and LTX-2.3)
        let ffStart = 3
        let (shiftMLP, scaleMLP, gateMLP) = getAdaValues(
            batchSize: x.dim(0),
            timestep: args.timesteps,
            start: ffStart,
            end: ffStart + 3
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
class TransformerBlocks: Module {
    @ModuleInfo(key: "blocks") var blocks: [BasicTransformerBlock]

    let memoryOptimization: MemoryOptimizationConfig

    init(
        numLayers: Int,
        dim: Int,
        numHeads: Int,
        headDim: Int,
        contextDim: Int,
        ropeType: LTXRopeType = .split,
        normEps: Float = 1e-6,
        memoryOptimization: MemoryOptimizationConfig = .default
    ) {
        self.memoryOptimization = memoryOptimization

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

    func callAsFunction(_ args: TransformerArgs) -> TransformerArgs {
        var current = args
        let evalFreq = memoryOptimization.evalFrequency

        for (i, block) in blocks.enumerated() {
            current = block(current)

            // Periodic evaluation to manage memory
            if evalFreq > 0 && (i + 1) % evalFreq == 0 {
                eval(current.x)
                if memoryOptimization.clearCacheOnEval {
                    Memory.clearCache()
                }
            }
        }

        return current
    }
}
