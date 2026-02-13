// LTXTransformer.swift - Main LTX-2 Transformer Model
// Copyright 2025

import Foundation
@preconcurrency import MLX
import MLXFast
import MLXNN

// MARK: - LTX Model

/// LTX-2 Transformer Model for Video Generation
///
/// Architecture:
/// - Input: Patchified video latents (B, T, C) where T = F' * H' * W'
/// - 48 transformer blocks with self-attention, cross-attention, and FFN
/// - AdaLN conditioning on timestep
/// - Output: Velocity predictions for diffusion (B, T, C)
///
/// This is the core denoising model that predicts velocities.
public class LTXTransformer: Module {
    let config: LTXTransformerConfig
    let ropeType: LTXRopeType
    let normEps: Float
    let evalFrequency: Int

    // Input projection
    @ModuleInfo(key: "patchify_proj") var patchifyProj: Linear

    // AdaLN for timestep
    @ModuleInfo(key: "adaln_single") var adalnSingle: AdaLayerNormSingle

    // Caption projection
    @ModuleInfo(key: "caption_projection") var captionProjection: PixArtAlphaTextProjection

    // Transformer blocks
    @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [BasicTransformerBlock]

    // Output projection
    @ModuleInfo(key: "norm_out") var normOut: LayerNorm
    @ModuleInfo(key: "proj_out") var projOut: Linear

    // Scale-shift table for output
    @ParameterInfo(key: "scale_shift_table") var scaleShiftTable: MLXArray

    public init(
        config: LTXTransformerConfig = .default,
        ropeType: LTXRopeType = .split,
        evalFrequency: Int = 4
    ) {
        self.config = config
        self.ropeType = ropeType
        self.normEps = config.normEps
        self.evalFrequency = evalFrequency

        let innerDim = config.innerDim

        // Input projection: project latent channels to inner dimension
        self._patchifyProj.wrappedValue = Linear(config.inChannels, innerDim, bias: true)

        // AdaLN for timestep (6 embeddings: scale, shift, gate for self-attn and ffn)
        self._adalnSingle.wrappedValue = AdaLayerNormSingle(innerDim: innerDim, numEmbeddings: 6)

        // Caption projection: Gemma 3840 -> inner dim 4096
        self._captionProjection.wrappedValue = PixArtAlphaTextProjection(
            inFeatures: config.captionChannels,
            hiddenSize: innerDim
        )

        // Transformer blocks
        self._transformerBlocks.wrappedValue = (0..<config.numLayers).map { _ in
            BasicTransformerBlock(
                dim: innerDim,
                numHeads: config.numAttentionHeads,
                headDim: config.attentionHeadDim,
                contextDim: config.crossAttentionDim,
                ropeType: ropeType,
                normEps: config.normEps
            )
        }

        // Output projection
        self._normOut.wrappedValue = LayerNorm(dimensions: innerDim, eps: config.normEps, affine: false)
        self._projOut.wrappedValue = Linear(innerDim, config.outChannels)

        // Scale-shift table for output (2 values: scale, shift)
        self._scaleShiftTable.wrappedValue = MLXArray.zeros([2, innerDim])
    }

    /// Prepare timestep embeddings
    private func prepareTimestep(
        timestep: MLXArray,
        batchSize: Int,
        numTokens: Int
    ) -> (emb: MLXArray, embeddedTimestep: MLXArray) {
        // Scale timestep
        let scaledTimestep = timestep * Float(config.timestepScaleMultiplier)

        // Get AdaLN embeddings
        let (emb, embeddedTimestep) = adalnSingle(scaledTimestep.flattened())

        // Reshape emb to (B, num_tokens, num_embeddings, inner_dim)
        let numEmbeddings = 6
        let reshapedEmb = emb.reshaped([batchSize, -1, numEmbeddings, config.innerDim])

        // Reshape embedded_timestep to (B, num_tokens, inner_dim)
        let reshapedEmbedded = embeddedTimestep.reshaped([batchSize, -1, config.innerDim])

        return (reshapedEmb, reshapedEmbedded)
    }

    /// Prepare context (caption) for cross-attention
    private func prepareContext(
        context: MLXArray,
        batchSize: Int
    ) -> MLXArray {
        var projected = captionProjection(context)
        projected = projected.reshaped([batchSize, -1, config.innerDim])
        return projected
    }

    /// Prepare attention mask for cross-attention
    private func prepareAttentionMask(
        _ mask: MLXArray?,
        targetDtype: DType = .float32
    ) -> MLXArray? {
        guard let mask = mask else { return nil }

        // If already a float mask, return as-is
        if mask.dtype == .float16 || mask.dtype == .float32 || mask.dtype == .bfloat16 {
            return mask
        }

        // Use dtype-appropriate max value for masking
        let maskValue: Float
        switch targetDtype {
        case .float16:
            maskValue = -65504.0
        case .bfloat16:
            maskValue = -3.38e38
        default:
            maskValue = -3.40e38
        }

        // Convert boolean mask to additive mask
        // True = attend (0), False = don't attend (large negative)
        let floatMask = (1 - mask.asType(.float32)) * maskValue
        let reshapedMask = floatMask.reshaped([mask.dim(0), 1, 1, mask.dim(-1)])
        return reshapedMask.asType(targetDtype)
    }

    /// Prepare positional embeddings (RoPE)
    private func preparePositionalEmbeddings(
        batchSize: Int,
        frames: Int,
        height: Int,
        width: Int
    ) -> (cos: MLXArray, sin: MLXArray) {
        // Create pixel-space position grid (matching Python's get_pixel_coords pipeline)
        let positions = createPositionGrid(
            batchSize: batchSize,
            frames: frames,
            height: height,
            width: width
        )

        // Log position diagnostics
        let tPos = positions[0, 0, 0...]  // temporal positions for first batch
        let hPos = positions[0, 1, 0...]  // height positions
        let wPos = positions[0, 2, 0...]  // width positions
        LTXDebug.log("  [DIAG] Position grid (pixel-space): t_min=\(tPos.min().item(Float.self)), t_max=\(tPos.max().item(Float.self)), h_min=\(hPos.min().item(Float.self)), h_max=\(hPos.max().item(Float.self)), w_min=\(wPos.min().item(Float.self)), w_max=\(wPos.max().item(Float.self))")

        // Precompute frequencies
        return precomputeFreqsCis(
            indicesGrid: positions,
            dim: config.innerDim,
            theta: config.ropeTheta,
            maxPos: config.maxPos,
            numAttentionHeads: config.numAttentionHeads,
            ropeType: ropeType
        )
    }

    /// Process output with scale-shift
    private func processOutput(
        _ x: MLXArray,
        embeddedTimestep: MLXArray
    ) -> MLXArray {
        // Get scale and shift from table + timestep
        let scaleShiftValues = scaleShiftTable.reshaped([1, 1, 2, -1]) + embeddedTimestep.reshaped([embeddedTimestep.dim(0), -1, 1, config.innerDim])

        let shift = scaleShiftValues[0..., 0..., 0, 0...]
        let scale = scaleShiftValues[0..., 0..., 1, 0...]

        // Apply normalization with scale and shift
        var output = normOut(x)
        output = output * (1 + scale) + shift

        // Project to output channels
        return projOut(output)
    }

    /// Forward pass
    ///
    /// - Parameters:
    ///   - latent: Patchified video latents (B, T, C) where T = F' * H' * W'
    ///   - context: Text context embeddings (B, S, D_ctx)
    ///   - timesteps: Timestep values (B,)
    ///   - contextMask: Optional attention mask for text (B, S)
    ///   - latentShape: Shape of latent (frames, height, width) for position embeddings
    /// - Returns: Velocity predictions (B, T, C)
    public func callAsFunction(
        latent: MLXArray,
        context: MLXArray,
        timesteps: MLXArray,
        contextMask: MLXArray? = nil,
        latentShape: (frames: Int, height: Int, width: Int)
    ) -> MLXArray {
        let startTime = Date()
        var lastTime = startTime

        let batchSize = latent.dim(0)
        let numTokens = latent.dim(1)

        LTXDebug.log("Transformer input shapes:")
        LTXDebug.log("  latent: \(latent.shape)")
        LTXDebug.log("  context: \(context.shape)")
        LTXDebug.log("  timesteps: \(timesteps.shape)")
        LTXDebug.log("  latentShape: \(latentShape)")

        // Project latents to inner dimension
        var x = patchifyProj(latent)
        eval(x)
        var now = Date()
        LTXDebug.log("  [TIME] patchifyProj: \(String(format: "%.3f", now.timeIntervalSince(lastTime)))s")
        lastTime = now
        LTXDebug.log("  after patchifyProj: \(x.shape)")

        // Prepare timestep embeddings
        let (timestepEmb, embeddedTimestep) = prepareTimestep(
            timestep: timesteps,
            batchSize: batchSize,
            numTokens: numTokens
        )
        eval(timestepEmb, embeddedTimestep)
        now = Date()
        LTXDebug.log("  [TIME] prepareTimestep: \(String(format: "%.3f", now.timeIntervalSince(lastTime)))s")
        lastTime = now

        // Prepare context (caption projection)
        let projectedContext = prepareContext(context: context, batchSize: batchSize)
        eval(projectedContext)
        now = Date()
        LTXDebug.log("  [TIME] prepareContext: \(String(format: "%.3f", now.timeIntervalSince(lastTime)))s")
        lastTime = now

        // Prepare attention mask
        let preparedMask = prepareAttentionMask(contextMask)

        // Prepare positional embeddings (RoPE)
        let pe = preparePositionalEmbeddings(
            batchSize: batchSize,
            frames: latentShape.frames,
            height: latentShape.height,
            width: latentShape.width
        )
        eval(pe.cos, pe.sin)
        now = Date()
        LTXDebug.log("  [TIME] preparePositionalEmbeddings (RoPE): \(String(format: "%.3f", now.timeIntervalSince(lastTime)))s")
        lastTime = now
        LTXDebug.log("  RoPE cos shape: \(pe.cos.shape)")
        LTXDebug.log("  RoPE sin shape: \(pe.sin.shape)")

        // Create transformer args
        var args = TransformerArgs(
            x: x,
            context: projectedContext,
            timesteps: timestepEmb,
            positionalEmbeddings: pe,
            contextMask: preparedMask,
            embeddedTimestep: embeddedTimestep
        )

        // Process through transformer blocks
        let blocksStart = Date()
        for (i, block) in transformerBlocks.enumerated() {
            let blockStart = Date()

            args = block(args)

            // Evaluate periodically to avoid Metal GPU timeout from accumulated graph
            if (i + 1) % evalFrequency == 0 || i == transformerBlocks.count - 1 {
                eval(args.x)
            }

            if i < 4 || i % 8 == 0 || i == transformerBlocks.count - 1 {
                let blockTime = Date().timeIntervalSince(blockStart)
                let xMean = args.x.mean().item(Float.self)
                let hasNaN = MLX.any(MLX.isNaN(args.x)).item(Bool.self)
                LTXDebug.log("  [TIME] block \(i): \(String(format: "%.3f", blockTime))s, mean=\(xMean), nan=\(hasNaN)")
            }
        }
        now = Date()
        LTXDebug.log("  [TIME] all transformer blocks: \(String(format: "%.3f", now.timeIntervalSince(blocksStart)))s")
        lastTime = now

        // Process output
        let output = processOutput(args.x, embeddedTimestep: embeddedTimestep)
        eval(output)
        now = Date()
        LTXDebug.log("  [TIME] processOutput: \(String(format: "%.3f", now.timeIntervalSince(lastTime)))s")
        LTXDebug.log("  [TIME] TOTAL transformer forward: \(String(format: "%.3f", now.timeIntervalSince(startTime)))s")

        return output
    }
}

// MARK: - Convenience Initializers

extension LTXTransformer {
    /// Create transformer for a specific model variant
    public convenience init(model: LTXModel, evalFrequency: Int = 4) {
        self.init(
            config: model.transformerConfig,
            ropeType: .split,  // LTX-2 distilled uses SPLIT
            evalFrequency: evalFrequency
        )
    }
}
