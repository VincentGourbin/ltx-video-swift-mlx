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
    /// Set to true to dump detailed intermediates on the next forward pass (step 0 diagnostics)
    nonisolated(unsafe) public static var dumpNextForwardPass = false

    let config: LTXTransformerConfig
    let ropeType: LTXRopeType
    let normEps: Float
    let memoryOptimization: MemoryOptimizationConfig

    /// Cached RoPE embeddings — reused across forward passes when dimensions don't change
    private var cachedRoPE: (cos: MLXArray, sin: MLXArray)?
    private var cachedRoPEKey: String?

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
        memoryOptimization: MemoryOptimizationConfig = .default,
        evalFrequency: Int? = nil
    ) {
        self.config = config
        self.ropeType = ropeType
        self.normEps = config.normEps
        // evalFrequency parameter is a legacy convenience — it overrides the config value
        if let freq = evalFrequency {
            var opt = memoryOptimization
            opt.evalFrequency = freq
            self.memoryOptimization = opt
        } else {
            self.memoryOptimization = memoryOptimization
        }

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
    ///
    /// Matches Diffusers: `(1 - mask) * -10000.0` then unsqueeze
    /// Input mask: (B, S) with 1=attend, 0=pad
    /// Output: (B, 1, 1, S) additive bias with 0=attend, -10000=pad
    private func prepareAttentionMask(
        _ mask: MLXArray?
    ) -> MLXArray? {
        guard let mask = mask else { return nil }

        // If already a float mask, return as-is
        if mask.dtype == .float16 || mask.dtype == .float32 || mask.dtype == .bfloat16 {
            return mask
        }

        // Convert boolean/int mask to additive mask matching Diffusers
        // True/1 = attend (0.0), False/0 = don't attend (-10000.0)
        let floatMask = (1 - mask.asType(.float32)) * Float(-10000.0)
        let reshapedMask = floatMask.reshaped([mask.dim(0), 1, 1, mask.dim(-1)])
        return reshapedMask
    }

    /// Prepare positional embeddings (RoPE) with caching
    ///
    /// Positions depend only on (B, F, H, W) which are constant during denoising,
    /// so we cache the result after the first computation (matching Python's precompute_freqs_cis).
    private func preparePositionalEmbeddings(
        batchSize: Int,
        frames: Int,
        height: Int,
        width: Int
    ) -> (cos: MLXArray, sin: MLXArray) {
        let cacheKey = "\(batchSize)_\(frames)_\(height)_\(width)"
        if let cached = cachedRoPE, cachedRoPEKey == cacheKey {
            LTXDebug.log("  [DIAG] RoPE cache hit (\(cacheKey))")
            return cached
        }

        // Create pixel-space position grid (matching Python's get_pixel_coords pipeline)
        let positions = createPositionGrid(
            batchSize: batchSize,
            frames: frames,
            height: height,
            width: width
        )

        // Precompute frequencies (double precision matching Python mlx-video)
        // Python: double_precision_rope=True in convert.py — computes frequencies in float64
        // for accurate high-frequency RoPE components that compound over 40+ denoising steps
        let result = precomputeFreqsCis(
            indicesGrid: positions,
            dim: config.innerDim,
            theta: config.ropeTheta,
            maxPos: config.maxPos,
            numAttentionHeads: config.numAttentionHeads,
            ropeType: ropeType,
            doublePrecision: true
        )

        // Cache for reuse across forward passes
        cachedRoPE = result
        cachedRoPEKey = cacheKey
        return result
    }

    /// Clear the RoPE cache (e.g., when switching resolution between stages)
    public func clearRoPECache() {
        cachedRoPE = nil
        cachedRoPEKey = nil
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

        let dumpMode = LTXTransformer.dumpNextForwardPass

        // Project latents to inner dimension
        var x = patchifyProj(latent)
        eval(x)
        var now = Date()
        LTXDebug.log("  [TIME] patchifyProj: \(String(format: "%.3f", now.timeIntervalSince(lastTime)))s")
        lastTime = now
        LTXDebug.log("  after patchifyProj: \(x.shape)")

        if dumpMode {
            let slice0 = x[0, 0, ..<5].asType(.float32)
            let slice640 = x[0, 640, ..<5].asType(.float32)
            eval(slice0, slice640)
            print("[DUMP] patchifyProj [0,0,:5]=\(slice0.asArray(Float.self))")
            print("[DUMP] patchifyProj [0,640,:5]=\(slice640.asArray(Float.self))")
            print("[DUMP] patchifyProj mean=\(x.mean().item(Float.self))")
        }

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

        if dumpMode {
            // timestepEmb is (B, T, 6, D) after reshape in prepareTimestep
            // But we also want the raw emb before reshape — get from adalnSingle directly
            let scaledTs = timesteps * Float(config.timestepScaleMultiplier)
            let (rawEmb, rawEmbedded) = adalnSingle(scaledTs.flattened())
            eval(rawEmb, rawEmbedded)
            let embSlice = rawEmb[0, ..<5].asType(.float32)
            let embTsSlice = rawEmbedded[0, ..<5].asType(.float32)
            eval(embSlice, embTsSlice)
            print("[DUMP] scaled_timestep=\((timesteps * Float(config.timestepScaleMultiplier)).item(Float.self))")
            print("[DUMP] timestep_emb shape=\(rawEmb.shape), mean=\(rawEmb.mean().item(Float.self))")
            print("[DUMP] timestep_emb[:5]=\(embSlice.asArray(Float.self))")
            print("[DUMP] embedded_timestep shape=\(rawEmbedded.shape), mean=\(rawEmbedded.mean().item(Float.self))")
            print("[DUMP] embedded_timestep[:5]=\(embTsSlice.asArray(Float.self))")
            print("[DUMP] timestepEmb reshaped: \(timestepEmb.shape)")
            print("[DUMP] embeddedTimestep reshaped: \(embeddedTimestep.shape)")
        }

        // Prepare context (caption projection)
        let projectedContext = prepareContext(context: context, batchSize: batchSize)
        eval(projectedContext)
        now = Date()
        LTXDebug.log("  [TIME] prepareContext: \(String(format: "%.3f", now.timeIntervalSince(lastTime)))s")
        lastTime = now

        if dumpMode {
            let ctxSlice0 = projectedContext[0, 0, ..<5].asType(.float32)
            let ctxSlice512 = projectedContext[0, 512, ..<5].asType(.float32)
            eval(ctxSlice0, ctxSlice512)
            print("[DUMP] caption_proj shape=\(projectedContext.shape), mean=\(projectedContext.mean().item(Float.self))")
            print("[DUMP] caption_proj [0,0,:5]=\(ctxSlice0.asArray(Float.self))")
            print("[DUMP] caption_proj [0,512,:5]=\(ctxSlice512.asArray(Float.self))")
        }

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

        if dumpMode {
            let cosH0 = pe.cos[0, 0, 0, ..<5].asType(.float32)
            let cosH0Tail = pe.cos[0, 0, 0, (-5)...].asType(.float32)
            let cosH16 = pe.cos[0, 16, 0, ..<5].asType(.float32)
            let sinH0 = pe.sin[0, 0, 0, ..<5].asType(.float32)
            let sinH0Tail = pe.sin[0, 0, 0, (-5)...].asType(.float32)
            eval(cosH0, cosH0Tail, cosH16, sinH0, sinH0Tail)
            print("[DUMP] RoPE cos [0,0,0,:5]=\(cosH0.asArray(Float.self))")
            print("[DUMP] RoPE cos [0,0,0,-5:]=\(cosH0Tail.asArray(Float.self))")
            print("[DUMP] RoPE cos [0,16,0,:5]=\(cosH16.asArray(Float.self))")
            print("[DUMP] RoPE sin [0,0,0,:5]=\(sinH0.asArray(Float.self))")
            print("[DUMP] RoPE sin [0,0,0,-5:]=\(sinH0Tail.asArray(Float.self))")
        }

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

        // If dumpMode, manually trace block 0 first
        if dumpMode {
            let block0 = transformerBlocks[0]
            let sst = block0.scaleShiftTable
            eval(sst)
            print("[DUMP] block0 SST shape=\(sst.shape), dtype=\(sst.dtype)")
            print("[DUMP] block0 SST [0,:5]=\(sst[0, ..<5].asType(.float32).asArray(Float.self))")
            print("[DUMP] block0 SST [3,:5]=\(sst[3, ..<5].asType(.float32).asArray(Float.self))")
            print("[DUMP] block0 SST mean=\(sst.mean().item(Float.self))")

            // Compute AdaLN values for self-attention (indices 0,1,2)
            let tableSliceAttn = sst[0..<3]  // (3, D)
            let tableExpAttn = tableSliceAttn.reshaped([1, 1, 3, -1])
            let tsSliceAttn = args.timesteps[0..., 0..., 0..<3, 0...]
            let adaAttn = tableExpAttn + tsSliceAttn
            eval(adaAttn)

            let shiftMSA = adaAttn[0..., 0..., 0, 0...]
            let scaleMSA = adaAttn[0..., 0..., 1, 0...]
            let gateMSA = adaAttn[0..., 0..., 2, 0...]
            eval(shiftMSA, scaleMSA, gateMSA)
            print("[DUMP] block0 shift_msa mean=\(shiftMSA.mean().item(Float.self)), [0,0,:5]=\(shiftMSA[0, 0, ..<5].asType(.float32).asArray(Float.self))")
            print("[DUMP] block0 scale_msa mean=\(scaleMSA.mean().item(Float.self)), [0,0,:5]=\(scaleMSA[0, 0, ..<5].asType(.float32).asArray(Float.self))")
            print("[DUMP] block0 gate_msa mean=\(gateMSA.mean().item(Float.self)), [0,0,:5]=\(gateMSA[0, 0, ..<5].asType(.float32).asArray(Float.self))")

            // Apply RMSNorm + AdaLN
            let weight0 = MLXArray.ones([x.dim(-1)]).asType(x.dtype)
            let normedX = MLXFast.rmsNorm(x, weight: weight0, eps: normEps)
            let adalNormed = normedX * (1 + scaleMSA) + shiftMSA
            eval(adalNormed)
            print("[DUMP] block0 after AdaLN norm mean=\(adalNormed.mean().item(Float.self)), [0,0,:5]=\(adalNormed[0, 0, ..<5].asType(.float32).asArray(Float.self))")

            // Self-attention
            let selfAttnOut = block0.attn1(adalNormed, pe: args.positionalEmbeddings)
            eval(selfAttnOut)
            print("[DUMP] block0 self-attn output mean=\(selfAttnOut.mean().item(Float.self)), [0,0,:5]=\(selfAttnOut[0, 0, ..<5].asType(.float32).asArray(Float.self))")

            let xAfterAttn1 = x + selfAttnOut * gateMSA
            eval(xAfterAttn1)
            print("[DUMP] block0 after self-attn residual mean=\(xAfterAttn1.mean().item(Float.self))")

            // Cross-attention (no pre-norm, matching Diffusers)
            let crossOut = block0.attn2(xAfterAttn1, context: args.context, mask: args.contextMask)
            eval(crossOut)
            print("[DUMP] block0 cross-attn output mean=\(crossOut.mean().item(Float.self)), [0,0,:5]=\(crossOut[0, 0, ..<5].asType(.float32).asArray(Float.self))")

            let xAfterAttn2 = xAfterAttn1 + crossOut
            eval(xAfterAttn2)
            print("[DUMP] block0 after cross-attn residual mean=\(xAfterAttn2.mean().item(Float.self))")

            // FFN AdaLN
            let tableSliceFF = sst[3..<6]
            let tableExpFF = tableSliceFF.reshaped([1, 1, 3, -1])
            let tsSliceFF = args.timesteps[0..., 0..., 3..<6, 0...]
            let adaFF = tableExpFF + tsSliceFF
            eval(adaFF)

            let shiftMLP = adaFF[0..., 0..., 0, 0...]
            let scaleMLP = adaFF[0..., 0..., 1, 0...]
            let gateMLP = adaFF[0..., 0..., 2, 0...]

            let normedFF = MLXFast.rmsNorm(xAfterAttn2, weight: MLXArray.ones([xAfterAttn2.dim(-1)]).asType(xAfterAttn2.dtype), eps: normEps)
            let ffInput = normedFF * (1 + scaleMLP) + shiftMLP
            let ffOut = block0.ff(ffInput)
            eval(ffOut)
            print("[DUMP] block0 FFN output mean=\(ffOut.mean().item(Float.self)), [0,0,:5]=\(ffOut[0, 0, ..<5].asType(.float32).asArray(Float.self))")

            let xAfterFF = xAfterAttn2 + ffOut * gateMLP
            eval(xAfterFF)
            print("[DUMP] block0 output mean=\(xAfterFF.mean().item(Float.self)), [0,0,:5]=\(xAfterFF[0, 0, ..<5].asType(.float32).asArray(Float.self))")

            // Now run block 0 normally through the standard path to update args
            args = block0(args)
            eval(args.x)

            // Verify the normal block 0 output matches our manual trace
            let block0Mean = args.x.mean().item(Float.self)
            print("[DUMP] block0 (via normal path) mean=\(block0Mean)")
            print("[BLOCK_MEAN] block 0: mean=\(block0Mean)")

            // Now turn off dump mode and skip block 0 in the normal loop
            LTXTransformer.dumpNextForwardPass = false
        }

        for (i, block) in transformerBlocks.enumerated() {
            // Skip block 0 if we already traced it in dump mode
            if dumpMode && i == 0 { continue }

            args = block(args)

            if dumpMode {
                // In dump mode: eval every block for accurate per-block means
                eval(args.x)
                let xMean = args.x.mean().item(Float.self)
                print("[BLOCK_MEAN] block \(i): mean=\(xMean)")
            } else if i == transformerBlocks.count - 1 {
                // Match Python: only eval after the last block
                // Python evaluates the full 48-block graph at once
                eval(args.x)
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

// MARK: - Configuration Methods

extension LTXTransformer {
    /// Set cross-attention scaling for specified blocks
    ///
    /// - Parameters:
    ///   - scale: Cross-attention output multiplier (1.0 = no change)
    ///   - blocks: Range of block indices to apply to (nil = all blocks)
    public func setCrossAttentionScale(_ scale: Float, forBlocks blocks: ClosedRange<Int>? = nil) {
        let range = blocks ?? 0...(transformerBlocks.count - 1)
        for i in range {
            guard i >= 0 && i < transformerBlocks.count else { continue }
            transformerBlocks[i].crossAttentionScale = scale
        }
        LTXDebug.log("Set cross-attention scale to \(scale) for blocks \(range)")
    }

    /// Set STG skip flags on specified blocks
    ///
    /// - Parameters:
    ///   - skipSelfAttention: Whether to skip self-attention
    ///   - skipFeedForward: Whether to skip feed-forward
    ///   - blockIndices: Which block indices to modify
    public func setSTGSkipFlags(skipSelfAttention: Bool, skipFeedForward: Bool = false, blockIndices: [Int]) {
        for i in blockIndices {
            guard i >= 0 && i < transformerBlocks.count else { continue }
            transformerBlocks[i].skipSelfAttention = skipSelfAttention
            transformerBlocks[i].skipFeedForward = skipFeedForward
        }
    }

    /// Reset all STG skip flags to false
    public func clearSTGSkipFlags() {
        for block in transformerBlocks {
            block.skipSelfAttention = false
            block.skipFeedForward = false
        }
    }
}

// MARK: - Convenience Initializers

extension LTXTransformer {
    /// Create transformer for a specific model variant
    public convenience init(model: LTXModel, memoryOptimization: MemoryOptimizationConfig = .default) {
        self.init(
            config: model.transformerConfig,
            ropeType: .split,
            memoryOptimization: memoryOptimization
        )
    }
}
