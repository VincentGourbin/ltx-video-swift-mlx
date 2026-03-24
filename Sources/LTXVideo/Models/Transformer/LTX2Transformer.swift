// LTX2Transformer.swift - LTX-2 Dual Video/Audio Transformer Model
// Implements the LTX2VideoTransformer3DModel from HuggingFace Diffusers
// Copyright 2025

import Foundation
@preconcurrency import MLX
import MLXFast
import MLXNN

// MARK: - LTX2 Transformer Model

/// LTX-2 Dual Video/Audio Transformer Model
///
/// Extends the video-only LTXTransformer with:
/// - Audio input/output projections
/// - Audio timestep embeddings
/// - Audio caption projection
/// - Cross-modal attention scale/shift embeddings
/// - 1D audio RoPE
///
/// Weight keys match Python Diffusers `LTX2VideoTransformer3DModel`.
class LTX2Transformer: Module {
    /// Number of leading transformer blocks that are frozen (no LoRA).
    /// When > 0, stopGradient is inserted after block `frozenBlockCount - 1`
    /// to prevent backward graph from tracing into frozen blocks.
    var frozenBlockCount: Int = 0

    let config: LTXTransformerConfig
    let ropeType: LTXRopeType
    let normEps: Float
    let memoryOptimization: MemoryOptimizationConfig

    // --- Video modules (same keys as LTXTransformer) ---
    @ModuleInfo(key: "patchify_proj") var patchifyProj: Linear
    @ModuleInfo(key: "adaln_single") var adalnSingle: AdaLayerNormSingle
    @ModuleInfo(key: "caption_projection") var captionProjection: PixArtAlphaTextProjection?
    @ModuleInfo(key: "norm_out") var normOut: LayerNorm
    @ModuleInfo(key: "proj_out") var projOut: Linear
    @ParameterInfo(key: "scale_shift_table") var scaleShiftTable: MLXArray

    // --- Audio modules ---
    @ModuleInfo(key: "audio_patchify_proj") var audioProjIn: Linear
    @ModuleInfo(key: "audio_proj_out") var audioProjOut: Linear
    @ModuleInfo(key: "audio_norm_out") var audioNormOut: LayerNorm
    @ModuleInfo(key: "audio_adaln_single") var audioTimeEmbed: AdaLayerNormSingle
    @ModuleInfo(key: "audio_caption_projection") var audioCaptionProjection: PixArtAlphaTextProjection?
    @ParameterInfo(key: "audio_scale_shift_table") var audioScaleShiftTable: MLXArray

    // --- Cross-modal timestep embeddings ---
    @ModuleInfo(key: "av_ca_video_scale_shift_adaln_single") var avCrossAttnVideoScaleShift: AdaLayerNormSingle
    @ModuleInfo(key: "av_ca_a2v_gate_adaln_single") var avCrossAttnVideoA2VGate: AdaLayerNormSingle
    @ModuleInfo(key: "av_ca_audio_scale_shift_adaln_single") var avCrossAttnAudioScaleShift: AdaLayerNormSingle
    @ModuleInfo(key: "av_ca_v2a_gate_adaln_single") var avCrossAttnAudioV2AGate: AdaLayerNormSingle

    // --- Prompt AdaLN for cross-attention (LTX-2.3) ---
    @ModuleInfo(key: "prompt_adaln_single") var promptAdalnSingle: AdaLayerNormSingle?
    @ModuleInfo(key: "audio_prompt_adaln_single") var audioPromptAdalnSingle: AdaLayerNormSingle?

    // --- Dual video/audio transformer blocks ---
    @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [LTX2TransformerBlock]

    // --- Cached RoPE ---
    private var cachedVideoRoPE: (cos: MLXArray, sin: MLXArray)?
    private var cachedVideoRoPEKey: String?
    private var cachedAudioRoPE: (cos: MLXArray, sin: MLXArray)?
    private var cachedAudioRoPEKey: String?

    init(
        config: LTXTransformerConfig = .default,
        ropeType: LTXRopeType = .split,
        memoryOptimization: MemoryOptimizationConfig = .default
    ) {
        self.config = config
        self.ropeType = ropeType
        self.normEps = config.normEps
        self.memoryOptimization = memoryOptimization

        let videoDim = config.innerDim
        let audioDim = config.audioInnerDim
        let gated = config.gatedAttention
        let crossAttnAdaLN = config.crossAttentionAdaLN
        let numEmb = crossAttnAdaLN ? 9 : 6

        // --- Video ---
        self._patchifyProj.wrappedValue = Linear(config.inChannels, videoDim, bias: true)
        self._adalnSingle.wrappedValue = AdaLayerNormSingle(innerDim: videoDim, numEmbeddings: numEmb)
        if !config.captionProjBeforeConnector && config.captionChannels != videoDim {
            self._captionProjection.wrappedValue = PixArtAlphaTextProjection(
                inFeatures: config.captionChannels, hiddenSize: videoDim
            )
        }
        self._normOut.wrappedValue = LayerNorm(dimensions: videoDim, eps: config.normEps, affine: false)
        self._projOut.wrappedValue = Linear(videoDim, config.outChannels)
        self._scaleShiftTable.wrappedValue = MLXArray.zeros([2, videoDim])

        // --- Audio ---
        self._audioProjIn.wrappedValue = Linear(config.audioInChannels, audioDim, bias: true)
        self._audioProjOut.wrappedValue = Linear(audioDim, config.audioOutChannels)
        self._audioNormOut.wrappedValue = LayerNorm(dimensions: audioDim, eps: config.normEps, affine: false)
        self._audioTimeEmbed.wrappedValue = AdaLayerNormSingle(innerDim: audioDim, numEmbeddings: numEmb)
        if !config.captionProjBeforeConnector && config.captionChannels != audioDim {
            self._audioCaptionProjection.wrappedValue = PixArtAlphaTextProjection(
                inFeatures: config.captionChannels, hiddenSize: audioDim
            )
        }
        self._audioScaleShiftTable.wrappedValue = MLXArray.zeros([2, audioDim])

        // --- Cross-modal timestep embeddings ---
        self._avCrossAttnVideoScaleShift.wrappedValue = AdaLayerNormSingle(
            innerDim: videoDim, numEmbeddings: 4
        )
        self._avCrossAttnVideoA2VGate.wrappedValue = AdaLayerNormSingle(
            innerDim: videoDim, numEmbeddings: 1
        )
        self._avCrossAttnAudioScaleShift.wrappedValue = AdaLayerNormSingle(
            innerDim: audioDim, numEmbeddings: 4
        )
        self._avCrossAttnAudioV2AGate.wrappedValue = AdaLayerNormSingle(
            innerDim: audioDim, numEmbeddings: 1
        )

        // --- Prompt AdaLN (LTX-2.3) ---
        if crossAttnAdaLN {
            self._promptAdalnSingle.wrappedValue = AdaLayerNormSingle(innerDim: videoDim, numEmbeddings: 2)
            self._audioPromptAdalnSingle.wrappedValue = AdaLayerNormSingle(innerDim: audioDim, numEmbeddings: 2)
        }

        // --- Transformer blocks ---
        self._transformerBlocks.wrappedValue = (0..<config.numLayers).map { _ in
            LTX2TransformerBlock(
                videoDim: videoDim,
                videoNumHeads: config.numAttentionHeads,
                videoHeadDim: config.attentionHeadDim,
                videoCrossAttentionDim: config.crossAttentionDim,
                audioDim: audioDim,
                audioNumHeads: config.audioNumAttentionHeads,
                audioHeadDim: config.audioAttentionHeadDim,
                audioCrossAttentionDim: config.audioCrossAttentionDim,
                ropeType: ropeType,
                normEps: config.normEps,
                gatedAttention: gated,
                crossAttentionAdaLN: crossAttnAdaLN
            )
        }
    }

    /// Clear cached RoPE embeddings
    func clearRoPECache() {
        cachedVideoRoPE = nil
        cachedVideoRoPEKey = nil
        cachedAudioRoPE = nil
        cachedAudioRoPEKey = nil
    }

    // MARK: - RoPE Preparation

    /// Prepare video RoPE (3D: frame, height, width) with caching
    private func prepareVideoRoPE(
        batchSize: Int,
        frames: Int,
        height: Int,
        width: Int
    ) -> (cos: MLXArray, sin: MLXArray) {
        let cacheKey = "\(batchSize)_\(frames)_\(height)_\(width)"
        if let cached = cachedVideoRoPE, cachedVideoRoPEKey == cacheKey {
            return cached
        }

        let positions = createPositionGrid(
            batchSize: batchSize, frames: frames, height: height, width: width
        )
        let result = precomputeFreqsCis(
            indicesGrid: positions,
            dim: config.innerDim,
            theta: config.ropeTheta,
            maxPos: config.maxPos,
            numAttentionHeads: config.numAttentionHeads,
            ropeType: ropeType,
            doublePrecision: true
        )
        cachedVideoRoPE = result
        cachedVideoRoPEKey = cacheKey
        return result
    }

    /// Prepare audio RoPE (1D temporal)
    private func prepareAudioRoPE(
        batchSize: Int,
        audioFrames: Int
    ) -> (cos: MLXArray, sin: MLXArray) {
        let cacheKey = "\(batchSize)_\(audioFrames)"
        if let cached = cachedAudioRoPE, cachedAudioRoPEKey == cacheKey {
            return cached
        }

        // Audio positions are 1D: just temporal indices
        let positions = createAudioPositionGrid(
            batchSize: batchSize, audioFrames: audioFrames
        )
        let result = precomputeFreqsCis(
            indicesGrid: positions,
            dim: config.audioInnerDim,
            theta: config.ropeTheta,
            maxPos: config.audioMaxPos,
            numAttentionHeads: config.audioNumAttentionHeads,
            ropeType: ropeType,
            doublePrecision: true
        )
        cachedAudioRoPE = result
        cachedAudioRoPEKey = cacheKey
        return result
    }

    /// Prepare cross-modal RoPE for cross-attention
    ///
    /// Python uses temporal-only coordinates for cross-modal attention:
    ///   `video_coords[:, 0:1, :]` and `audio_coords[:, 0:1, :]`
    /// This makes physical sense: cross-modal attention aligns video and audio
    /// on time, not spatial position.
    private func prepareCrossModalRoPE(
        batchSize: Int,
        videoFrames: Int,
        videoHeight: Int,
        videoWidth: Int,
        audioFrames: Int
    ) -> (video: (cos: MLXArray, sin: MLXArray), audio: (cos: MLXArray, sin: MLXArray)) {
        // Video side: temporal-only (1D) RoPE for cross-attention
        // Python: self.cross_attn_rope(video_coords[:, 0:1, :])
        let videoPositions3D = createPositionGrid(
            batchSize: batchSize, frames: videoFrames, height: videoHeight, width: videoWidth
        )
        // Extract temporal coordinate only: (B, 3, T) → (B, 1, T)
        let videoTemporalOnly = videoPositions3D[0..., 0..<1, 0...]
        let videoCrossRoPE = precomputeFreqsCis(
            indicesGrid: videoTemporalOnly,
            dim: config.audioCrossAttentionDim,
            theta: config.ropeTheta,
            maxPos: config.audioMaxPos,  // Temporal only → use audioMaxPos (frame-based)
            numAttentionHeads: config.audioNumAttentionHeads,
            ropeType: ropeType,
            doublePrecision: true
        )

        // Audio side: temporal-only (1D) RoPE for cross-attention
        // Python: self.cross_attn_audio_rope(audio_coords[:, 0:1, :])
        let audioPositions = createAudioPositionGrid(
            batchSize: batchSize, audioFrames: audioFrames
        )
        let audioCrossRoPE = precomputeFreqsCis(
            indicesGrid: audioPositions,
            dim: config.audioCrossAttentionDim,
            theta: config.ropeTheta,
            maxPos: config.audioMaxPos,
            numAttentionHeads: config.audioNumAttentionHeads,
            ropeType: ropeType,
            doublePrecision: true
        )

        return (videoCrossRoPE, audioCrossRoPE)
    }

    // MARK: - Forward Pass

    /// Dual video/audio forward pass
    ///
    /// - Parameters:
    ///   - videoLatent: Patchified video latents (B, T_video, C)
    ///   - audioLatent: Packed audio latents (B, T_audio, C_audio)
    ///   - videoContext: Video text embeddings (B, S, D_text)
    ///   - audioContext: Audio text embeddings (B, S, D_text)
    ///   - videoTimesteps: Video timestep values (B,)
    ///   - audioTimesteps: Audio timestep values (B,)
    ///   - videoContextMask: Optional video text attention mask (B, S)
    ///   - audioContextMask: Optional audio text attention mask (B, S)
    ///   - videoLatentShape: Shape of video latent (frames, height, width)
    ///   - audioNumFrames: Number of audio latent frames
    /// - Returns: (videoOutput, audioOutput) velocity predictions
    func callAsFunction(
        videoLatent: MLXArray,
        audioLatent: MLXArray,
        videoContext: MLXArray,
        audioContext: MLXArray,
        videoTimesteps: MLXArray,
        audioTimesteps: MLXArray,
        videoContextMask: MLXArray? = nil,
        audioContextMask: MLXArray? = nil,
        videoLatentShape: (frames: Int, height: Int, width: Int),
        audioNumFrames: Int
    ) -> (video: MLXArray, audio: MLXArray) {
        let batchSize = videoLatent.dim(0)
        let videoDim = config.innerDim
        let audioDim = config.audioInnerDim

        let numEmb = config.crossAttentionAdaLN ? 9 : 6

        // --- Video preparation ---
        let videoX = patchifyProj(videoLatent)
        let scaledVideoTs = videoTimesteps * Float(config.timestepScaleMultiplier)
        let (videoTemb, videoEmbeddedTs) = adalnSingle(scaledVideoTs.flattened())
        let videoTembReshaped = videoTemb.reshaped([batchSize, -1, numEmb, videoDim])
        let projectedVideoCtx: MLXArray
        if let proj = captionProjection {
            projectedVideoCtx = proj(videoContext).reshaped([batchSize, -1, videoDim])
        } else {
            projectedVideoCtx = videoContext.reshaped([batchSize, -1, videoDim])
        }

        // --- Audio preparation ---
        let audioX = audioProjIn(audioLatent)
        let scaledAudioTs = audioTimesteps * Float(config.timestepScaleMultiplier)
        let (audioTemb, audioEmbeddedTs) = audioTimeEmbed(scaledAudioTs.flattened())
        let audioTembReshaped = audioTemb.reshaped([batchSize, -1, numEmb, audioDim])
        let projectedAudioCtx: MLXArray
        if let proj = audioCaptionProjection {
            projectedAudioCtx = proj(audioContext).reshaped([batchSize, -1, audioDim])
        } else {
            projectedAudioCtx = audioContext.reshaped([batchSize, -1, audioDim])
        }

        // --- Prompt AdaLN for cross-attention (LTX-2.3) ---
        // Prompt AdaLN modulates text context, so use scalar timestep
        // even when video uses per-token timesteps (I2V conditioning mask)
        var videoPromptTs: MLXArray? = nil
        var audioPromptTs: MLXArray? = nil
        if let promptAdaln = promptAdalnSingle {
            let scalarVideoTs: MLXArray
            if videoTimesteps.ndim > 1 {
                scalarVideoTs = videoTimesteps.max(axis: 1)  // (B,)
            } else {
                scalarVideoTs = videoTimesteps  // Already scalar (B,)
            }
            let scaledTs = scalarVideoTs * Float(config.timestepScaleMultiplier)
            let (pEmb, _) = promptAdaln(scaledTs.flattened())
            videoPromptTs = pEmb.reshaped([batchSize, -1, 2, videoDim])
        }
        if let audioPromptAdaln = audioPromptAdalnSingle {
            let scalarAudioTs: MLXArray
            if audioTimesteps.ndim > 1 {
                scalarAudioTs = audioTimesteps.max(axis: 1)
            } else {
                scalarAudioTs = audioTimesteps
            }
            let scaledTs = scalarAudioTs * Float(config.timestepScaleMultiplier)
            let (apEmb, _) = audioPromptAdaln(scaledTs.flattened())
            audioPromptTs = apEmb.reshaped([batchSize, -1, 2, audioDim])
        }

        // --- Cross-modal timestep embeddings ---
        // Python Diffusers uses per-token timesteps for cross-modal (not scalar).
        // In I2V mode, videoTimesteps is (B, T) where frame 0 = 0, others = sigma.
        // scaledVideoTs = videoTimesteps * 1000, so flattened gives (B*T,) with per-token values.
        // Each token gets its own cross-modal modulation via AdaLN.
        // Gate factor = cross_attn_timestep_scale_multiplier / timestep_scale_multiplier = 1000/1000 = 1.0
        let flatVideoTs = scaledVideoTs.flattened()

        let (crossVideoSSEmb, _) = avCrossAttnVideoScaleShift(flatVideoTs)
        let crossVideoSSReshaped = crossVideoSSEmb.reshaped([batchSize, -1, 4, videoDim])

        let (crossVideoGateEmb, _) = avCrossAttnVideoA2VGate(flatVideoTs)

        // Concatenate scale/shift (4) + gate (1) = 5 values to match per-block SST shape
        let crossVideoSSFull = MLX.concatenated([
            crossVideoSSReshaped,
            crossVideoGateEmb.reshaped([batchSize, -1, 1, videoDim])
        ], axis: 2)

        let flatAudioTs = scaledAudioTs.flattened()
        let (crossAudioSSEmb, _) = avCrossAttnAudioScaleShift(flatAudioTs)
        let crossAudioSSReshaped = crossAudioSSEmb.reshaped([batchSize, -1, 4, audioDim])

        let (crossAudioGateEmb, _) = avCrossAttnAudioV2AGate(flatAudioTs)

        let crossAudioSSFull = MLX.concatenated([
            crossAudioSSReshaped,
            crossAudioGateEmb.reshaped([batchSize, -1, 1, audioDim])
        ], axis: 2)

        // --- Prepare attention masks ---
        let preparedVideoMask = prepareAttentionMask(videoContextMask)
        let preparedAudioMask = prepareAttentionMask(audioContextMask)

        // --- Prepare RoPE ---
        let videoRoPE = prepareVideoRoPE(
            batchSize: batchSize,
            frames: videoLatentShape.frames,
            height: videoLatentShape.height,
            width: videoLatentShape.width
        )
        let audioRoPE = prepareAudioRoPE(batchSize: batchSize, audioFrames: audioNumFrames)

        let (crossVideoRoPE, crossAudioRoPE) = prepareCrossModalRoPE(
            batchSize: batchSize,
            videoFrames: videoLatentShape.frames,
            videoHeight: videoLatentShape.height,
            videoWidth: videoLatentShape.width,
            audioFrames: audioNumFrames
        )

        eval(videoX, audioX, videoTembReshaped, audioTembReshaped)

        // --- Create args ---
        var videoArgs = TransformerArgs(
            x: videoX,
            context: projectedVideoCtx,
            timesteps: videoTembReshaped,
            positionalEmbeddings: videoRoPE,
            contextMask: preparedVideoMask,
            embeddedTimestep: videoEmbeddedTs,
            promptTimesteps: videoPromptTs
        )

        var audioArgs = AudioTransformerArgs(
            x: audioX,
            context: projectedAudioCtx,
            timesteps: audioTembReshaped,
            positionalEmbeddings: audioRoPE,
            contextMask: preparedAudioMask,
            embeddedTimestep: audioEmbeddedTs,
            crossVideoScaleShift: crossVideoSSFull,
            crossAudioScaleShift: crossAudioSSFull,
            crossVideoRoPE: crossVideoRoPE,
            crossAudioRoPE: crossAudioRoPE,
            videoPromptTimesteps: videoPromptTs,
            audioPromptTimesteps: audioPromptTs
        )

        // --- Process through transformer blocks ---
        for (i, block) in transformerBlocks.enumerated() {
                (videoArgs, audioArgs) = block(videoArgs, audio: audioArgs)

                // Cut backward graph after last frozen block (selective LoRA)
                if frozenBlockCount > 0 && i == frozenBlockCount - 1 {
                    videoArgs.x = stopGradient(videoArgs.x)
                    audioArgs.x = stopGradient(audioArgs.x)
                    eval(videoArgs.x, audioArgs.x)
                    Memory.clearCache()
                }

                if memoryOptimization.evalFrequency > 0
                    && (i + 1) % memoryOptimization.evalFrequency == 0 {
                    eval(videoArgs.x, audioArgs.x)
                    if memoryOptimization.clearCacheOnEval {
                        Memory.clearCache()
                    }
                }
            }
        eval(videoArgs.x, audioArgs.x)

        // --- Video output ---
        let videoSSOut = scaleShiftTable.reshaped([1, 1, 2, videoDim])
            + videoEmbeddedTs.reshaped([batchSize, -1, 1, videoDim])
        let videoShiftOut = videoSSOut[0..., 0..., 0, 0...]
        let videoScaleOut = videoSSOut[0..., 0..., 1, 0...]
        var videoOutput = normOut(videoArgs.x) * (1 + videoScaleOut) + videoShiftOut
        videoOutput = projOut(videoOutput)

        // --- Audio output ---
        let audioSSOut = audioScaleShiftTable.reshaped([1, 1, 2, audioDim])
            + audioEmbeddedTs.reshaped([batchSize, -1, 1, audioDim])
        let audioShiftOut = audioSSOut[0..., 0..., 0, 0...]
        let audioScaleOut = audioSSOut[0..., 0..., 1, 0...]
        var audioOutput = audioNormOut(audioArgs.x) * (1 + audioScaleOut) + audioShiftOut
        audioOutput = audioProjOut(audioOutput)

        return (videoOutput, audioOutput)
    }

    /// Set/clear STG video self-attention skip on specified blocks
    func setSTGBlocks(_ blocks: [Int]) {
        for (i, block) in transformerBlocks.enumerated() {
            block.skipVideoSelfAttention = blocks.contains(i)
        }
    }

    /// Clear all STG perturbation flags
    func clearSTG() {
        for block in transformerBlocks {
            block.skipVideoSelfAttention = false
        }
    }

    // MARK: - Helpers

    private func prepareAttentionMask(_ mask: MLXArray?) -> MLXArray? {
        guard let mask = mask else { return nil }
        if mask.dtype == .float16 || mask.dtype == .float32 || mask.dtype == .bfloat16 {
            return mask
        }
        let floatMask = (1 - mask.asType(.float32)) * Float(-10000.0)
        return floatMask.reshaped([mask.dim(0), 1, 1, mask.dim(-1)])
    }
}
