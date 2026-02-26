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
    let config: LTXTransformerConfig
    let ropeType: LTXRopeType
    let normEps: Float
    let memoryOptimization: MemoryOptimizationConfig

    // --- Video modules (same keys as LTXTransformer) ---
    @ModuleInfo(key: "patchify_proj") var patchifyProj: Linear
    @ModuleInfo(key: "adaln_single") var adalnSingle: AdaLayerNormSingle
    @ModuleInfo(key: "caption_projection") var captionProjection: PixArtAlphaTextProjection
    @ModuleInfo(key: "norm_out") var normOut: LayerNorm
    @ModuleInfo(key: "proj_out") var projOut: Linear
    @ParameterInfo(key: "scale_shift_table") var scaleShiftTable: MLXArray

    // --- Audio modules ---
    @ModuleInfo(key: "audio_proj_in") var audioProjIn: Linear
    @ModuleInfo(key: "audio_proj_out") var audioProjOut: Linear
    @ModuleInfo(key: "audio_norm_out") var audioNormOut: LayerNorm
    @ModuleInfo(key: "audio_time_embed") var audioTimeEmbed: AdaLayerNormSingle
    @ModuleInfo(key: "audio_caption_projection") var audioCaptionProjection: PixArtAlphaTextProjection
    @ParameterInfo(key: "audio_scale_shift_table") var audioScaleShiftTable: MLXArray

    // --- Cross-modal timestep embeddings ---
    @ModuleInfo(key: "av_cross_attn_video_scale_shift") var avCrossAttnVideoScaleShift: AdaLayerNormSingle
    @ModuleInfo(key: "av_cross_attn_video_a2v_gate") var avCrossAttnVideoA2VGate: AdaLayerNormSingle
    @ModuleInfo(key: "av_cross_attn_audio_scale_shift") var avCrossAttnAudioScaleShift: AdaLayerNormSingle
    @ModuleInfo(key: "av_cross_attn_audio_v2a_gate") var avCrossAttnAudioV2AGate: AdaLayerNormSingle

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

        // --- Video ---
        self._patchifyProj.wrappedValue = Linear(config.inChannels, videoDim, bias: true)
        self._adalnSingle.wrappedValue = AdaLayerNormSingle(innerDim: videoDim, numEmbeddings: 6)
        self._captionProjection.wrappedValue = PixArtAlphaTextProjection(
            inFeatures: config.captionChannels, hiddenSize: videoDim
        )
        self._normOut.wrappedValue = LayerNorm(dimensions: videoDim, eps: config.normEps, affine: false)
        self._projOut.wrappedValue = Linear(videoDim, config.outChannels)
        self._scaleShiftTable.wrappedValue = MLXArray.zeros([2, videoDim])

        // --- Audio ---
        self._audioProjIn.wrappedValue = Linear(config.audioInChannels, audioDim, bias: true)
        self._audioProjOut.wrappedValue = Linear(audioDim, config.audioOutChannels)
        self._audioNormOut.wrappedValue = LayerNorm(dimensions: audioDim, eps: config.normEps, affine: false)
        self._audioTimeEmbed.wrappedValue = AdaLayerNormSingle(innerDim: audioDim, numEmbeddings: 6)
        self._audioCaptionProjection.wrappedValue = PixArtAlphaTextProjection(
            inFeatures: config.captionChannels, hiddenSize: audioDim
        )
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
                normEps: config.normEps
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
    /// Video RoPE is reused; audio cross-attn RoPE uses audio cross-attention dim
    private func prepareCrossModalRoPE(
        batchSize: Int,
        videoFrames: Int,
        videoHeight: Int,
        videoWidth: Int,
        audioFrames: Int
    ) -> (video: (cos: MLXArray, sin: MLXArray), audio: (cos: MLXArray, sin: MLXArray)) {
        // Video side: same 3D RoPE but for the cross-attention dimension
        let videoPositions = createPositionGrid(
            batchSize: batchSize, frames: videoFrames, height: videoHeight, width: videoWidth
        )
        let videoCrossRoPE = precomputeFreqsCis(
            indicesGrid: videoPositions,
            dim: config.audioCrossAttentionDim,  // Cross-modal uses audio head dims
            theta: config.ropeTheta,
            maxPos: config.maxPos,
            numAttentionHeads: config.audioNumAttentionHeads,
            ropeType: ropeType,
            doublePrecision: true
        )

        // Audio side: 1D temporal RoPE for cross-attention dimension
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

        // --- Video preparation ---
        let videoX = patchifyProj(videoLatent)
        let scaledVideoTs = videoTimesteps * Float(config.timestepScaleMultiplier)
        let (videoTemb, videoEmbeddedTs) = adalnSingle(scaledVideoTs.flattened())
        let videoTembReshaped = videoTemb.reshaped([batchSize, -1, 6, videoDim])
        let projectedVideoCtx = captionProjection(videoContext).reshaped([batchSize, -1, videoDim])

        // --- Audio preparation ---
        let audioX = audioProjIn(audioLatent)
        let scaledAudioTs = audioTimesteps * Float(config.timestepScaleMultiplier)
        let (audioTemb, audioEmbeddedTs) = audioTimeEmbed(scaledAudioTs.flattened())
        let audioTembReshaped = audioTemb.reshaped([batchSize, -1, 6, audioDim])
        let projectedAudioCtx = audioCaptionProjection(audioContext).reshaped([batchSize, -1, audioDim])

        // --- Cross-modal timestep embeddings ---
        let (crossVideoSSEmb, _) = avCrossAttnVideoScaleShift(scaledVideoTs.flattened())
        let crossVideoSSReshaped = crossVideoSSEmb.reshaped([batchSize, -1, 4, videoDim])
        // Pad to 5 values (block SST has 5 entries, but global only provides 4 for scale/shift)
        let crossVideoSSPadded = MLX.concatenated([
            crossVideoSSReshaped,
            MLXArray.zeros([batchSize, 1, 1, videoDim])
        ], axis: 2)

        let (crossVideoGateEmb, _) = avCrossAttnVideoA2VGate(scaledVideoTs.flattened())
        let crossVideoGateReshaped = crossVideoGateEmb.reshaped([batchSize, -1, 1, videoDim])

        let (crossAudioSSEmb, _) = avCrossAttnAudioScaleShift(scaledAudioTs.flattened())
        let crossAudioSSReshaped = crossAudioSSEmb.reshaped([batchSize, -1, 4, audioDim])
        let crossAudioSSPadded = MLX.concatenated([
            crossAudioSSReshaped,
            MLXArray.zeros([batchSize, 1, 1, audioDim])
        ], axis: 2)

        let (crossAudioGateEmb, _) = avCrossAttnAudioV2AGate(scaledAudioTs.flattened())
        let crossAudioGateReshaped = crossAudioGateEmb.reshaped([batchSize, -1, 1, audioDim])

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
            embeddedTimestep: videoEmbeddedTs
        )

        var audioArgs = AudioTransformerArgs(
            x: audioX,
            context: projectedAudioCtx,
            timesteps: audioTembReshaped,
            positionalEmbeddings: audioRoPE,
            contextMask: preparedAudioMask,
            embeddedTimestep: audioEmbeddedTs,
            crossVideoScaleShift: crossVideoSSPadded,
            crossVideoGate: crossVideoGateReshaped,
            crossAudioScaleShift: crossAudioSSPadded,
            crossAudioGate: crossAudioGateReshaped,
            crossVideoRoPE: crossVideoRoPE,
            crossAudioRoPE: crossAudioRoPE
        )

        // --- Process through transformer blocks ---
        for (i, block) in transformerBlocks.enumerated() {
            (videoArgs, audioArgs) = block(videoArgs, audio: audioArgs)

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
