// LTX2TransformerBlock.swift - Dual Video/Audio Transformer Block for LTX-2
// Implements the LTX2VideoTransformerBlock from HuggingFace Diffusers
// Each block processes parallel video and audio streams with cross-modal attention
// Copyright 2025

import Foundation
@preconcurrency import MLX
import MLXFast
import MLXNN

// MARK: - Audio Transformer Arguments

/// Arguments for the audio stream in dual video/audio transformer blocks
struct AudioTransformerArgs {
    /// Audio hidden states (B, T_audio, D_audio)
    var x: MLXArray

    /// Audio text context for cross-attention (B, S, D_audio)
    var context: MLXArray

    /// Audio timestep embeddings for AdaLN (B, 1, 6, D_audio)
    var timesteps: MLXArray

    /// Audio RoPE position embeddings (cos, sin) — 1D temporal
    var positionalEmbeddings: (cos: MLXArray, sin: MLXArray)

    /// Audio attention mask for text cross-attention
    var contextMask: MLXArray?

    /// Raw audio embedded timestep
    var embeddedTimestep: MLXArray?

    // Cross-modal modulation (from global timestep embeddings)

    /// Video cross-modal scale/shift: (B, 1, 4, D_video) for a2v/v2a modulation
    var crossVideoScaleShift: MLXArray

    /// Video a2v gate: (B, 1, 1, D_video)
    var crossVideoGate: MLXArray

    /// Audio cross-modal scale/shift: (B, 1, 4, D_audio)
    var crossAudioScaleShift: MLXArray

    /// Audio v2a gate: (B, 1, 1, D_audio)
    var crossAudioGate: MLXArray

    /// Cross-modal video RoPE (for KV in v2a, Q in a2v cross-attention)
    var crossVideoRoPE: (cos: MLXArray, sin: MLXArray)

    /// Cross-modal audio RoPE (for KV in a2v, Q in v2a cross-attention)
    var crossAudioRoPE: (cos: MLXArray, sin: MLXArray)
}

// MARK: - LTX2 Transformer Block

/// Dual video/audio transformer block with cross-modal attention
///
/// Architecture per block:
///     1. Video self-attention  |  Audio self-attention
///     2. Video text cross-attention  |  Audio text cross-attention
///     3. A2V cross-attention + V2A cross-attention
///     4. Video FFN  |  Audio FFN
///
/// Weight keys match Python Diffusers `LTX2VideoTransformerBlock`:
///   Video: norm1, attn1, norm2, attn2, norm3, ff, scale_shift_table
///   Audio: audio_norm1, audio_attn1, ..., audio_scale_shift_table
///   Cross-modal: audio_to_video_{norm,attn}, video_to_audio_{norm,attn}
///   Cross-modal SST: video_a2v_cross_attn_scale_shift_table, audio_a2v_cross_attn_scale_shift_table
class LTX2TransformerBlock: Module {
    let normEps: Float
    let videoDim: Int
    let audioDim: Int

    // --- Video modules ---
    @ModuleInfo(key: "norm1") var norm1: RMSNorm
    @ModuleInfo(key: "attn1") var attn1: LTXAttention
    @ModuleInfo(key: "norm2") var norm2: RMSNorm
    @ModuleInfo(key: "attn2") var attn2: LTXAttention
    @ModuleInfo(key: "norm3") var norm3: RMSNorm
    @ModuleInfo(key: "ff") var ff: LTXFeedForward
    @ParameterInfo(key: "scale_shift_table") var scaleShiftTable: MLXArray

    // --- Audio modules ---
    @ModuleInfo(key: "audio_norm1") var audioNorm1: RMSNorm
    @ModuleInfo(key: "audio_attn1") var audioAttn1: LTXAttention
    @ModuleInfo(key: "audio_norm2") var audioNorm2: RMSNorm
    @ModuleInfo(key: "audio_attn2") var audioAttn2: LTXAttention
    @ModuleInfo(key: "audio_norm3") var audioNorm3: RMSNorm
    @ModuleInfo(key: "audio_ff") var audioFf: LTXFeedForward
    @ParameterInfo(key: "audio_scale_shift_table") var audioScaleShiftTable: MLXArray

    // --- Cross-modal modules ---
    @ModuleInfo(key: "audio_to_video_norm") var audioToVideoNorm: RMSNorm
    @ModuleInfo(key: "audio_to_video_attn") var audioToVideoAttn: LTXAttention
    @ModuleInfo(key: "video_to_audio_norm") var videoToAudioNorm: RMSNorm
    @ModuleInfo(key: "video_to_audio_attn") var videoToAudioAttn: LTXAttention

    // --- Cross-modal scale-shift tables ---
    @ParameterInfo(key: "video_a2v_cross_attn_scale_shift_table") var videoA2VCrossAttnSST: MLXArray
    @ParameterInfo(key: "audio_a2v_cross_attn_scale_shift_table") var audioA2VCrossAttnSST: MLXArray

    init(
        videoDim: Int,
        videoNumHeads: Int,
        videoHeadDim: Int,
        videoCrossAttentionDim: Int,
        audioDim: Int,
        audioNumHeads: Int,
        audioHeadDim: Int,
        audioCrossAttentionDim: Int,
        ropeType: LTXRopeType = .split,
        normEps: Float = 1e-6
    ) {
        self.normEps = normEps
        self.videoDim = videoDim
        self.audioDim = audioDim

        // --- Video ---
        self._norm1.wrappedValue = RMSNorm(dims: videoDim, eps: normEps)
        self._attn1.wrappedValue = LTXAttention(
            queryDim: videoDim, contextDim: nil,
            heads: videoNumHeads, dimHead: videoHeadDim,
            normEps: normEps, ropeType: ropeType
        )
        self._norm2.wrappedValue = RMSNorm(dims: videoDim, eps: normEps)
        self._attn2.wrappedValue = LTXAttention(
            queryDim: videoDim, contextDim: videoCrossAttentionDim,
            heads: videoNumHeads, dimHead: videoHeadDim,
            normEps: normEps, ropeType: ropeType
        )
        self._norm3.wrappedValue = RMSNorm(dims: videoDim, eps: normEps)
        self._ff.wrappedValue = LTXFeedForward(dim: videoDim, dimOut: videoDim)
        self._scaleShiftTable.wrappedValue = MLXArray.zeros([6, videoDim])

        // --- Audio ---
        self._audioNorm1.wrappedValue = RMSNorm(dims: audioDim, eps: normEps)
        self._audioAttn1.wrappedValue = LTXAttention(
            queryDim: audioDim, contextDim: nil,
            heads: audioNumHeads, dimHead: audioHeadDim,
            normEps: normEps, ropeType: ropeType
        )
        self._audioNorm2.wrappedValue = RMSNorm(dims: audioDim, eps: normEps)
        self._audioAttn2.wrappedValue = LTXAttention(
            queryDim: audioDim, contextDim: audioCrossAttentionDim,
            heads: audioNumHeads, dimHead: audioHeadDim,
            normEps: normEps, ropeType: ropeType
        )
        self._audioNorm3.wrappedValue = RMSNorm(dims: audioDim, eps: normEps)
        self._audioFf.wrappedValue = LTXFeedForward(dim: audioDim, dimOut: audioDim)
        self._audioScaleShiftTable.wrappedValue = MLXArray.zeros([6, audioDim])

        // --- Cross-modal ---
        // A2V: Q from video (videoDim), KV from audio (audioDim), uses audio head dims
        self._audioToVideoNorm.wrappedValue = RMSNorm(dims: videoDim, eps: normEps)
        self._audioToVideoAttn.wrappedValue = LTXAttention(
            queryDim: videoDim, contextDim: audioDim,
            heads: audioNumHeads, dimHead: audioHeadDim,
            normEps: normEps, ropeType: ropeType
        )

        // V2A: Q from audio (audioDim), KV from video (videoDim), uses audio head dims
        self._videoToAudioNorm.wrappedValue = RMSNorm(dims: audioDim, eps: normEps)
        self._videoToAudioAttn.wrappedValue = LTXAttention(
            queryDim: audioDim, contextDim: videoDim,
            heads: audioNumHeads, dimHead: audioHeadDim,
            normEps: normEps, ropeType: ropeType
        )

        // Cross-modal scale-shift tables
        // Video: 5 values [a2v_scale, a2v_shift, v2a_scale, v2a_shift, a2v_gate]
        self._videoA2VCrossAttnSST.wrappedValue = MLXArray.zeros([5, videoDim])
        // Audio: 5 values [a2v_scale, a2v_shift, v2a_scale, v2a_shift, v2a_gate]
        self._audioA2VCrossAttnSST.wrappedValue = MLXArray.zeros([5, audioDim])
    }

    // MARK: - Forward Pass

    func callAsFunction(
        _ videoArgs: TransformerArgs,
        audio audioArgs: AudioTransformerArgs
    ) -> (TransformerArgs, AudioTransformerArgs) {
        var videoX = videoArgs.x
        var audioX = audioArgs.x

        // Get video AdaLN values (6 values from SST + timestep)
        let videoSST = scaleShiftTable.reshaped([1, 1, 6, videoDim]) + videoArgs.timesteps
        let (vShiftMSA, vScaleMSA, vGateMSA) = (
            videoSST[0..., 0..., 0, 0...],
            videoSST[0..., 0..., 1, 0...],
            videoSST[0..., 0..., 2, 0...]
        )
        let (vShiftMLP, vScaleMLP, vGateMLP) = (
            videoSST[0..., 0..., 3, 0...],
            videoSST[0..., 0..., 4, 0...],
            videoSST[0..., 0..., 5, 0...]
        )

        // Get audio AdaLN values
        let audioSST = audioScaleShiftTable.reshaped([1, 1, 6, audioDim]) + audioArgs.timesteps
        let (aShiftMSA, aScaleMSA, aGateMSA) = (
            audioSST[0..., 0..., 0, 0...],
            audioSST[0..., 0..., 1, 0...],
            audioSST[0..., 0..., 2, 0...]
        )
        let (aShiftMLP, aScaleMLP, aGateMLP) = (
            audioSST[0..., 0..., 3, 0...],
            audioSST[0..., 0..., 4, 0...],
            audioSST[0..., 0..., 5, 0...]
        )

        // Phase 1: Video self-attention
        let normV1 = norm1(videoX) * (1 + vScaleMSA) + vShiftMSA
        let vSelfOut = attn1(normV1, pe: videoArgs.positionalEmbeddings)
        videoX = videoX + vSelfOut * vGateMSA

        // Phase 2: Audio self-attention
        let normA1 = audioNorm1(audioX) * (1 + aScaleMSA) + aShiftMSA
        let aSelfOut = audioAttn1(normA1, pe: audioArgs.positionalEmbeddings)
        audioX = audioX + aSelfOut * aGateMSA

        // Phase 3: Video text cross-attention (no RoPE)
        let normV2 = norm2(videoX)
        let vCrossOut = attn2(normV2, context: videoArgs.context, mask: videoArgs.contextMask)
        videoX = videoX + vCrossOut

        // Phase 4: Audio text cross-attention (no RoPE)
        let normA2 = audioNorm2(audioX)
        let aCrossOut = audioAttn2(normA2, context: audioArgs.context, mask: audioArgs.contextMask)
        audioX = audioX + aCrossOut

        // Phase 5-6: Cross-modal attention (A2V and V2A)
        // Compute cross-modal modulation from per-block SST + global timestep embeddings
        let videoCA = videoA2VCrossAttnSST.reshaped([1, 1, 5, videoDim]) + audioArgs.crossVideoScaleShift
        let audioCA = audioA2VCrossAttnSST.reshaped([1, 1, 5, audioDim]) + audioArgs.crossAudioScaleShift

        // Video modulation values
        let vA2VScale = videoCA[0..., 0..., 0, 0...]
        let vA2VShift = videoCA[0..., 0..., 1, 0...]
        let vV2AScale = videoCA[0..., 0..., 2, 0...]
        let vV2AShift = videoCA[0..., 0..., 3, 0...]
        let vA2VGate = videoCA[0..., 0..., 4, 0...] * audioArgs.crossVideoGate

        // Audio modulation values
        let aA2VScale = audioCA[0..., 0..., 0, 0...]
        let aA2VShift = audioCA[0..., 0..., 1, 0...]
        let aV2AScale = audioCA[0..., 0..., 2, 0...]
        let aV2AShift = audioCA[0..., 0..., 3, 0...]
        let aV2AGate = audioCA[0..., 0..., 4, 0...] * audioArgs.crossAudioGate

        // Norm video and audio for cross-modal
        let normVCA = audioToVideoNorm(videoX)
        let normACA = videoToAudioNorm(audioX)

        // Phase 5: A2V cross-attention (audio modulates video)
        let modV_a2v = normVCA * (1 + vA2VScale) + vA2VShift
        let modA_a2v = normACA * (1 + aA2VScale) + aA2VShift
        let a2vOut = audioToVideoAttn(
            modV_a2v,
            context: modA_a2v,
            pe: audioArgs.crossVideoRoPE,
            kPe: audioArgs.crossAudioRoPE
        )
        videoX = videoX + a2vOut * vA2VGate

        // Phase 6: V2A cross-attention (video modulates audio)
        let modA_v2a = normACA * (1 + aV2AScale) + aV2AShift
        let modV_v2a = normVCA * (1 + vV2AScale) + vV2AShift
        let v2aOut = videoToAudioAttn(
            modA_v2a,
            context: modV_v2a,
            pe: audioArgs.crossAudioRoPE,
            kPe: audioArgs.crossVideoRoPE
        )
        audioX = audioX + v2aOut * aV2AGate

        // Phase 7: Video FFN
        let normV3 = norm3(videoX) * (1 + vScaleMLP) + vShiftMLP
        let vFFOut = ff(normV3)
        videoX = videoX + vFFOut * vGateMLP

        // Phase 8: Audio FFN
        let normA3 = audioNorm3(audioX) * (1 + aScaleMLP) + aShiftMLP
        let aFFOut = audioFf(normA3)
        audioX = audioX + aFFOut * aGateMLP

        return (
            videoArgs.replacing(x: videoX),
            AudioTransformerArgs(
                x: audioX,
                context: audioArgs.context,
                timesteps: audioArgs.timesteps,
                positionalEmbeddings: audioArgs.positionalEmbeddings,
                contextMask: audioArgs.contextMask,
                embeddedTimestep: audioArgs.embeddedTimestep,
                crossVideoScaleShift: audioArgs.crossVideoScaleShift,
                crossVideoGate: audioArgs.crossVideoGate,
                crossAudioScaleShift: audioArgs.crossAudioScaleShift,
                crossAudioGate: audioArgs.crossAudioGate,
                crossVideoRoPE: audioArgs.crossVideoRoPE,
                crossAudioRoPE: audioArgs.crossAudioRoPE
            )
        )
    }
}
