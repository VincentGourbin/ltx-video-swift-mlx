// AudioReference.swift — Audio reference latent conditioning for LipDub
// Copyright 2025
//
// Implements the audio side of Lightricks `AudioConditionByReferenceLatent`:
// the reference audio is VAE-encoded, packed, appended to the audio token
// sequence with RoPE temporal positions shifted into the NEGATIVE range so
// the reference sits "before" the target audio. The transformer learns
// (via the LipDub IC-LoRA) to use the reference for conditioning while
// leaving the target audio in `[0, target_dur]` unaffected by index.

import Foundation
@preconcurrency import MLX

/// Constant-across-denoising-steps state for an audio reference appended to
/// the audio stream of `LTX2Transformer`.
///
/// - `guideTokens`: packed audio tokens, shape `(1, T_ref, C_audio)` where
///   `C_audio = audioLatentChannels * audioLatentMelBins` (= 128 today).
/// - `extRoPE`: 1D temporal RoPE for the full extended sequence
///   (target audio + reference audio), passed as
///   `LTX2Transformer.precomputedAudioRoPE`.
/// - `extCrossRoPE`: cross-modal RoPE for the full extended audio sequence,
///   passed as `LTX2Transformer.precomputedCrossAudioRoPE`.
/// - `originalCount`: target audio token count — what's kept after `cropToOriginal`.
/// - `guideCount`: appended reference audio token count.
struct AudioReferenceContext {
    let guideTokens: MLXArray
    let extRoPE: (cos: MLXArray, sin: MLXArray)
    let extCrossRoPE: (cos: MLXArray, sin: MLXArray)
    let originalCount: Int
    let guideCount: Int
}

/// Build an `AudioReferenceContext` for the LipDub pipeline.
///
/// Replicates `patchify_lipdub_audio_reference_latent` from Lightricks `lipdub.py`
/// (single-frame narrowing on the audio side is implicit — each audio latent
/// frame already represents one short window, see `createAudioPositionGrid`).
///
/// Position math: each audio latent token covers `temporalScale * hopLength /
/// sampleRate = 0.04` seconds (40 ms). `createAudioPositionGrid` returns the
/// MIDDLE of that range. To shift the entire reference into the negative
/// range with a 40 ms gap before the target audio, we subtract
/// `max(refPositions) + halfTokenWidth + 0.04` from each position. After
/// the shift, the LAST reference token's END time sits at exactly `-0.04`,
/// just before the target audio (which starts at `~0` in non-negative time).
///
/// - Parameters:
///   - referenceLatent: VAE-encoded reference audio, shape `(1, 8, T_ref, 16)`
///     (output of `AudioVAE.encode`, mean only — no reparameterization).
///   - targetAudioFrames: target audio latent frame count (used to size the
///     extended self-attention RoPE for `target + reference`).
///   - refConfig: transformer config providing `audioInnerDim`,
///     `audioCrossAttentionDim`, `audioMaxPos`, `audioNumAttentionHeads`,
///     `ropeTheta`.
///   - audioLatentChannels: VAE channel count (default 8).
///   - audioLatentMelBins: VAE mel bin count after compression (default 16).
///   - hopLength / sampleRate / temporalScale: must match `createAudioPositionGrid`
///     defaults (160 / 16000 / 4).
func buildAudioReference(
    referenceLatent: MLXArray,
    targetAudioFrames: Int,
    refConfig: LTXTransformerConfig,
    audioLatentChannels: Int = 8,
    audioLatentMelBins: Int = 16,
    hopLength: Int = 160,
    sampleRate: Int = 16000,
    temporalScale: Int = 4
) -> AudioReferenceContext {
    // Pack: (B, C, T, M) -> (B, T, C, M) -> (B, T, C*M)
    let transposed = referenceLatent.transposed(0, 2, 1, 3)
    let packed = transposed.reshaped([transposed.dim(0), transposed.dim(1), -1])
    return buildAudioReferenceFromPacked(
        packed: packed,
        targetAudioFrames: targetAudioFrames,
        refConfig: refConfig,
        hopLength: hopLength,
        sampleRate: sampleRate,
        temporalScale: temporalScale
    )
}

/// Variant for Stage 2 of LipDub: the new reference is the Stage 1 audio output,
/// which is already in PACKED format `(B, T_ref, C_audio*M)` — no need to re-pack.
func buildAudioReferenceFromPacked(
    packed: MLXArray,
    targetAudioFrames: Int,
    refConfig: LTXTransformerConfig,
    hopLength: Int = 160,
    sampleRate: Int = 16000,
    temporalScale: Int = 4
) -> AudioReferenceContext {
    let refFrames = packed.dim(1)

    // 1D temporal positions for the reference, in the same convention as the
    // base audio sequence (middle of each token's time range, in seconds).
    let refPositions = createAudioPositionGrid(
        batchSize: 1,
        audioFrames: refFrames,
        hopLength: hopLength,
        sampleRate: sampleRate,
        temporalScale: temporalScale
    )

    // Shift into negative range. Python `patchify_lipdub_audio_reference_latent`
    // computes `aud_dur = positions[:, :, -1, 1].max()` (END of last token) then
    // `positions = positions - aud_dur - 0.04`. After the shift, the last token's
    // END time sits at exactly -0.04, just before the target audio at t=0.
    //
    // Our `createAudioPositionGrid` returns the MIDDLE of each token (Python's
    // RoPE averages start+end into the middle anyway — see rope.py:138-139).
    // To match Python's shift, we compute END_last = middle_last + halfTokenWidth
    // by reading the ACTUAL middle from refPositions (don't recompute — the
    // causal clamp at frame 0 makes the closed-form `(N-1)*width + width/2`
    // incorrect by one half-step at small offsets).
    let tokenWidth = Float(temporalScale) * Float(hopLength) / Float(sampleRate)
    let halfWidth = tokenWidth / 2.0
    let lastMid = refPositions[0, 0, refFrames - 1].item(Float.self)
    let shift = lastMid + halfWidth + 0.04
    let shiftedPositions = refPositions - MLXArray(shift)

    // Target audio positions (positive range) — same call as the LTX2Transformer
    // does internally when no precomputedAudioRoPE is provided.
    let targetPositions = createAudioPositionGrid(
        batchSize: 1,
        audioFrames: targetAudioFrames,
        hopLength: hopLength,
        sampleRate: sampleRate,
        temporalScale: temporalScale
    )

    // Concatenate target positions + shifted reference positions along the token axis.
    let extPositions = MLX.concatenated([targetPositions, shiftedPositions], axis: 2)

    // [DIAG] Print positions for debugging
    if ProcessInfo.processInfo.environment["LTX_LIPDUB_DUMP_POS"] == "1" {
        let refArr = refPositions.asArray(Float.self)
        let shiftedArr = shiftedPositions.asArray(Float.self)
        let targetArr = targetPositions.asArray(Float.self)
        print("[DIAG-AUDIO-POS] refFrames=\(refFrames) targetFrames=\(targetAudioFrames) tokenWidth=\(tokenWidth) halfWidth=\(halfWidth) shift=\(shift)")
        print("[DIAG-AUDIO-POS] refPositions  (first/last 3): \(refArr.prefix(3)) ... \(refArr.suffix(3))")
        print("[DIAG-AUDIO-POS] shifted (= ref - shift) (first/last 3): \(shiftedArr.prefix(3)) ... \(shiftedArr.suffix(3))")
        print("[DIAG-AUDIO-POS] targetPositions (first/last 3): \(targetArr.prefix(3)) ... \(targetArr.suffix(3))")
    }

    // Pre-compute RoPE for the extended sequence (audio self-attention).
    let extRoPE = precomputeFreqsCis(
        indicesGrid: extPositions,
        dim: refConfig.audioInnerDim,
        theta: refConfig.ropeTheta,
        maxPos: refConfig.audioMaxPos,
        numAttentionHeads: refConfig.audioNumAttentionHeads,
        ropeType: .split,
        doublePrecision: true
    )
    MLX.eval(extRoPE.cos, extRoPE.sin)

    // Pre-compute cross-modal RoPE (audio side) for the extended sequence.
    let extCrossRoPE = precomputeFreqsCis(
        indicesGrid: extPositions,
        dim: refConfig.audioCrossAttentionDim,
        theta: refConfig.ropeTheta,
        maxPos: refConfig.audioMaxPos,
        numAttentionHeads: refConfig.audioNumAttentionHeads,
        ropeType: .split,
        doublePrecision: true
    )
    MLX.eval(extCrossRoPE.cos, extCrossRoPE.sin)

    return AudioReferenceContext(
        guideTokens: packed,
        extRoPE: extRoPE,
        extCrossRoPE: extCrossRoPE,
        originalCount: targetAudioFrames,
        guideCount: refFrames
    )
}
