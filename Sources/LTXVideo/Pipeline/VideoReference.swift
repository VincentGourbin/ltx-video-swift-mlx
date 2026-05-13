// VideoReference.swift — Video reference latent conditioning (IC-LoRA pattern, LipDub)
// Copyright 2025
//
// Implements the video side of Lightricks `VideoConditionByReferenceLatent`:
// the reference video is VAE-encoded at a downscaled resolution (e.g. 384px
// for a 768px target), patchified into multi-latent-frame tokens, and
// appended to the video token sequence with normal (positive) RoPE positions
// scaled spatially by `downscaleFactor` so they map back to the target
// coordinate space the IC-LoRA was trained on.
//
// Distinct from `AppendedGuideTokens` (single-frame keyframe) because:
//   - the reference is multi-frame (a whole video, not a still),
//   - spatial positions are scaled to compensate for downscaled encoding,
//   - no single-frame temporal narrowing is applied.

import Foundation
@preconcurrency import MLX

// The video-side context produced here is structurally identical to
// `AppendKeyframeContext` (defined in AppendedGuideTokens.swift) — same
// fields, same consumer (`runDenoiseStep`). We reuse the type rather than
// duplicate it.

/// Build an `AppendKeyframeContext` for IC-LoRA-based video reference conditioning (LipDub).
///
/// Replicates `VideoConditionByReferenceLatent.apply_to` from Lightricks
/// `reference_video_cond.py`. Position math: standard `createPositionGrid`
/// for the reference's actual `(F_ref, H_ref, W_ref)` latent shape, then
/// spatial coords (height + width) are multiplied by `downscaleFactor` so
/// they sit in the same coordinate space as the target video tokens
/// (which were generated at full resolution). Temporal coords are unchanged.
///
/// - Parameters:
///   - referenceLatent: VAE-encoded reference video, shape
///     `(1, 128, F_ref, H_ref, W_ref)`, already normalized via per-channel stats.
///   - targetShape: target video latent shape (used to size the extended
///     self-attention RoPE for `target + reference`).
///   - downscaleFactor: ratio between target and reference spatial resolution.
///     E.g. target 768px / reference 384px → factor 2. Read from the IC-LoRA
///     safetensors metadata (`reference_downscale_factor`); pass `1` if the
///     reference was encoded at full resolution.
///   - hasAudio: when `true`, also computes `extCrossVideoRoPE` so the
///     `LTX2Transformer` audio path can attend to the extended video sequence.
///   - refConfig: transformer config providing RoPE dimensions.
///   - dtype: token dtype — must match the transformer's patchified video
///     latent input dtype (bfloat16 today).
func buildVideoReference(
    referenceLatent: MLXArray,
    targetShape: VideoLatentShape,
    downscaleFactor: Int,
    hasAudio: Bool,
    refConfig: LTXTransformerConfig,
    dtype: DType = .bfloat16
) -> AppendKeyframeContext {
    let refFrames = referenceLatent.dim(2)
    let refHeight = referenceLatent.dim(3)
    let refWidth = referenceLatent.dim(4)

    // Patchify (1, 128, F, H, W) -> (1, F*H*W, 128)
    let guideTokens = patchify(referenceLatent).asType(dtype)
    let guideCount = guideTokens.dim(1)

    // Standard positions for the reference's own latent shape, then spatial
    // axes scaled by downscaleFactor to land in the target's coordinate space.
    let refPositions = createPositionGrid(
        batchSize: 1,
        frames: refFrames,
        height: refHeight,
        width: refWidth
    )
    let scaledRefPositions: MLXArray
    if downscaleFactor != 1 {
        let scale = Float(downscaleFactor)
        // positions shape (1, 3, T): axis 1 holds (time, height, width).
        // Multiply spatial axes only.
        let timePart = refPositions[0..., 0..<1, 0...]
        let heightPart = refPositions[0..., 1..<2, 0...] * MLXArray(scale)
        let widthPart = refPositions[0..., 2..<3, 0...] * MLXArray(scale)
        scaledRefPositions = MLX.concatenated([timePart, heightPart, widthPart], axis: 1)
    } else {
        scaledRefPositions = refPositions
    }

    // Target positions — same call the transformer makes internally when no
    // precomputedRoPE is provided.
    let targetPositions = createPositionGrid(
        batchSize: 1,
        frames: targetShape.frames,
        height: targetShape.height,
        width: targetShape.width
    )

    let extPositions = MLX.concatenated([targetPositions, scaledRefPositions], axis: 2)
    let originalCount = targetShape.tokenCount

    // Self-attention RoPE for the extended video sequence (target + reference).
    let extRoPE = precomputeFreqsCis(
        indicesGrid: extPositions,
        dim: refConfig.innerDim,
        theta: refConfig.ropeTheta,
        maxPos: refConfig.maxPos,
        numAttentionHeads: refConfig.numAttentionHeads,
        ropeType: .split,
        doublePrecision: true
    )
    MLX.eval(extRoPE.cos, extRoPE.sin)

    // Cross-modal video RoPE (temporal-only) for the extended sequence —
    // only needed when running the dual LTX2Transformer.
    var extCrossRoPE: (cos: MLXArray, sin: MLXArray)? = nil
    if hasAudio {
        let temporalOnly = extPositions[0..., 0..<1, 0...]
        let crossRoPE = precomputeFreqsCis(
            indicesGrid: temporalOnly,
            dim: refConfig.audioCrossAttentionDim,
            theta: refConfig.ropeTheta,
            maxPos: refConfig.audioMaxPos,
            numAttentionHeads: refConfig.audioNumAttentionHeads,
            ropeType: .split,
            doublePrecision: true
        )
        MLX.eval(crossRoPE.cos, crossRoPE.sin)
        extCrossRoPE = crossRoPE
    }

    return AppendKeyframeContext(
        guideTokens: guideTokens,
        extRoPE: extRoPE,
        extCrossVideoRoPE: extCrossRoPE,
        originalCount: originalCount,
        guideCount: guideCount
    )
}
