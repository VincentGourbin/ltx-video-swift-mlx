// AppendedGuideTokens.swift — Keyframe conditioning via appended guide tokens (issue #21 fix)
// Copyright 2025

import Foundation
@preconcurrency import MLX

/// A group of "guide tokens" that get concatenated to the video token sequence
/// inside the transformer to condition generation on a keyframe image.
///
/// - `tokens`: patchified VAE-encoded keyframe, shape `(1, K, C)` where
///   `K = latentH * latentW` and `C` is the VAE channel count (128 for LTX-2.3).
/// - `positions`: 3D RoPE positions for those tokens, shape `(1, 3, K)`. The
///   temporal coord points to `(pixelFrameIndex + 0.5) / fps` (single-frame
///   narrowing per Lightricks `keyframe_cond.py`); spatial coords are the same
///   pixel-space midpoints used by the base video sequence.
struct AppendedGuideTokens {
    let tokens: MLXArray
    let positions: MLXArray
}

/// All the constant-across-denoising-steps state needed to run a stage with
/// appended keyframe guide tokens. Built once per stage by `prepareKeyframeAppend`,
/// consumed each step by the transformer call site.
///
/// - `guideTokens`: concatenated keyframe tokens, shape `(1, K_total, C)`.
/// - `extRoPE`: RoPE for the full extended sequence (video + guides), one
///   `(cos, sin)` pair shared by `LTXTransformer.precomputedRoPE` and
///   `LTX2Transformer.precomputedVideoRoPE`.
/// - `extCrossVideoRoPE`: cross-modal video RoPE (temporal-only) for the
///   extended sequence — only set when audio is enabled, used by
///   `LTX2Transformer.precomputedCrossVideoRoPE`.
/// - `originalCount`: token count of the un-extended video latent — what the
///   scheduler step operates on after `cropToOriginal`.
/// - `guideCount`: total appended guide token count (sum over all keyframes).
struct AppendKeyframeContext {
    let guideTokens: MLXArray
    let extRoPE: (cos: MLXArray, sin: MLXArray)
    let extCrossVideoRoPE: (cos: MLXArray, sin: MLXArray)?
    let originalCount: Int
    let guideCount: Int
}

/// Build the constant-across-steps `AppendKeyframeContext` for one stage.
///
/// Encodes each keyframe into a guide token group, concatenates them, computes
/// the extended RoPE (and the cross-modal video RoPE when audio is enabled).
/// Returns `nil` when there are no keyframes.
///
/// - Parameters:
///   - encoded: VAE-encoded keyframes (output of `encodeKeyframes`).
///   - shape: latent shape of the stage (used for `originalCount` and base positions).
///   - hasAudio: when `true`, also computes `extCrossVideoRoPE` for the LTX2 audio path.
///   - refConfig: transformer config providing RoPE dimensions / theta / maxPos.
///   - stageLabel: human-readable tag (e.g. "Stage 1") used in debug logs.
func prepareKeyframeAppend(
    encoded: [EncodedKeyframe],
    shape: VideoLatentShape,
    hasAudio: Bool,
    refConfig: LTXTransformerConfig,
    stageLabel: String
) -> AppendKeyframeContext? {
    guard !encoded.isEmpty else { return nil }

    let basePos = createPositionGrid(
        batchSize: 1,
        frames: shape.frames,
        height: shape.height,
        width: shape.width
    )
    let guides = encoded.map { kf in
        buildKeyframeGuideToken(
            encodedLatent: kf.latent,
            pixelFrameIndex: kf.pixelFrameIndex,
            fps: 24.0
        )
    }
    let allGuideTokens = MLX.concatenated(guides.map { $0.tokens }, axis: 1)
    let allGuidePositions = MLX.concatenated(guides.map { $0.positions }, axis: 2)
    let originalCount = shape.tokenCount
    let guideCount = allGuideTokens.dim(1)

    let extPositions = MLX.concatenated([basePos, allGuidePositions], axis: 2)

    let pe = precomputeFreqsCis(
        indicesGrid: extPositions,
        dim: refConfig.innerDim,
        theta: refConfig.ropeTheta,
        maxPos: refConfig.maxPos,
        numAttentionHeads: refConfig.numAttentionHeads,
        ropeType: .split,
        doublePrecision: true
    )
    MLX.eval(pe.cos, pe.sin)

    var crossPE: (cos: MLXArray, sin: MLXArray)? = nil
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
        crossPE = crossRoPE
    }

    LTXDebug.log("[append] \(stageLabel) extended sequence: \(originalCount) video + \(guideCount) guide tokens")

    return AppendKeyframeContext(
        guideTokens: allGuideTokens,
        extRoPE: pe,
        extCrossVideoRoPE: crossPE,
        originalCount: originalCount,
        guideCount: guideCount
    )
}

/// Build a per-token timestep tensor of shape `(B, originalCount + guideCount)` where
/// the first `originalCount` tokens hold the schedule's current `sigma` and the
/// trailing `guideCount` tokens hold `0` (clean reference, no denoising).
func buildExtendedTimestep(
    sigma: Float,
    originalCount: Int,
    guideCount: Int,
    batchSize: Int = 1
) -> MLXArray {
    let totalCount = originalCount + guideCount
    var values = [Float](repeating: sigma, count: totalCount)
    for i in originalCount..<totalCount {
        values[i] = 0.0
    }
    let arr = MLXArray(values, [1, totalCount])
    if batchSize == 1 {
        return arr
    }
    return MLX.broadcast(arr, to: [batchSize, totalCount])
}

/// Slice `velocity` (shape `(B, T_total, C)`) down to the first `originalCount`
/// tokens — the appended guide tokens' predicted velocity is discarded since the
/// guides never enter the denoised latent.
func cropToOriginal(velocity: MLXArray, originalCount: Int) -> MLXArray {
    return velocity[0..., 0..<originalCount, 0...]
}

/// Build a guide token group from a VAE-encoded keyframe latent.
///
/// Replicates Lightricks `VideoConditionByKeyframeIndex` semantics for `num_pixel_frames=1`:
/// the temporal position is `(pixelFrameIndex + 0.5) / fps` (the "middle" of a 1-frame range
/// starting at `pixelFrameIndex`), spatial positions are pixel-space middles like the base
/// sequence. This matches the Swift RoPE convention (middle of bounds, not start/end pair).
///
/// - Parameters:
///   - encodedLatent: VAE-encoded keyframe of shape `(1, 128, 1, latentH, latentW)`,
///                    already normalized via per-channel stats.
///   - pixelFrameIndex: Target pixel frame index where this keyframe should sit (0-based).
///   - fps: Pixel frame rate, default 24.0 (matches createPositionGrid default).
///   - spatialScale: VAE spatial compression factor, default 32.
///   - dtype: Output dtype for the token tensor — must match the dtype the
///            transformer expects for its patchified video latent input
///            (bfloat16 today). Positions stay float32 for RoPE precision.
func buildKeyframeGuideToken(
    encodedLatent: MLXArray,
    pixelFrameIndex: Int,
    fps: Float = 24.0,
    spatialScale: Int = 32,
    dtype: DType = .bfloat16
) -> AppendedGuideTokens {
    let h = encodedLatent.dim(3)
    let w = encodedLatent.dim(4)
    let numTokens = h * w

    let patched = patchify(encodedLatent).asType(dtype)

    let tPos = (Float(pixelFrameIndex) + 0.5) / fps
    let tCoords = [Float](repeating: tPos, count: numTokens)

    let sScale = Float(spatialScale)
    var hCoords = [Float]()
    hCoords.reserveCapacity(numTokens)
    var wCoords = [Float]()
    wCoords.reserveCapacity(numTokens)
    for hi in 0..<h {
        let hp = Float(hi) * sScale + sScale / 2.0
        for wi in 0..<w {
            let wp = Float(wi) * sScale + sScale / 2.0
            hCoords.append(hp)
            wCoords.append(wp)
        }
    }

    let tArr = MLXArray(tCoords)
    let hArr = MLXArray(hCoords)
    let wArr = MLXArray(wCoords)
    let stacked = MLX.stacked([tArr, hArr, wArr], axis: 0)
    let positions = stacked.expandedDimensions(axis: 0).asType(.float32)

    return AppendedGuideTokens(tokens: patched, positions: positions)
}
