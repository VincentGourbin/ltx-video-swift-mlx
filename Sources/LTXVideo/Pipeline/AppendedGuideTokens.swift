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
