// AppendedGuideTokens.swift — Prototype primitive for keyframe append fix (issue #21)
// Copyright 2025

import Foundation
@preconcurrency import MLX

struct AppendedGuideTokens {
    let tokens: MLXArray
    let positions: MLXArray
}

func appendVideoGuides(
    videoTokens: MLXArray,
    basePositions: MLXArray,
    guides: [AppendedGuideTokens]
) -> (tokens: MLXArray, positions: MLXArray, originalCount: Int) {
    let originalCount = videoTokens.dim(1)
    guard !guides.isEmpty else {
        return (videoTokens, basePositions, originalCount)
    }

    var allTokens: [MLXArray] = [videoTokens]
    var allPositions: [MLXArray] = [basePositions]
    for g in guides {
        allTokens.append(g.tokens)
        allPositions.append(g.positions)
    }
    let tokens = MLX.concatenated(allTokens, axis: 1)
    let positions = MLX.concatenated(allPositions, axis: 2)
    return (tokens, positions, originalCount)
}

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
///   - dtype: Output dtype for the token tensor; positions stay float32.
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

/// Construct the base position grid for a video latent (same as createPositionGrid)
/// but exposed here so the pipeline can extend it with guide positions before
/// calling precomputeFreqsCis.
func buildBaseVideoPositions(
    batchSize: Int,
    frames: Int,
    height: Int,
    width: Int,
    temporalScale: Int = 8,
    spatialScale: Int = 32,
    fps: Float = 24.0,
    causalFix: Bool = true
) -> MLXArray {
    return createPositionGrid(
        batchSize: batchSize,
        frames: frames,
        height: height,
        width: width,
        temporalScale: temporalScale,
        spatialScale: spatialScale,
        fps: fps,
        causalFix: causalFix
    )
}
