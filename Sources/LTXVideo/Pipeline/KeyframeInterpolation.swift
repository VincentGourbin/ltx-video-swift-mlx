// KeyframeInterpolation.swift - Multi-keyframe interpolation types & helpers
// Copyright 2025

import Foundation
@preconcurrency import MLX
import MLXRandom

// MARK: - Internal Types

/// One keyframe after VAE encoding: its latent slot index plus the encoded tensor
/// (shape `(1, 128, 1, latentH, latentW)`).
struct EncodedKeyframe {
    let latentIdx: Int
    let latent: MLXArray
}

// MARK: - Public Types

/// A single keyframe used to constrain video generation at a specific frame position.
///
/// Multiple keyframes can be combined to interpolate between them — e.g. a first frame
/// at position 0, an optional middle frame, and a last frame at `numFrames - 1`.
///
/// Pixel positions are mapped to latent positions internally (latent stride = 8).
public struct KeyframeInput: Sendable, Equatable {
    /// Path to the keyframe image file.
    public let path: String

    /// Pixel-space frame index where this keyframe applies (0-based, < numFrames).
    public let pixelFrameIndex: Int

    /// Conditioning strength. Currently must be `1.0` (hard injection — the latent
    /// is forced to the encoded image). Values `!= 1.0` are rejected by
    /// `validateKeyframes()` until soft conditioning is wired through.
    public let strength: Float

    public init(path: String, pixelFrameIndex: Int, strength: Float = 1.0) {
        self.path = path
        self.pixelFrameIndex = pixelFrameIndex
        self.strength = strength
    }
}

// MARK: - Frame Index Mapping

/// Convert a pixel-space frame index to its latent-space frame index.
///
/// LTX-2 latent layout: `output_frames = 8 * (latent_frames - 1) + 1`.
/// - Pixel frame 0 maps to latent frame 0 (the standalone "+1" frame).
/// - Pixel frames 1..8 map to latent frame 1, 9..16 to latent frame 2, etc.
public func pixelFrameToLatentFrame(_ pixelFrame: Int) -> Int {
    if pixelFrame <= 0 { return 0 }
    return (pixelFrame + 7) / 8
}

// MARK: - Keyframe List Validation

/// Validate a list of keyframes against the target video configuration.
///
/// Checks: file existence, frame indices in `[0, numFrames - 1]`, no duplicate
/// pixel positions, no duplicate latent positions (since each latent slot can only
/// hold one keyframe), strength in `(0, 1]`.
public func validateKeyframes(_ keyframes: [KeyframeInput], numFrames: Int) throws {
    guard !keyframes.isEmpty else { return }

    var seenPixelIndices = Set<Int>()
    var seenLatentIndices = Set<Int>()

    for kf in keyframes {
        guard FileManager.default.fileExists(atPath: kf.path) else {
            throw LTXError.fileNotFound("Keyframe image not found: \(kf.path)")
        }
        guard kf.pixelFrameIndex >= 0 && kf.pixelFrameIndex < numFrames else {
            throw LTXError.invalidConfiguration(
                "Keyframe pixelFrameIndex \(kf.pixelFrameIndex) out of range [0, \(numFrames - 1)]"
            )
        }
        guard kf.strength == 1.0 else {
            throw LTXError.invalidConfiguration(
                "Keyframe strength must be 1.0 (got \(kf.strength) for \(kf.path)). " +
                "Soft conditioning (strength < 1.0) is not yet implemented; injection is hard."
            )
        }
        guard seenPixelIndices.insert(kf.pixelFrameIndex).inserted else {
            throw LTXError.invalidConfiguration(
                "Duplicate keyframe at pixel frame \(kf.pixelFrameIndex)"
            )
        }
        let latentIdx = pixelFrameToLatentFrame(kf.pixelFrameIndex)
        guard seenLatentIndices.insert(latentIdx).inserted else {
            throw LTXError.invalidConfiguration(
                "Multiple keyframes collide on latent frame \(latentIdx). " +
                "Latent stride is 8 — keyframes within the same 8-frame group cannot coexist."
            )
        }
    }
}

// MARK: - Mask & Injection Helpers

/// Build a per-token conditioning mask. Tokens belonging to a keyframe latent frame
/// get value `1.0`, all others get `0.0`. Output shape: `[1, shape.tokenCount]`.
///
/// The mask is consumed by the per-token timestep formula
/// `videoTimestep = sigma * (1 - mask)` so that keyframe tokens are denoised with
/// `σ = 0` (i.e. treated as clean references) while the rest follow the schedule.
func buildKeyframeMask(latentIndices: [Int], shape: VideoLatentShape) -> MLXArray {
    let tokensPerFrame = shape.height * shape.width
    var maskValues = [Float](repeating: 0.0, count: shape.tokenCount)
    for idx in latentIndices {
        guard idx >= 0 && idx < shape.frames else { continue }
        let start = idx * tokensPerFrame
        for t in start..<(start + tokensPerFrame) {
            maskValues[t] = 1.0
        }
    }
    return MLXArray(maskValues, [1, shape.tokenCount])
}

/// Overwrite latent slots at each keyframe position with the encoded keyframe tensor.
///
/// When `sigma > 0` and `noiseScale > 0`, noise is added so the conditioned slot
/// looks "realistic at the current schedule level" before the transformer pass.
/// With defaults (`sigma = 0`, `noiseScale = 0`) the clean reference is restored —
/// useful as a post-step re-injection that keeps keyframe slots pristine across
/// the loop and into the final latent.
///
/// `latent` must have shape `(1, C, F, H, W)` and each `EncodedKeyframe.latent`
/// must have shape `(1, C, 1, H, W)`.
func injectKeyframeLatents(
    into latent: inout MLXArray,
    encoded: [EncodedKeyframe],
    sigma: Float = 0.0,
    noiseScale: Float = 0.0
) {
    for kf in encoded {
        let slot = kf.latentIdx
        if sigma > 0 && noiseScale > 0 {
            let noise = MLXRandom.normal(kf.latent.shape)
            let noised = kf.latent + MLXArray(noiseScale) * noise * MLXArray(sigma * sigma)
            latent[0..., 0..., slot..<(slot + 1), 0..., 0...] = noised
        } else {
            latent[0..., 0..., slot..<(slot + 1), 0..., 0...] = kf.latent
        }
    }
}
