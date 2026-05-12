// KeyframeInterpolation.swift - Multi-keyframe interpolation types & helpers
// Copyright 2025

import Foundation
@preconcurrency import MLX

// MARK: - Internal Types

/// One keyframe after VAE encoding: its latent slot index, the encoded tensor
/// (shape `(1, 128, 1, latentH, latentW)`), and the original pixel-space index
/// (used to compute the appended guide token's RoPE temporal position).
struct EncodedKeyframe {
    let latentIdx: Int
    let latent: MLXArray
    let pixelFrameIndex: Int
}

// MARK: - Public Types

/// A single keyframe used to constrain video generation at a specific frame position.
///
/// Multiple keyframes can be combined to interpolate between them — e.g. a first frame
/// at position 0, an optional middle frame, and a last frame at `numFrames - 1`.
public struct KeyframeInput: Sendable, Equatable {
    /// Path to the keyframe image file.
    public let path: String

    /// Pixel-space frame index where this keyframe applies (0-based, < numFrames).
    public let pixelFrameIndex: Int

    /// Conditioning strength in `(0, 1]`. `1.0` = clean reference (σ=0 on the
    /// appended guide token); `< 1.0` is reserved for future soft conditioning
    /// and is rejected by `validateKeyframes()` until wired through.
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
///
/// With the appended-guide-token approach, the latent slot is informational only:
/// each keyframe sits in its own appended token sequence with a RoPE temporal
/// position of `(pixelFrameIndex + 0.5) / fps`, independent of slot collisions.
public func pixelFrameToLatentFrame(_ pixelFrame: Int) -> Int {
    if pixelFrame <= 0 { return 0 }
    return (pixelFrame + 7) / 8
}

// MARK: - Keyframe List Validation

/// Validate a list of keyframes against the target video configuration.
///
/// Checks: file existence, frame indices in `[0, numFrames - 1]`, no duplicate
/// pixel positions, strength == 1.0. Keyframes within the same latent stride
/// group (e.g. pixel 8 and pixel 12 both mapping to latent slot 1) ARE allowed
/// since the append path gives each its own RoPE temporal position.
public func validateKeyframes(_ keyframes: [KeyframeInput], numFrames: Int) throws {
    guard !keyframes.isEmpty else { return }

    var seenPixelIndices = Set<Int>()

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
                "Soft conditioning (strength < 1.0) is not yet implemented."
            )
        }
        guard seenPixelIndices.insert(kf.pixelFrameIndex).inserted else {
            throw LTXError.invalidConfiguration(
                "Duplicate keyframe at pixel frame \(kf.pixelFrameIndex)"
            )
        }
    }
}
