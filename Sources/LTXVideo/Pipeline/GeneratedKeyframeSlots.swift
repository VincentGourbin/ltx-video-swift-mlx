// GeneratedKeyframeSlots.swift — keyframes the model invents, then reuses as anchors
// Copyright 2026

import Foundation
@preconcurrency import MLX

/// Where the generated-keyframe slots sit inside an extended token sequence.
///
/// Slots are appended *after* any frozen guide tokens, so the sequence reads
/// `[video | guides | slots]` and the slot range is always trailing — which is
/// what lets the keyframe marker and the per-token timestep address it as one
/// contiguous span.
struct GeneratedKeyframeLayout: Sendable {
    /// Target pixel-frame index of each slot, strictly increasing.
    let pixelFrameIndices: [Int]
    /// Tokens one slot occupies: the target's full latent spatial grid.
    let tokensPerSlot: Int
    /// Index of the first slot token in the extended sequence.
    let firstToken: Int

    var slotCount: Int { pixelFrameIndices.count }
    var tokenCount: Int { tokensPerSlot * slotCount }
    var tokenRange: Range<Int> { firstToken ..< (firstToken + tokenCount) }
}

/// A generated keyframe slot differs from an ordinary appended guide token in
/// three ways, all of which have to hold together for the model to fill it:
///
/// 1. **It is denoised.** Guides carry timestep 0 (clean context) and their
///    predicted velocity is thrown away. A slot carries the schedule's current
///    sigma and its velocity is stepped exactly like the video latent's.
/// 2. **It is marked.** The learned `keyframes_abs_pos_embedding` is added to
///    its tokens, which is what tells the model this token is one pixel frame
///    rather than an eight-frame span.
/// 3. **Its RoPE span is one pixel frame**, `[t, t+1)`, not the eight-frame
///    span of a regular latent frame.
///
/// The payoff is an anchor generated *by the same pass* that generates the
/// video: a frame at a chosen position, at full quality, that later stages and
/// later temporal tiles can condition on. That is what keeps a long clip on one
/// identity across seams, where anchoring on a decoded neighbour frame drifts.
enum GeneratedKeyframeSlots {

    /// Build the slot token groups for one stage.
    ///
    /// - Parameters:
    ///   - pixelFrameIndices: target pixel frames, strictly increasing, inside the clip.
    ///   - shape: the stage's latent shape — slots take its spatial grid.
    ///   - initial: optional `(1, C, K, H, W)` starting content, e.g. the previous
    ///     stage's slots upscaled. `nil` starts every slot from zeros, which is
    ///     what a first stage does.
    ///   - fps: pixel frame rate the positions are expressed in.
    static func build(
        pixelFrameIndices: [Int],
        shape: VideoLatentShape,
        initial: MLXArray? = nil,
        fps: Float = 24.0,
        spatialScale: Int = 32,
        dtype: DType = .bfloat16
    ) -> (guides: [AppendedGuideTokens], layout: GeneratedKeyframeLayout) {
        precondition(!pixelFrameIndices.isEmpty, "a slot request needs at least one index")
        precondition(zip(pixelFrameIndices, pixelFrameIndices.dropFirst()).allSatisfy { $0 < $1 },
                     "slot indices must be strictly increasing, got \(pixelFrameIndices)")
        if let initial {
            precondition(initial.dim(2) == pixelFrameIndices.count,
                         "initial slots K=\(initial.dim(2)) ≠ \(pixelFrameIndices.count) indices")
            precondition(initial.dim(3) == shape.height && initial.dim(4) == shape.width,
                         "initial slots are \(initial.dim(3))×\(initial.dim(4)), stage is "
                         + "\(shape.height)×\(shape.width)")
        }

        let guides = pixelFrameIndices.enumerated().map { index, pixelFrame -> AppendedGuideTokens in
            let content = initial.map { $0[0..., 0..., index ..< (index + 1), 0..., 0...] }
                ?? MLXArray.zeros([1, shape.channels, 1, shape.height, shape.width], dtype: .float32)
            return buildKeyframeGuideToken(
                encodedLatent: content, pixelFrameIndex: pixelFrame,
                fps: fps, spatialScale: spatialScale, dtype: dtype)
        }
        let layout = GeneratedKeyframeLayout(
            pixelFrameIndices: pixelFrameIndices,
            tokensPerSlot: shape.height * shape.width,
            firstToken: 0)   // caller re-bases once the guide count is known
        return (guides, layout)
    }

    /// Unpack denoised slot tokens back into a `(1, C, K, H, W)` latent.
    static func unpack(
        tokens: MLXArray,
        layout: GeneratedKeyframeLayout,
        shape: VideoLatentShape
    ) -> MLXArray {
        precondition(tokens.dim(1) == layout.tokenCount,
                     "expected \(layout.tokenCount) slot tokens, got \(tokens.dim(1))")
        let frameShape = VideoLatentShape(
            batch: shape.batch, channels: shape.channels,
            frames: 1, height: shape.height, width: shape.width)
        let frames = (0 ..< layout.slotCount).map { index -> MLXArray in
            let start = index * layout.tokensPerSlot
            let slice = tokens[0..., start ..< (start + layout.tokensPerSlot), 0...]
            return unpatchify(slice, shape: frameShape)
        }
        return MLX.concatenated(frames, axis: 2)
    }

}

/// Slice the slot span out of an extended velocity, `nil` when the stage has no
/// slots. Kept next to the layout so the range convention has one owner.
func sliceSlotVelocity(_ velocity: MLXArray, layout: GeneratedKeyframeLayout?) -> MLXArray? {
    guard let layout else { return nil }
    let range = layout.tokenRange
    precondition(velocity.dim(1) >= range.upperBound,
                 "velocity has \(velocity.dim(1)) tokens, slots end at \(range.upperBound)")
    return velocity[0..., range.lowerBound ..< range.upperBound, 0...].asType(.float32)
}

/// Check a slot request against the clip it belongs to.
///
/// Upstream raises on out-of-range, unsorted or duplicate indices rather than
/// silently repairing them: a slot at the wrong frame produces a plausible video
/// anchored on the wrong moment, which is the expensive kind of wrong.
func validatedSlotIndices(_ indices: [Int], numFrames: Int) throws -> [Int] {
    guard indices.allSatisfy({ $0 >= 0 && $0 < numFrames }) else {
        throw LTXError.invalidConfiguration(
            "keyframe slot outside the clip: \(indices) against \(numFrames) frames")
    }
    guard zip(indices, indices.dropFirst()).allSatisfy({ $0 < $1 }) else {
        throw LTXError.invalidConfiguration(
            "keyframe slots must be strictly increasing, got \(indices)")
    }
    return indices
}
