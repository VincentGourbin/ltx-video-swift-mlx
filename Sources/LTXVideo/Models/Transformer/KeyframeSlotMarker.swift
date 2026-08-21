// KeyframeSlotMarker.swift — the learned marker that tells the model "this token is a keyframe"
// Copyright 2026

import Foundation
@preconcurrency import MLX

/// Add the checkpoint's learned keyframe absolute-position embedding to a range
/// of projected tokens.
///
/// LTX-2.5 checkpoints ship a single `(1, innerDim)` vector under
/// `keyframes_abs_pos_embedding`. Upstream adds it, right after `patchify_proj`,
/// to every token that represents *one pixel frame* rather than the usual
/// eight-frame span: the sequence's first latent frame and any generated
/// keyframe slot. It is what distinguishes a slot from an ordinary latent frame
/// beyond RoPE, and a slot denoised without it lands off-distribution.
///
/// Marking nothing is the correct behaviour for every checkpoint that predates
/// the feature: their weight is absent, the parameter stays at its zero
/// initialisation, and adding zero is an exact no-op — the same guarantee
/// upstream documents for `enable_keyframes_abs_pos_embedding`.
///
/// - Parameters:
///   - x: projected tokens, `(B, T, D)`.
///   - range: half-open token range to mark, or `nil` to mark nothing.
///   - embedding: the `(1, D)` learned marker.
func applyKeyframeMarker(
    _ x: MLXArray,
    range: Range<Int>?,
    embedding: MLXArray
) -> MLXArray {
    guard let range, !range.isEmpty else { return x }
    precondition(range.lowerBound >= 0 && range.upperBound <= x.dim(1),
                 "keyframe marker range \(range) outside \(x.dim(1)) tokens")

    let marked = x[0..., range.lowerBound ..< range.upperBound, 0...]
        + embedding.asType(x.dtype)
    var pieces: [MLXArray] = []
    if range.lowerBound > 0 {
        pieces.append(x[0..., 0 ..< range.lowerBound, 0...])
    }
    pieces.append(marked)
    if range.upperBound < x.dim(1) {
        pieces.append(x[0..., range.upperBound ..< x.dim(1), 0...])
    }
    return pieces.count == 1 ? pieces[0] : MLX.concatenated(pieces, axis: 1)
}
