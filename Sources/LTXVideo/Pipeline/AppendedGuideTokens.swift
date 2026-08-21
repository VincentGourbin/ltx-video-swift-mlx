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
    let guideTokens: MLXArray?
    let extRoPE: (cos: MLXArray, sin: MLXArray)
    let extCrossVideoRoPE: (cos: MLXArray, sin: MLXArray)?
    let originalCount: Int
    let guideCount: Int
    /// Layout of the trailing generated-keyframe slots, when the stage asked for
    /// any. Slots are denoised rather than frozen, so unlike `guideTokens` their
    /// content changes at every step and lives with the caller's latent state.
    let slots: GeneratedKeyframeLayout?
    /// Starting content for those slots — zeros for a first stage, the previous
    /// stage's slots for a refinement.
    let slotInitialTokens: MLXArray?

    init(
        guideTokens: MLXArray?,
        extRoPE: (cos: MLXArray, sin: MLXArray),
        extCrossVideoRoPE: (cos: MLXArray, sin: MLXArray)?,
        originalCount: Int,
        guideCount: Int,
        slots: GeneratedKeyframeLayout? = nil,
        slotInitialTokens: MLXArray? = nil
    ) {
        self.guideTokens = guideTokens
        self.extRoPE = extRoPE
        self.extCrossVideoRoPE = extCrossVideoRoPE
        self.originalCount = originalCount
        self.guideCount = guideCount
        self.slots = slots
        self.slotInitialTokens = slotInitialTokens
    }

    /// Total appended tokens — frozen guides plus denoised slots.
    var appendedCount: Int { guideCount + (slots?.tokenCount ?? 0) }
}

/// Assemble a context from guide tokens the caller already built.
///
/// `prepareKeyframeAppend` builds guides from encoded keyframe images; anchoring
/// a sequence on its own frames needs the same assembly over guides built with
/// explicit coordinates, so the shared half lives here.
func assembleAppendContext(
    guides: [AppendedGuideTokens],
    slotIndices: [Int] = [],
    slotInitial: MLXArray? = nil,
    shape: VideoLatentShape,
    hasAudio: Bool,
    refConfig: LTXTransformerConfig,
    stageLabel: String,
    fps: Float = 24.0
) -> AppendKeyframeContext? {
    guard !guides.isEmpty || !slotIndices.isEmpty else { return nil }
    // `fps` is the rate of the sequence being denoised, not of its source. A
    // temporally densified clip runs at twice the source rate and must be
    // positioned at that rate, or the model reads it as a clip of twice the
    // duration — half-speed motion, and temporal coordinates that can run past
    // the RoPE range on a long canvas.
    let basePos = createPositionGrid(
        batchSize: 1, frames: shape.frames, height: shape.height, width: shape.width,
        fps: fps)
    return buildContext(
        guides: guides, slotIndices: slotIndices, slotInitial: slotInitial,
        basePos: basePos, shape: shape,
        hasAudio: hasAudio, refConfig: refConfig, stageLabel: stageLabel, fps: fps)
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
    stageLabel: String,
    slotIndices: [Int] = [],
    slotInitial: MLXArray? = nil
) -> AppendKeyframeContext? {
    guard !encoded.isEmpty || !slotIndices.isEmpty else { return nil }

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
    return buildContext(
        guides: guides, slotIndices: slotIndices, slotInitial: slotInitial,
        basePos: basePos, shape: shape,
        hasAudio: hasAudio, refConfig: refConfig, stageLabel: stageLabel)
}

private func buildContext(
    guides: [AppendedGuideTokens],
    slotIndices: [Int] = [],
    slotInitial: MLXArray? = nil,
    basePos: MLXArray,
    shape: VideoLatentShape,
    hasAudio: Bool,
    refConfig: LTXTransformerConfig,
    stageLabel: String,
    fps: Float = 24.0
) -> AppendKeyframeContext {
    let allGuideTokens = guides.isEmpty
        ? nil : MLX.concatenated(guides.map { $0.tokens }, axis: 1)
    let originalCount = shape.tokenCount
    let guideCount = allGuideTokens?.dim(1) ?? 0

    // Slots come last so the marked, denoised span is trailing and contiguous.
    var slotLayout: GeneratedKeyframeLayout? = nil
    var slotTokens: MLXArray? = nil
    var slotGuides: [AppendedGuideTokens] = []
    if !slotIndices.isEmpty {
        let built = GeneratedKeyframeSlots.build(
            pixelFrameIndices: slotIndices, shape: shape, initial: slotInitial, fps: fps)
        slotGuides = built.guides
        slotTokens = MLX.concatenated(slotGuides.map { $0.tokens }, axis: 1)
        slotLayout = GeneratedKeyframeLayout(
            pixelFrameIndices: built.layout.pixelFrameIndices,
            tokensPerSlot: built.layout.tokensPerSlot,
            firstToken: originalCount + guideCount)
    }

    let appendedPositions = (guides + slotGuides).map { $0.positions }
    let extPositions = MLX.concatenated([basePos] + appendedPositions, axis: 2)

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

    LTXDebug.log("[append] \(stageLabel) extended sequence: \(originalCount) video + "
        + "\(guideCount) guide + \(slotLayout?.tokenCount ?? 0) slot tokens")

    return AppendKeyframeContext(
        guideTokens: allGuideTokens,
        extRoPE: pe,
        extCrossVideoRoPE: crossPE,
        originalCount: originalCount,
        guideCount: guideCount,
        slots: slotLayout,
        slotInitialTokens: slotTokens
    )
}

/// Build a per-token timestep tensor of shape `(B, originalCount + guideCount)` where
/// the first `originalCount` tokens hold the schedule's current `sigma` and the
/// trailing `guideCount` tokens hold `0` (clean reference, no denoising).
func buildExtendedTimestep(
    sigma: Float,
    originalCount: Int,
    guideCount: Int,
    slotCount: Int = 0,
    batchSize: Int = 1
) -> MLXArray {
    let totalCount = originalCount + guideCount + slotCount
    // Video tokens and slot tokens are both being denoised, so both carry the
    // schedule's sigma; only the frozen guides in between sit at 0.
    var values = [Float](repeating: sigma, count: totalCount)
    for i in originalCount ..< (originalCount + guideCount) {
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
    buildKeyframeGuideToken(
        encodedLatent: encodedLatent,
        temporalPosition: (Float(pixelFrameIndex) + 0.5) / fps,
        spatialScale: spatialScale, dtype: dtype)
}

/// Same, from an explicit temporal coordinate.
///
/// A guide built from an image belongs at a pixel frame, so the integer form
/// above is the natural one. A guide built from a latent frame *of the sequence
/// being denoised* has to land exactly on that frame's own grid coordinate —
/// which is `(8i - 3) / fps`, not expressible as `(pixel + 0.5) / fps`. Rounding
/// to the nearest pixel would offset the anchor by half a frame, so callers that
/// anchor on the sequence itself pass the coordinate directly.
func buildKeyframeGuideToken(
    encodedLatent: MLXArray,
    temporalPosition tPos: Float,
    spatialScale: Int = 32,
    dtype: DType = .bfloat16
) -> AppendedGuideTokens {
    let h = encodedLatent.dim(3)
    let w = encodedLatent.dim(4)
    let numTokens = h * w

    let patched = patchify(encodedLatent).asType(dtype)

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
