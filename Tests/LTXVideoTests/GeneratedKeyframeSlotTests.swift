// GeneratedKeyframeSlotTests.swift — slot layout, marking, timestep, round trip
// Copyright 2026

import Foundation
import Testing
@preconcurrency import MLX
@testable import LTXVideo

@Suite("Generated keyframe slots")
struct GeneratedKeyframeSlotTests {

    static let shape = VideoLatentShape(batch: 1, channels: 4, frames: 3, height: 2, width: 3)
    static let config = LTXTransformerConfig.default

    @Test func layoutSitsAfterVideoAndGuides() {
        // The marker and the per-token timestep both address the slots as one
        // trailing span; if the layout drifted from where the tokens actually
        // land, the model would be marking video tokens as keyframes.
        let guide = buildKeyframeGuideToken(
            encodedLatent: MLXArray.zeros([1, 4, 1, 2, 3], dtype: .float32), pixelFrameIndex: 0)
        let ctx = assembleAppendContext(
            guides: [guide], slotIndices: [8, 16], shape: Self.shape,
            hasAudio: false, refConfig: Self.config, stageLabel: "test")
        let layout = try! #require(ctx?.slots)

        #expect(layout.tokensPerSlot == 2 * 3)
        #expect(layout.tokenCount == 12)
        #expect(layout.firstToken == Self.shape.tokenCount + 6)
        #expect(layout.tokenRange.upperBound
            == Self.shape.tokenCount + ctx!.guideCount + layout.tokenCount)
        // RoPE must cover video + guides + slots, or the slots attend at position 0.
        #expect(ctx!.extRoPE.cos.shape.contains(layout.tokenRange.upperBound),
                "RoPE \(ctx!.extRoPE.cos.shape) has no axis of \(layout.tokenRange.upperBound) tokens")
    }

    @Test func slotsAloneNeedNoGuides() {
        // Text-to-video asks for slots with no image conditioning at all.
        let ctx = assembleAppendContext(
            guides: [], slotIndices: [0, 32], shape: Self.shape,
            hasAudio: false, refConfig: Self.config, stageLabel: "test")
        #expect(ctx?.guideTokens == nil)
        #expect(ctx?.guideCount == 0)
        #expect(ctx?.slots?.firstToken == Self.shape.tokenCount)
    }

    @Test func slotSpansOnePixelFrame() {
        // A slot's temporal coordinate is the middle of [t, t+1) — the single-frame
        // narrowing that distinguishes it from a latent frame spanning eight.
        let (guides, _) = GeneratedKeyframeSlots.build(
            pixelFrameIndices: [24], shape: Self.shape)
        let t = guides[0].positions[0, 0, 0].item(Float.self)
        #expect(abs(t - (24.0 + 0.5) / 24.0) < 1e-6)
    }

    @Test func matchesUpstreamSlotPositions() {
        // Ground truth from `scripts/keyframe_slot_reference.py`, which runs
        // Lightricks' own `VideoGeneratedKeyframeSlots._slot_positions` on a
        // 3×2×3 latent grid at 24 fps. Upstream stores `[start, end)` bounds and
        // this port stores their middle, so the reference is the mean of the pair.
        let reference: [Int: (t: Float, h: [Float], w: [Float])] = [
            0:  (0.020833334, [16, 16, 16, 48, 48, 48], [16, 48, 80, 16, 48, 80]),
            24: (1.0208334,   [16, 16, 16, 48, 48, 48], [16, 48, 80, 16, 48, 80]),
            96: (4.020833,    [16, 16, 16, 48, 48, 48], [16, 48, 80, 16, 48, 80]),
        ]
        for (index, expected) in reference {
            let (guides, _) = GeneratedKeyframeSlots.build(
                pixelFrameIndices: [index], shape: Self.shape)
            let positions = guides[0].positions
            MLX.eval(positions)
            let t = positions[0, 0, 0...].asArray(Float.self)
            let h = positions[0, 1, 0...].asArray(Float.self)
            let w = positions[0, 2, 0...].asArray(Float.self)
            #expect(t.allSatisfy { abs($0 - expected.t) < 1e-5 }, "slot \(index) temporal: \(t)")
            #expect(h == expected.h, "slot \(index) height: \(h)")
            #expect(w == expected.w, "slot \(index) width: \(w)")
        }
    }

    @Test func unpackInvertsBuild() {
        // Slots go out as tokens and come back as a (1, C, K, H, W) latent; a
        // transposed axis here would hand the next stage scrambled anchors that
        // still have the right shape.
        MLXRandom.seed(3)
        let initial = MLXRandom.normal([1, 4, 2, 2, 3]).asType(DType.float32)
        let (guides, layout) = GeneratedKeyframeSlots.build(
            pixelFrameIndices: [0, 40], shape: Self.shape, initial: initial, dtype: .float32)
        let tokens = MLX.concatenated(guides.map { $0.tokens }, axis: 1)
        let back = GeneratedKeyframeSlots.unpack(
            tokens: tokens, layout: layout, shape: Self.shape)
        MLX.eval(back)
        #expect(back.shape == initial.shape)
        #expect(MLX.abs(back - initial).max().item(Float.self) < 1e-6)
    }

    @Test func timestepKeepsGuidesCleanAndSlotsNoisy() {
        // Guides are context (σ=0); slots are being generated (σ=current). Getting
        // this backwards freezes the slot at noise or dissolves the guide.
        let ts = buildExtendedTimestep(sigma: 0.7, originalCount: 4, guideCount: 2, slotCount: 3)
        let values = ts.asArray(Float.self)
        #expect(values.count == 9)
        #expect(values[0 ..< 4].allSatisfy { $0 == 0.7 })
        #expect(values[4 ..< 6].allSatisfy { $0 == 0.0 })
        #expect(values[6 ..< 9].allSatisfy { $0 == 0.7 })
    }

    @Test func markerTouchesOnlyItsRange() {
        let x = MLXArray.zeros([1, 5, 3], dtype: DType.float32)
        let embedding = MLXArray([1.0, 2.0, 3.0] as [Float], [1, 3])
        let marked = applyKeyframeMarker(x, range: 2 ..< 4, embedding: embedding)
        MLX.eval(marked)
        #expect(marked.shape == x.shape)
        let perToken = marked.sum(axis: 2).asArray(Float.self)
        #expect(perToken == [0, 0, 6, 6, 0])
    }

    @Test func markerIsANoOpWithoutARange() {
        let x = MLXArray.ones([1, 4, 3], dtype: DType.float32)
        let same = applyKeyframeMarker(x, range: nil, embedding: MLXArray.ones([1, 3]))
        MLX.eval(same)
        #expect(MLX.abs(same - x).max().item(Float.self) == 0)
    }

    @Test func velocitySliceMatchesTheLayout() {
        let layout = GeneratedKeyframeLayout(
            pixelFrameIndices: [0, 8], tokensPerSlot: 2, firstToken: 5)
        let velocity = MLXArray(0 ..< 27).reshaped([1, 9, 3]).asType(DType.float32)
        let slots = try! #require(sliceSlotVelocity(velocity, layout: layout))
        MLX.eval(slots)
        #expect(slots.shape == [1, 4, 3])
        #expect(slots[0, 0, 0].item(Float.self) == 15)   // token 5, channel 0
        #expect(sliceSlotVelocity(velocity, layout: nil) == nil)
    }

    @Test func invalidRequestsAreRefused() throws {
        #expect(throws: LTXError.self) { try validatedSlotIndices([0, 300], numFrames: 241) }
        #expect(throws: LTXError.self) { try validatedSlotIndices([96, 8], numFrames: 241) }
        #expect(throws: LTXError.self) { try validatedSlotIndices([8, 8], numFrames: 241) }
        #expect(try validatedSlotIndices([0, 96], numFrames: 241) == [0, 96])
    }
}
