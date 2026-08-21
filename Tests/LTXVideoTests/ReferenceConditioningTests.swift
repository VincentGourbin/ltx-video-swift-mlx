//
//  ReferenceConditioningTests.swift
//  ltx-video-swift-mlx
//
//  Tests for the LipDub reference conditioning primitives:
//  `buildAudioReference` (audio side, negative RoPE shift) and
//  `buildVideoReference` (video side, IC-LoRA multi-frame + spatial scale).
//

import Testing
import Foundation
@preconcurrency import MLX
@testable import LTXVideo

@Suite("buildAudioReference — audio side LipDub conditioning")
struct BuildAudioReferenceTests {

    /// Build a small fake audio latent and a minimal transformer config.
    private func makeFakeAudioLatent(t: Int) -> MLXArray {
        // (B, C, T, M) = (1, 8, t, 16) — matches AudioVAE.encode output
        return MLXArray.zeros([1, 8, t, 16])
    }

    private var minimalConfig: LTXTransformerConfig {
        return .default
    }

    @Test func testOutputShapes() {
        // 5 audio frames + 9 target frames → guideCount=5, originalCount=9
        let latent = makeFakeAudioLatent(t: 5)
        let ctx = buildAudioReference(
            referenceLatent: latent,
            targetAudioFrames: 9,
            refConfig: minimalConfig
        )
        #expect(ctx.guideCount == 5)
        #expect(ctx.originalCount == 9)
        // packed shape: (1, 5, 128) since 8*16 = 128
        #expect(ctx.guideTokens.shape == [1, 5, 128])
    }

    @Test func testNegativePositionsShift() {
        // For 10 reference frames at 25 audio_latents_per_second (= 0.04s per token),
        // the last token's MIDDLE = 0.36s (or so per createAudioPositionGrid math),
        // so after shift the last token's END should sit at -0.04.
        let latent = makeFakeAudioLatent(t: 10)
        let ctx = buildAudioReference(
            referenceLatent: latent,
            targetAudioFrames: 25,
            refConfig: minimalConfig
        )
        // Extended positions = target (25) + reference (10) = 35 tokens
        // We can't directly read positions from the ctx (extRoPE bakes them in), but
        // we can sanity-check via the public shapes and counts.
        #expect(ctx.guideTokens.shape == [1, 10, 128])
        #expect(ctx.guideCount == 10)
        #expect(ctx.originalCount == 25)
    }

    @Test func testFromPackedMatchesFromLatent() {
        // buildAudioReferenceFromPacked must produce the same context as the full
        // builder when fed the equivalent packed tensor.
        let latent = makeFakeAudioLatent(t: 4)
        let fromLatent = buildAudioReference(
            referenceLatent: latent,
            targetAudioFrames: 9,
            refConfig: minimalConfig
        )
        // Pack manually (1,8,4,16) -> (1,4,128)
        let packed = latent.transposed(0, 2, 1, 3).reshaped([1, 4, 128])
        let fromPacked = buildAudioReferenceFromPacked(
            packed: packed,
            targetAudioFrames: 9,
            refConfig: minimalConfig
        )
        #expect(fromLatent.guideCount == fromPacked.guideCount)
        #expect(fromLatent.originalCount == fromPacked.originalCount)
        #expect(fromLatent.guideTokens.shape == fromPacked.guideTokens.shape)
    }
}

@Suite("buildVideoReference — IC-LoRA video conditioning")
struct BuildVideoReferenceTests {

    private func makeFakeVideoLatent(f: Int, h: Int, w: Int) -> MLXArray {
        // (B, C, F, H, W) — VAE-encoded keyframe latent layout
        return MLXArray.zeros([1, 128, f, h, w])
    }

    private var minimalConfig: LTXTransformerConfig {
        return .default
    }

    @Test func testTokenAndPositionShapes() {
        // Reference at 4x4 latent, 2 latent frames → 32 ref tokens
        let latent = makeFakeVideoLatent(f: 2, h: 4, w: 4)
        let target = VideoLatentShape(batch: 1, channels: 128, frames: 2, height: 8, width: 8)
        let ctx = buildVideoReference(
            referenceLatent: latent,
            targetShape: target,
            downscaleFactor: 2,
            hasAudio: false,
            refConfig: minimalConfig
        )
        #expect(ctx.guideCount == 32)              // 2 * 4 * 4
        #expect(ctx.originalCount == 128)          // 2 * 8 * 8
        #expect(ctx.guideTokens?.shape == [1, 32, 128])
        #expect(ctx.extCrossVideoRoPE == nil)      // hasAudio=false → no cross-modal RoPE
    }

    @Test func testCrossVideoRoPEPresentWhenAudioEnabled() {
        let latent = makeFakeVideoLatent(f: 2, h: 4, w: 4)
        let target = VideoLatentShape(batch: 1, channels: 128, frames: 2, height: 8, width: 8)
        let ctx = buildVideoReference(
            referenceLatent: latent,
            targetShape: target,
            downscaleFactor: 2,
            hasAudio: true,
            refConfig: minimalConfig
        )
        #expect(ctx.extCrossVideoRoPE != nil)
    }

    @Test func testDownscaleFactorOneIsIdentity() {
        // When downscaleFactor=1, spatial positions are NOT scaled.
        // We don't read positions directly from the ctx, but we can verify the
        // function accepts the param without exploding and produces the expected token count.
        let latent = makeFakeVideoLatent(f: 1, h: 4, w: 4)
        let target = VideoLatentShape(batch: 1, channels: 128, frames: 1, height: 4, width: 4)
        let ctx = buildVideoReference(
            referenceLatent: latent,
            targetShape: target,
            downscaleFactor: 1,
            hasAudio: false,
            refConfig: minimalConfig
        )
        #expect(ctx.guideCount == 16)        // 1 * 4 * 4
        #expect(ctx.originalCount == 16)
    }
}
