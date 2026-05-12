//
//  AppendedGuideTokensTests.swift
//  ltx-video-swift-mlx
//
//  Tests for the keyframe-as-appended-guide-tokens primitive that backs the
//  issue #21 fix: keyframe latents are concatenated to the video token sequence
//  with shifted RoPE positions and σ=0 timesteps, then cropped after the
//  transformer forward pass.
//

import Testing
import Foundation
@preconcurrency import MLX
@testable import LTXVideo

@Suite("AppendedGuideTokens — primitive helpers")
struct AppendedGuideTokensHelperTests {

    // MARK: - buildExtendedTimestep

    @Test func testExtendedTimestepIsSigmaThenZero() {
        let ts = buildExtendedTimestep(sigma: 0.5, originalCount: 6, guideCount: 3)
        #expect(ts.shape == [1, 9])
        let arr = ts.asArray(Float.self)
        for i in 0..<6 { #expect(arr[i] == 0.5, "original token \(i) should hold sigma") }
        for i in 6..<9 { #expect(arr[i] == 0.0, "guide token \(i) should hold σ=0") }
    }

    @Test func testExtendedTimestepZeroGuides() {
        let ts = buildExtendedTimestep(sigma: 1.0, originalCount: 4, guideCount: 0)
        #expect(ts.shape == [1, 4])
        let arr = ts.asArray(Float.self)
        #expect(arr.allSatisfy { $0 == 1.0 })
    }

    @Test func testExtendedTimestepZeroSigma() {
        // Even when sigma=0 (last denoise step), the timestep tensor should be all zeros —
        // the structure is preserved.
        let ts = buildExtendedTimestep(sigma: 0.0, originalCount: 4, guideCount: 2)
        #expect(ts.shape == [1, 6])
        let arr = ts.asArray(Float.self)
        #expect(arr.allSatisfy { $0 == 0.0 })
    }

    // MARK: - cropToOriginal

    @Test func testCropToOriginalKeepsFirstNTokens() {
        // Build a velocity (1, 5, 3) where each token's first channel = its index
        let values: [Float] = [
            10, 0, 0, 11, 0, 0, 12, 0, 0, 13, 0, 0, 14, 0, 0
        ]
        let v = MLXArray(values, [1, 5, 3])
        let cropped = cropToOriginal(velocity: v, originalCount: 3)
        #expect(cropped.shape == [1, 3, 3])
        let arr = cropped.asArray(Float.self)
        #expect(arr[0] == 10)
        #expect(arr[3] == 11)
        #expect(arr[6] == 12)
    }

    @Test func testCropToOriginalNoCropWhenCountEqualsTotal() {
        let v = MLXArray.zeros([1, 8, 4])
        let cropped = cropToOriginal(velocity: v, originalCount: 8)
        #expect(cropped.shape == [1, 8, 4])
    }
}

@Suite("buildKeyframeGuideToken — keyframe → guide token + positions")
struct BuildKeyframeGuideTokenTests {

    private func makeFakeLatent(h: Int, w: Int) -> MLXArray {
        // shape (1, 128, 1, h, w) — match the VAE-encoded keyframe layout
        return MLXArray.zeros([1, 128, 1, h, w])
    }

    @Test func testOutputShapes() {
        // 8×8 latent (e.g. 256×256 pixel at 32× spatial scale)
        let kf = makeFakeLatent(h: 8, w: 8)
        let guide = buildKeyframeGuideToken(encodedLatent: kf, pixelFrameIndex: 0)
        // 8*8 = 64 tokens, 128 channels (VAE channels) after patchify
        #expect(guide.tokens.shape == [1, 64, 128])
        #expect(guide.positions.shape == [1, 3, 64])
    }

    @Test func testTemporalPositionMatchesPixelFrame() {
        // For num_pixel_frames=1, time = (pixelFrameIndex + 0.5) / fps
        let kf = makeFakeLatent(h: 4, w: 4)
        let fps: Float = 24.0
        for pixelFrame in [0, 8, 16, 32, 60, 120] {
            let guide = buildKeyframeGuideToken(encodedLatent: kf, pixelFrameIndex: pixelFrame, fps: fps)
            let temporal = guide.positions[0, 0, 0...].asArray(Float.self)
            let expected = (Float(pixelFrame) + 0.5) / fps
            for t in temporal {
                #expect(abs(t - expected) < 1e-4, "pixel \(pixelFrame): got \(t), expected \(expected)")
            }
        }
    }

    @Test func testSpatialPositionsAreMiddleOfPixelRange() {
        // Spatial coord for latent index i = i*scale + scale/2 (no fps division)
        let kf = makeFakeLatent(h: 2, w: 2)
        let guide = buildKeyframeGuideToken(encodedLatent: kf, pixelFrameIndex: 0, spatialScale: 32)
        let hCoords = guide.positions[0, 1, 0...].asArray(Float.self)
        let wCoords = guide.positions[0, 2, 0...].asArray(Float.self)
        // Token order: (h=0,w=0), (h=0,w=1), (h=1,w=0), (h=1,w=1)
        // h centers: 16, 48 ; w centers: 16, 48
        #expect(hCoords[0] == 16)
        #expect(hCoords[1] == 16)
        #expect(hCoords[2] == 48)
        #expect(hCoords[3] == 48)
        #expect(wCoords[0] == 16)
        #expect(wCoords[1] == 48)
        #expect(wCoords[2] == 16)
        #expect(wCoords[3] == 48)
    }

    @Test func testFrameZeroMatchesBaseSequenceSlot0Position() {
        // The canonical check: a keyframe at pixel 0 should have the SAME temporal
        // position as the base video sequence's slot 0 (causal_fix + 1-frame range).
        // base slot 0 (causalFix=true): start=0, end=1, middle = 0.5/fps
        // guide @ pixel 0: (0 + 0.5)/fps = 0.5/fps  ✓
        let kf = makeFakeLatent(h: 1, w: 1)
        let guide = buildKeyframeGuideToken(encodedLatent: kf, pixelFrameIndex: 0, fps: 24.0)
        let t = guide.positions[0, 0, 0].item(Float.self)
        #expect(abs(t - 0.5 / 24.0) < 1e-5)
    }

    @Test func testTokensAreFlattenedPatches() {
        // Patchify (1, C=128, 1, H, W) → (1, H*W, C)
        let kf = makeFakeLatent(h: 3, w: 5)
        let guide = buildKeyframeGuideToken(encodedLatent: kf, pixelFrameIndex: 0)
        #expect(guide.tokens.shape == [1, 15, 128])
    }
}
