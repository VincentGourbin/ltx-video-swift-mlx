//
//  KeyframeInterpolationTests.swift
//  ltx-video-swift-mlx
//
//  Tests for multi-keyframe interpolation public API (KeyframeInput, validation,
//  pixel→latent index mapping). The append-based conditioning math is covered in
//  AppendedGuideTokensTests.
//

import Testing
import Foundation
import MLX
@testable import LTXVideo

// MARK: - pixelFrameToLatentFrame

@Suite("pixelFrameToLatentFrame")
struct PixelFrameToLatentFrameTests {
    @Test func testPixelZeroMapsToLatentZero() {
        #expect(pixelFrameToLatentFrame(0) == 0)
    }

    @Test func testFirstEightPixelsMapToLatentOne() {
        for p in 1...8 {
            #expect(pixelFrameToLatentFrame(p) == 1, "pixel \(p) should map to latent 1")
        }
    }

    @Test func testNextEightPixelsMapToLatentTwo() {
        for p in 9...16 {
            #expect(pixelFrameToLatentFrame(p) == 2, "pixel \(p) should map to latent 2")
        }
    }

    @Test func testHigherIndices() {
        #expect(pixelFrameToLatentFrame(120) == 15)  // 121-frame video, last pixel
        #expect(pixelFrameToLatentFrame(240) == 30)  // 241-frame video, last pixel
        #expect(pixelFrameToLatentFrame(8 * 30) == 30)
        #expect(pixelFrameToLatentFrame(8 * 30 + 1) == 31)
    }

    @Test func testNegativeClampsToZero() {
        #expect(pixelFrameToLatentFrame(-5) == 0)
    }
}

// MARK: - validateKeyframes

@Suite("validateKeyframes")
struct ValidateKeyframesTests {
    /// Create a temporary file we can point keyframes at.
    private func makeTempImage() throws -> String {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("kf-test-\(UUID().uuidString).png")
        try Data([0x89, 0x50, 0x4E, 0x47]).write(to: url)
        return url.path
    }

    @Test func testEmptyListAlwaysValid() throws {
        try validateKeyframes([], numFrames: 121)
    }

    @Test func testValidSingleKeyframe() throws {
        let path = try makeTempImage()
        defer { try? FileManager.default.removeItem(atPath: path) }
        try validateKeyframes([KeyframeInput(path: path, pixelFrameIndex: 0)], numFrames: 121)
    }

    @Test func testValidMultiKeyframe() throws {
        let path = try makeTempImage()
        defer { try? FileManager.default.removeItem(atPath: path) }
        try validateKeyframes([
            KeyframeInput(path: path, pixelFrameIndex: 0),
            KeyframeInput(path: path, pixelFrameIndex: 60),
            KeyframeInput(path: path, pixelFrameIndex: 120)
        ], numFrames: 121)
    }

    /// With the append-based conditioning, two keyframes within the same 8-frame
    /// latent stride group are allowed — each appended guide token has its own
    /// RoPE temporal position derived from `pixelFrameIndex`, not the latent slot.
    @Test func testSameLatentStrideGroupIsAllowed() throws {
        let path = try makeTempImage()
        defer { try? FileManager.default.removeItem(atPath: path) }
        // Pixel 1 and pixel 8 both map to latent slot 1 — now permitted.
        try validateKeyframes([
            KeyframeInput(path: path, pixelFrameIndex: 1),
            KeyframeInput(path: path, pixelFrameIndex: 8)
        ], numFrames: 17)
    }

    @Test func testMissingFileFails() {
        let bogus = "/tmp/this-keyframe-does-not-exist-\(UUID().uuidString).png"
        #expect(throws: LTXError.self) {
            try validateKeyframes([KeyframeInput(path: bogus, pixelFrameIndex: 0)], numFrames: 9)
        }
    }

    @Test func testFrameIndexOutOfRangeFails() throws {
        let path = try makeTempImage()
        defer { try? FileManager.default.removeItem(atPath: path) }
        #expect(throws: LTXError.self) {
            try validateKeyframes([KeyframeInput(path: path, pixelFrameIndex: 121)], numFrames: 121)
        }
        #expect(throws: LTXError.self) {
            try validateKeyframes([KeyframeInput(path: path, pixelFrameIndex: -1)], numFrames: 121)
        }
    }

    @Test func testStrengthZeroFails() throws {
        let path = try makeTempImage()
        defer { try? FileManager.default.removeItem(atPath: path) }
        #expect(throws: LTXError.self) {
            try validateKeyframes([KeyframeInput(path: path, pixelFrameIndex: 0, strength: 0.0)], numFrames: 9)
        }
    }

    @Test func testStrengthAboveOneFails() throws {
        let path = try makeTempImage()
        defer { try? FileManager.default.removeItem(atPath: path) }
        #expect(throws: LTXError.self) {
            try validateKeyframes([KeyframeInput(path: path, pixelFrameIndex: 0, strength: 1.5)], numFrames: 9)
        }
    }

    @Test func testStrengthBelowOneFails() throws {
        // Soft conditioning is not yet implemented — values in (0, 1) must be rejected.
        let path = try makeTempImage()
        defer { try? FileManager.default.removeItem(atPath: path) }
        #expect(throws: LTXError.self) {
            try validateKeyframes([KeyframeInput(path: path, pixelFrameIndex: 0, strength: 0.5)], numFrames: 9)
        }
    }

    @Test func testDuplicatePixelIndexFails() throws {
        let path = try makeTempImage()
        defer { try? FileManager.default.removeItem(atPath: path) }
        #expect(throws: LTXError.self) {
            try validateKeyframes([
                KeyframeInput(path: path, pixelFrameIndex: 8),
                KeyframeInput(path: path, pixelFrameIndex: 8)
            ], numFrames: 17)
        }
    }
}

// MARK: - KeyframeInput

@Suite("KeyframeInput")
struct KeyframeInputTests {
    @Test func testDefaultStrengthIsOne() {
        let kf = KeyframeInput(path: "/tmp/x.png", pixelFrameIndex: 0)
        #expect(kf.strength == 1.0)
    }

    @Test func testEquatable() {
        let a = KeyframeInput(path: "/x.png", pixelFrameIndex: 5, strength: 0.8)
        let b = KeyframeInput(path: "/x.png", pixelFrameIndex: 5, strength: 0.8)
        let c = KeyframeInput(path: "/y.png", pixelFrameIndex: 5, strength: 0.8)
        #expect(a == b)
        #expect(a != c)
    }
}
