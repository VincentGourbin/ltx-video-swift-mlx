//
//  KeyframeInterpolationTests.swift
//  ltx-video-swift-mlx
//
//  Tests for multi-keyframe interpolation helpers.
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
        // Slots 0 and 15 — different latent groups, no collision
        try validateKeyframes([
            KeyframeInput(path: path, pixelFrameIndex: 0),
            KeyframeInput(path: path, pixelFrameIndex: 120)
        ], numFrames: 121)
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
        // Soft conditioning is not yet implemented — values in (0, 1) must be rejected
        // explicitly so users don't silently get hard injection when expecting blending.
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

    @Test func testCollidingLatentSlotsFails() throws {
        // Pixel 1 and pixel 8 both map to latent slot 1 — must be rejected.
        let path = try makeTempImage()
        defer { try? FileManager.default.removeItem(atPath: path) }
        #expect(throws: LTXError.self) {
            try validateKeyframes([
                KeyframeInput(path: path, pixelFrameIndex: 1),
                KeyframeInput(path: path, pixelFrameIndex: 8)
            ], numFrames: 17)
        }
    }
}

// MARK: - buildKeyframeMask

@Suite("buildKeyframeMask")
struct BuildKeyframeMaskTests {
    /// Latent shape: 4 frames × 3 height × 2 width = 24 tokens (12 tokens per latent frame? no — 6 per frame)
    /// Tokens per frame = height × width = 3 × 2 = 6. Total = 4 × 6 = 24.
    private let shape = VideoLatentShape(batch: 1, channels: 128, frames: 4, height: 3, width: 2)

    @Test func testEmptyMaskIsAllZeros() {
        let mask = buildKeyframeMask(latentIndices: [], shape: shape)
        #expect(mask.shape == [1, 24])
        let arr = mask.asArray(Float.self)
        #expect(arr.allSatisfy { $0 == 0.0 })
    }

    @Test func testSingleKeyframeAtFrameZero() {
        let mask = buildKeyframeMask(latentIndices: [0], shape: shape)
        let arr = mask.asArray(Float.self)
        #expect(arr.count == 24)
        // Tokens 0..5 should be 1.0, the rest 0.0
        for t in 0..<6 { #expect(arr[t] == 1.0) }
        for t in 6..<24 { #expect(arr[t] == 0.0) }
    }

    @Test func testKeyframeAtMiddleSlot() {
        let mask = buildKeyframeMask(latentIndices: [2], shape: shape)
        let arr = mask.asArray(Float.self)
        // Tokens 12..17 (slot 2) should be 1.0
        for t in 0..<12 { #expect(arr[t] == 0.0) }
        for t in 12..<18 { #expect(arr[t] == 1.0) }
        for t in 18..<24 { #expect(arr[t] == 0.0) }
    }

    @Test func testMultipleKeyframes() {
        let mask = buildKeyframeMask(latentIndices: [0, 3], shape: shape)
        let arr = mask.asArray(Float.self)
        // Slot 0: tokens 0..5 → 1.0
        // Slot 3: tokens 18..23 → 1.0
        // Slots 1,2: tokens 6..17 → 0.0
        for t in 0..<6 { #expect(arr[t] == 1.0) }
        for t in 6..<18 { #expect(arr[t] == 0.0) }
        for t in 18..<24 { #expect(arr[t] == 1.0) }
    }

    @Test func testOutOfRangeIndexIsIgnored() {
        // Index 99 is way outside the 4-frame shape — should be silently dropped.
        let mask = buildKeyframeMask(latentIndices: [99], shape: shape)
        let arr = mask.asArray(Float.self)
        #expect(arr.allSatisfy { $0 == 0.0 })
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
