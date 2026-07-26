//
//  VideoTailReadTests.swift
//  ltx-video-swift-mlx
//
//  Guards `loadVideo(tail: true)` — the native replacement for the ffmpeg
//  tail-clip recipe the continuation anchor used to require. Runs against the
//  reference clip shipped in docs/examples (121 frames, 768x512).
//

import Testing
import Foundation
@preconcurrency import MLX
@testable import LTXVideo

@Suite("Video tail read")
struct VideoTailReadTests {

    static var referenceVideoPath: String {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // LTXVideoTests
            .deletingLastPathComponent()   // Tests
            .deletingLastPathComponent()   // repo root
            .appendingPathComponent("docs/examples/lipdub/lipdub-teaser-french-ours-768x512-121f.mp4")
            .path
    }

    /// The tail read must return the requested frame count at the requested
    /// resolution, and it must NOT be the uniform read: on a 121-frame clip the
    /// last 9 frames span 0.375 s while the uniform read spans the whole 5 s, so
    /// the two tensors differ substantially.
    @Test func tailReadReturnsTheEndNotTheWholeClip() async throws {
        let path = Self.referenceVideoPath
        try #require(FileManager.default.fileExists(atPath: path),
                     "reference clip missing: \(path)")

        let w = 192, h = 128
        let tail = try await loadVideo(from: path, width: w, height: h, numFrames: 9, tail: true)
        let uniform = try await loadVideo(from: path, width: w, height: h, numFrames: 9)
        MLX.eval(tail, uniform)

        #expect(tail.shape == [1, 3, 9, h, w])
        #expect(uniform.shape == tail.shape)

        // A 0.375 s window is far more self-similar than a 5 s one: consecutive
        // tail frames differ much less than consecutive uniform-sample frames.
        func meanAdjacentDelta(_ x: MLXArray) -> Float {
            let a = x[0..., 0..., 0..<8, 0..., 0...]
            let b = x[0..., 0..., 1..<9, 0..., 0...]
            return MLX.abs(b - a).mean().item(Float.self)
        }
        let tailDelta = meanAdjacentDelta(tail)
        let uniformDelta = meanAdjacentDelta(uniform)
        #expect(tailDelta < uniformDelta,
                "tail frames should be more similar than uniform samples: tail=\(tailDelta) uniform=\(uniformDelta)")

        // And the two reads genuinely differ — the tail is not silently the
        // uniform read (which would make the anchor wrong without any error).
        let between = MLX.abs(tail - uniform).mean().item(Float.self)
        #expect(between > 0.01, "tail and uniform reads are identical (\(between))")
    }

    /// The last frame of the tail read is the clip's last frame, which the
    /// uniform read also samples. They must match closely (same source frame,
    /// same decode path) — this pins the tail's alignment to the end.
    @Test func tailEndsOnTheSameFinalFrameAsTheUniformRead() async throws {
        let path = Self.referenceVideoPath
        try #require(FileManager.default.fileExists(atPath: path))

        let w = 192, h = 128
        let tail = try await loadVideo(from: path, width: w, height: h, numFrames: 9, tail: true)
        let uniform = try await loadVideo(from: path, width: w, height: h, numFrames: 9)
        MLX.eval(tail, uniform)

        let tailLast = tail[0..., 0..., 8..<9, 0..., 0...]
        let uniformLast = uniform[0..., 0..., 8..<9, 0..., 0...]
        let delta = MLX.abs(tailLast - uniformLast).mean().item(Float.self)
        #expect(delta < 0.05, "tail should end on the clip's final frame (delta \(delta))")
    }

    /// Asking for more frames than the video holds must throw, not silently
    /// clamp — a short clip cannot provide a 9-frame anchor.
    @Test func tailReadThrowsWhenTheVideoIsTooShort() async throws {
        let path = Self.referenceVideoPath
        try #require(FileManager.default.fileExists(atPath: path))

        await #expect(throws: LTXError.self) {
            _ = try await loadVideo(from: path, width: 64, height: 64, numFrames: 100_000, tail: true)
        }
    }
}
