// DurationHeadE2ETests.swift — Gated numerical check of the LTX-2.5 duration head
// Copyright 2026
//
// The head is a regression model: its output is a number nobody can eyeball. A
// mis-ported attention pooler still returns a plausible-looking duration, so the
// port is pinned against upstream's own module on a deterministic synthetic
// input: `scripts/duration_head_reference.py` runs
// `ltx_core.duration_head.DurationHead` over the same weights.
//
//   torch float64 => log_duration = 2.4338593, seconds = 11.402804284
//   torch float32 => log_duration = 2.4338598, seconds = 11.402810097
//
// This previously pinned against a NumPy re-implementation written during the
// same porting effort, which could only catch a transcription slip — not a
// shared misreading. Running upstream also settled `num_pooler_heads = 4`,
// which the checkpoint metadata does not carry (`"duration_head": {}`).
//
// Gated behind LTX25_MODELS_DIR.
//
// Run:
//   TEST_RUNNER_LTX25_MODELS_DIR=/Volumes/Lexar/models/ltx-2.5 \
//   xcodebuild -scheme ltx-video-swift-mlx-Package -destination 'platform=macOS' \
//     -derivedDataPath .xcodebuild-tests -skipPackagePluginValidation \
//     -skipMacroValidation -configuration Debug test \
//     -only-testing:LTXVideoTests/DurationHeadE2ETests

import Foundation
import Testing
@preconcurrency import MLX
@testable import LTXVideo

@Suite("LTX-2.5 duration head (gated: LTX25_MODELS_DIR)",
       .enabled(if: ProcessInfo.processInfo.environment["LTX25_MODELS_DIR"] != nil),
       .serialized)
struct DurationHeadE2ETests {

    static var headURL: URL {
        URL(fileURLWithPath: ProcessInfo.processInfo.environment["LTX25_MODELS_DIR"] ?? "")
            .appendingPathComponent("ltx-2.5-duration-head-bf16.safetensors")
    }

    /// The same deterministic input the NumPy reference used: `sin(i/10 + j/100)`.
    static func syntheticTokens(tokens: Int = 8, dim: Int = 4096) -> MLXArray {
        let rows = MLXArray(0 ..< tokens).asType(.float32).reshaped([tokens, 1]) * 0.1
        let cols = MLXArray(0 ..< dim).asType(.float32).reshaped([1, dim]) * 0.01
        return MLX.sin(rows + cols).expandedDimensions(axis: 0)
    }

    @Test func matchesTheUpstreamReference() throws {
        let head = try LTXDurationHead.load(from: Self.headURL)
        let seconds = try head.predictSeconds(
            videoTokens: Self.syntheticTokens(), audioTokens: nil)

        // bf16 weights through a 4096-wide projection: ~1% is the precision floor,
        // and far tighter than any plausible mis-port (a transposed packed
        // projection or a wrong head split moves this by tens of percent).
        #expect(abs(seconds - 11.402804) / 11.402804 < 0.01,
                "predicted \(seconds)s against upstream's 11.402804s")
    }

    @Test func snapsFramesToTheGridAndReportsClamping() throws {
        let head = try LTXDurationHead.load(from: Self.headURL)
        let tokens = Self.syntheticTokens()

        let normal = try head.predictFrameCount(
            videoTokens: tokens, audioTokens: nil, frameRate: 24.0)
        #expect((normal.frames - 1) % 8 == 0)
        #expect(normal.wasClamped == false)
        #expect(normal.frames == 273)   // 11.4028 s x 24 = 273.7 -> 273 on the grid

        // A ceiling below the prediction must clamp, and say so.
        let clamped = try head.predictFrameCount(
            videoTokens: tokens, audioTokens: nil, frameRate: 24.0,
            minSeconds: 1.0, maxSeconds: 5.0)
        #expect(clamped.wasClamped)
        #expect(clamped.frames <= 121)
        #expect((clamped.frames - 1) % 8 == 0)
    }
}

@Suite("Duration grid snapping (pure)")
struct DurationGridSnapTests {

    /// Every value here came out of `scripts/duration_head_reference.py`, which
    /// runs upstream's own `seconds_to_clamped_num_frames` with the defaults
    /// `DurationPredictor.__call__` uses (1 s / 20 s @ 24 fps).
    ///
    /// Pure arithmetic, so unlike the rest of this file it needs no checkpoint.
    /// The four middle rows are durations the head actually produced on real
    /// prompts, including one that clamps.
    @Test(arguments: [
        (Float(5.15625), 121),   // measured, a phone-camera style prompt
        (Float(5.28125), 121),   // same prompt after enhancement
        (Float(16.875), 401),
        (Float(19.5), 465),
        (Float(23.5), 473),      // above the 20 s ceiling -> clamped
        (Float(0.5), 25),        // below the 1 s floor -> snapped up off the floor
        (Float(20.0), 473),      // the ceiling itself is 473, not 481
    ])
    func matchesUpstreamSecondsToFrames(seconds: Float, expected: Int) {
        let frames = LTXDurationHead.snapToGrid(
            seconds: seconds, frameRate: 24, minSeconds: 1.0, maxSeconds: 20.0)
        #expect(frames == expected, "\(seconds)s -> \(frames), upstream says \(expected)")
        #expect((frames - 1) % 8 == 0)
    }

    @Test func gridlessWindowStaysOnGrid() {
        // min = max = 5 s @ 24 fps → [120, 120] contains no 8k+1 point.
        // The old cap min(121, 120) returned 120 — off the grid, and the
        // documented handoff to LTXVideoGenerationConfig would throw.
        let f = LTXDurationHead.snapToGrid(seconds: 5.0, frameRate: 24, minSeconds: 5, maxSeconds: 5)
        #expect((f - 1) % 8 == 0, "must stay on the 8k+1 grid, got \(f)")
    }

    @Test func normalWindowsRespectBounds() {
        for (sec, minS, maxS) in [(Float(14.04), Float(1), Float(20)), (0.2, 1, 20), (99, 1, 20)] {
            let f = LTXDurationHead.snapToGrid(seconds: sec, frameRate: 24, minSeconds: minS, maxSeconds: maxS)
            #expect((f - 1) % 8 == 0)
            #expect(f >= Int((minS * 24).rounded()))
            #expect(f <= Int((maxS * 24).rounded()))
        }
    }
}
