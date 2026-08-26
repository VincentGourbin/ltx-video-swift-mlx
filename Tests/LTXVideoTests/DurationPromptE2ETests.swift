// DurationPromptE2ETests.swift — what the duration head does with real prompts
// Copyright 2026
//
// DurationHeadE2ETests pins the arithmetic on synthetic tokens. This drives the
// whole path — Gemma 4 encode, connector, head — on prompts a user would
// actually write, because the recurring question about `--frames auto` is not
// "is the forward correct" but "why did it ignore the duration I asked for".
//
// It ignores it because the head never sees text. It reads the connector's
// output: a semantic conditioning signal for the diffusion transformer, pooled
// by a single learned query into one 256-dim vector and regressed to one
// scalar. There is no path by which a numeral survives as a number. The
// controls below demonstrate that rather than asserting it.
//
// Needs the LTX-2.5 checkpoint (gated, ~70 GB) laid out as the *downloader*
// expects — a cache ROOT with per-component subdirectories:
//   <root>/ltx-2.5-distilled/*.safetensors
//   <root>/ltx-2.5-duration-head/*.safetensors
//
// Deliberately NOT `LTX25_MODELS_DIR`: four existing suites
// (DurationHeadE2ETests, VAERoundTripE2ETests, BigVGANVocoderE2ETests,
// LTX25CheckpointSourceE2ETests) read that as a FLAT directory of safetensors.
// One directory cannot be both, and overloading it would either send this suite
// on a 70 GB re-download or break all four.
//
// Run:
//   TEST_RUNNER_LTX25_CACHE_ROOT=/path/to/models \
//   xcodebuild -scheme ltx-video-swift-mlx-Package -destination 'platform=macOS' \
//     -derivedDataPath .xcodebuild-tests -skipPackagePluginValidation \
//     -skipMacroValidation -configuration Debug test \
//     -only-testing:LTXVideoTests/DurationPromptE2ETests

import Foundation
import Testing
@preconcurrency import MLX
@testable import LTXVideo

@Suite("Duration head on real prompts (gated: LTX25_CACHE_ROOT)",
       .enabled(if: ProcessInfo.processInfo.environment["LTX25_CACHE_ROOT"] != nil),
       .serialized)
final class DurationPromptE2ETests {

    static var cacheRoot: URL {
        URL(fileURLWithPath: ProcessInfo.processInfo.environment["LTX25_CACHE_ROOT"]!)
    }

    /// `customModelsDirectory` is process-global and shared with every other
    /// suite, so it is restored on teardown rather than left pointing here.
    private let previousModelsDirectory = LTXModelRegistry.customModelsDirectory

    /// One pipeline for the suite: each `predictFrameCount` is a sub-second
    /// forward behind a multi-minute load, and building three meant three loads.
    private let shared: LTXPipeline

    init() {
        LTXModelRegistry.customModelsDirectory = Self.cacheRoot
        shared = LTXPipeline(model: .v25Distilled)
    }

    deinit {
        LTXModelRegistry.customModelsDirectory = previousModelsDirectory
    }

    func pipeline() -> LTXPipeline { shared }

    static let scene = "A quiet late-night laundromat with flickering fluorescent lights."

    /// A written duration does not reach the head.
    ///
    /// Measured August 2026: prefixing the same scene with "3 seconds." returns
    /// byte-identical output to no prefix at all — 23.5 s both times. "15
    /// seconds." and "20 seconds." *do* move it, but to 16.9 s and 19.5 s: the
    /// head responds to what those tokens change in the scene representation,
    /// by learned correlation, not by reading them as quantities. If it parsed
    /// the numeral, "3 seconds" would return 3 s.
    @Test func aWrittenDurationIsNotAnInstruction() async throws {
        let pipeline = pipeline()

        let bare = try await pipeline.predictFrameCount(for: Self.scene)
        let asksForThree = try await pipeline.predictFrameCount(for: "3 seconds. \(Self.scene)")

        #expect(abs(bare.seconds - asksForThree.seconds) < 0.01,
                "a '3 seconds' prefix moved the prediction: \(bare.seconds)s -> \(asksForThree.seconds)s")
        // If this ever approaches 3, the head has started reading durations and
        // this whole suite's premise needs revisiting.
        #expect(asksForThree.seconds > 10,
                "asking for 3 s returned \(asksForThree.seconds)s")
    }

    /// Style-dense prompts read as short shots.
    ///
    /// The prompt below asks for 15 seconds and predicts ~5. It is saturated
    /// with close-framing language ("phone-camera feel", "delayed autofocus at
    /// close range"), which is what the head actually weighs.
    @Test func aStyleDensePromptPredictsAShortShot() async throws {
        let pipeline = pipeline()
        let prompt = """
            15 seconds, 16:9 landscape. Combine a live-action late-night \
            laundromat with hand-drawn luminous animation. The small \
            self-service laundromat has gently flickering fluorescent lights, \
            running washers, plastic baskets, a worn bench, and one sock on the \
            floor. Keep the space quiet and faintly nostalgic.

            Use a one-handed phone-camera feel with visible shake, exposure \
            fluctuation under white fluorescent light, environmental reflections \
            in glass, and delayed autofocus at close range. Avoid polished \
            commercial composition; it should feel like an authentic late-night \
            encounter, filmed while following a strange apparition.
            """

        let result = try await pipeline.predictFrameCount(for: prompt)
        #expect(result.frames == 121,
                "expected 121 frames, got \(result.frames) (\(result.seconds)s)")
        #expect(!result.wasClamped)
        // The point of the test: nowhere near the 361 frames the prompt asks for.
        #expect(result.frames < 200)
    }

    /// The effective ceiling of `--frames auto` is 473, not the 481 the config
    /// allows: 20 s x 24 fps = 480, and 480 floors to 473 on the 8k+1 grid.
    @Test func theCeilingIs473Frames() async throws {
        let pipeline = pipeline()
        let result = try await pipeline.predictFrameCount(for: Self.scene)
        #expect(result.wasClamped)
        #expect(result.frames == 473)
    }
}
