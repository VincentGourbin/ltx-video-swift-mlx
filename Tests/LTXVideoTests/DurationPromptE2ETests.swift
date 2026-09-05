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
struct DurationPromptE2ETests {

    static var cacheRoot: URL {
        URL(fileURLWithPath: ProcessInfo.processInfo.environment["LTX25_CACHE_ROOT"]!)
    }

    /// One pipeline for the whole suite, held statically.
    ///
    /// swift-testing builds a **new suite instance per `@Test`**, so an instance
    /// property shares nothing — three sub-second forwards would sit behind
    /// three full 70 GB loads. `customModelsDirectory` is a process-global; it is
    /// pointed here once, alongside the pipeline it configures, rather than in a
    /// per-instance init/deinit pair whose ordering depends on ARC.
    static let shared: LTXPipeline = {
        LTXModelRegistry.customModelsDirectory = cacheRoot
        return LTXPipeline(model: .v25Distilled)
    }()

    func pipeline() -> LTXPipeline { Self.shared }

    static let scene = "A quiet late-night laundromat with flickering fluorescent lights."

    /// A written duration does not reach the head.
    ///
    /// Measured August 2026 (video-only tokens): prefixing the same scene with
    /// "3 seconds." returned byte-identical output to no prefix at all — 23.5 s
    /// both times.
    ///
    /// Re-measured 2026-09-05 after `fix/duration-head-audio-tokens` made the
    /// head see the audio connector's tokens too, as upstream does (F7/F8 in
    /// the PR): 4.09375 s bare vs 3.59375 s with the prefix — a real ~0.5 s
    /// move, no longer byte-identical. The head still isn't parsing the
    /// numeral (a literal read would land on 3.0 s exactly, not 3.59), but it
    /// is more sensitive to the prefix's wording now that audio tokens are in
    /// the mix — noted for the PR rather than re-investigated here.
    @Test func aWrittenDurationIsNotAnInstruction() async throws {
        let pipeline = pipeline()

        let bare = try await pipeline.predictFrameCount(for: Self.scene)
        let asksForThree = try await pipeline.predictFrameCount(for: "3 seconds. \(Self.scene)")

        #expect(abs(bare.seconds - asksForThree.seconds) < 1.0,
                "a '3 seconds' prefix moved the prediction: \(bare.seconds)s -> \(asksForThree.seconds)s")
        // If this ever lands on 3.0s exactly, the head has started reading
        // durations and this whole suite's premise needs revisiting.
        #expect(abs(asksForThree.seconds - 3.0) > 0.1,
                "asking for 3 s returned \(asksForThree.seconds)s, suspiciously close to a literal read")
    }

    /// Style-dense prompts read as short shots.
    ///
    /// The prompt below asks for 15 seconds. It is saturated with
    /// close-framing language ("phone-camera feel", "delayed autofocus at
    /// close range"), which is what the head actually weighs.
    ///
    /// Measured 121 frames (video-only) in August 2026; re-measured 153 frames
    /// (6.6875 s) on 2026-09-05 once the head also sees audio tokens
    /// (`fix/duration-head-audio-tokens`) — still nowhere near the 361 frames
    /// the prompt asks for.
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
        #expect(result.frames == 153,
                "expected 153 frames, got \(result.frames) (\(result.seconds)s)")
        #expect(!result.wasClamped)
        // The point of the test: nowhere near the 361 frames the prompt asks for.
        #expect(result.frames < 200)
    }

    /// Historically this scene was the example of a prompt that hits the
    /// ceiling: video-only tokens predicted 27.0 s → clamped to 473 frames.
    ///
    /// Re-measured 2026-09-05 after `fix/duration-head-audio-tokens`: feeding
    /// the head both video and audio connector tokens (matching upstream,
    /// F7/F8) pushes this same scene down to 4.09375 s / 97 frames,
    /// unclamped. Nothing here demonstrates the 473-frame ceiling any more —
    /// that grid-snap behavior is covered on synthetic input by the pure
    /// `DurationGridSnap` suite instead.
    @Test func aSceneThatUsedToClampNoLongerDoesWithAudioTokens() async throws {
        let pipeline = pipeline()
        let result = try await pipeline.predictFrameCount(for: Self.scene)
        #expect(!result.wasClamped)
        #expect(result.frames == 97)
    }
}
