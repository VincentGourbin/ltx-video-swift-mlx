// RetakeAudioOnlyE2ETests.swift — Gated E2E for the audio-only retake
// Copyright 2026
//
// `.audioOnly` freezes the picture at σ = 0 and denoises only the audio stream.
// The two claims worth checking end to end are that the returned frames are the
// source frames — not a VAE round-trip of them — and that the audio really was
// regenerated rather than passed through.
//
// Gated behind LTX_E2E_RETAKE=1: it loads the 22B transformer plus the audio
// stack and runs a real (small) retake.
//
// Run:
//   LTX_E2E_RETAKE=1 \
//   xcodebuild -scheme ltx-video-swift-mlx-Package -destination 'platform=macOS' \
//     -derivedDataPath .xcodebuild-tests -skipPackagePluginValidation \
//     -skipMacroValidation -configuration Release test \
//     -only-testing:LTXVideoTests

import Foundation
import Testing
@preconcurrency import MLX
@testable import LTXVideo

@Suite("Audio-only retake E2E (gated: LTX_E2E_RETAKE=1)",
       .enabled(if: ProcessInfo.processInfo.environment["LTX_E2E_RETAKE"] == "1"),
       .serialized)
struct RetakeAudioOnlyE2ETests {

    /// Reference clip shipped with the repo (5 s of French speech, 768x512, with audio).
    static var sourceVideoPath: String {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // LTXVideoTests
            .deletingLastPathComponent()   // Tests
            .deletingLastPathComponent()   // repo root
            .appendingPathComponent("docs/examples/lipdub/lipdub-teaser-french-ours-768x512-121f.mp4")
            .path
    }

    static let width = 384
    static let height = 256
    static let frames = 33

    @Test func audioOnlyRetakeKeepsThePictureAndReplacesTheSound() async throws {
        let source = Self.sourceVideoPath
        try #require(FileManager.default.fileExists(atPath: source),
                     "Source video missing: \(source)")

        let pipeline = LTXPipeline(model: .distilled)
        try await pipeline.loadModels()
        // The audio stream is denoised, so the encoder is needed to turn the
        // source track into the latents it starts from.
        try await pipeline.loadAudioModels(includeEncoder: true)

        let config = LTXVideoGenerationConfig(
            width: Self.width, height: Self.height, numFrames: Self.frames,
            videoPath: source,
            retakeModality: .audioOnly)
        try config.validate()

        let result = try await pipeline.generateRetake(
            prompt: "Heavy rain drumming on a metal roof, distant thunder.",
            config: config,
            upscalerWeightsPath: "")

        #expect(result.frames.dim(0) == Self.frames)

        // The sound is regenerated, so it travels as a waveform. `sourceAudioPath`
        // is the passthrough channel and must stay empty — set, the exporter would
        // mux the original track over the new one.
        let waveform = try #require(result.audioWaveform, "audio-only retake returned no waveform")
        #expect(result.audioSampleRate != nil)
        #expect(result.sourceAudioPath == nil, "passthrough must not be armed for a regenerated track")
        #expect(waveform.dim(waveform.ndim - 1) > 0)

        // The picture must be the source, bit for bit: `.audioOnly` re-muxes the
        // decoded source frames rather than decoding the (untouched) latent, so a
        // VAE round-trip's losses never enter the output.
        let sourceTensor = try await loadVideo(
            from: source, width: Self.width, height: Self.height, numFrames: Self.frames)
        let expected = MLX.clip(
            (sourceTensor.squeezed(axis: 0).transposed(1, 2, 3, 0) + 1.0) / 2.0, min: 0, max: 1)
        MLX.eval(expected)
        #expect(expected.shape == result.frames.shape)
        let maxDiff = MLX.abs(result.frames.asType(.float32) - expected.asType(.float32))
            .max().item(Float.self)
        #expect(maxDiff == 0.0, "picture drifted from the source (maxdiff=\(maxDiff))")
    }

    /// The same clip through `.videoOnly` keeps the source audio as a path and
    /// regenerates the picture — the complementary half of the same switch.
    @Test func videoOnlyRetakeStillPassesTheSourceAudioThrough() async throws {
        let source = Self.sourceVideoPath
        try #require(FileManager.default.fileExists(atPath: source),
                     "Source video missing: \(source)")

        let pipeline = LTXPipeline(model: .distilled)
        try await pipeline.loadModels()

        let config = LTXVideoGenerationConfig(
            width: Self.width, height: Self.height, numFrames: Self.frames,
            videoPath: source,
            retakeModality: .videoOnly)

        let result = try await pipeline.generateRetake(
            prompt: "A person speaking in French, cinematic lighting.",
            config: config,
            upscalerWeightsPath: "")

        #expect(result.frames.dim(0) == Self.frames)
        #expect(result.audioWaveform == nil)
        #expect(result.sourceAudioPath == source)

        // A regenerated picture cannot be the source picture.
        let sourceTensor = try await loadVideo(
            from: source, width: Self.width, height: Self.height, numFrames: Self.frames)
        let expected = MLX.clip(
            (sourceTensor.squeezed(axis: 0).transposed(1, 2, 3, 0) + 1.0) / 2.0, min: 0, max: 1)
        MLX.eval(expected)
        let maxDiff = MLX.abs(result.frames.asType(.float32) - expected.asType(.float32))
            .max().item(Float.self)
        #expect(maxDiff > 0.0, "video-only retake returned the source frames unchanged")
    }

    /// A partial renoise enters the schedule below pure noise: fewer steps run,
    /// and the picture is still untouched.
    @Test func partialAudioStrengthShortensTheScheduleAndKeepsThePicture() async throws {
        let source = Self.sourceVideoPath
        try #require(FileManager.default.fileExists(atPath: source),
                     "Source video missing: \(source)")

        let pipeline = LTXPipeline(model: .distilled)
        try await pipeline.loadModels()
        try await pipeline.loadAudioModels(includeEncoder: true)

        // 0.8 snaps to the trained 0.725: [0.725, 0.421875, 0] — 2 steps of the 8.
        let config = LTXVideoGenerationConfig(
            width: Self.width, height: Self.height, numFrames: Self.frames,
            videoPath: source,
            retakeModality: .audioOnly,
            audioRetakeStrength: 0.8)
        try config.validate()

        let collector = StepCollector()
        let result = try await pipeline.generateRetake(
            prompt: "The same voice, in a larger room with a long reverb tail.",
            config: config,
            upscalerWeightsPath: "",
            onProgress: { progress in
                if progress.phase == .denoising { collector.record(progress.totalSteps) }
            })

        let totals = collector.totals
        #expect(totals == [2],
                "expected the truncated 2-step schedule, saw \(totals.sorted())")
        #expect(result.audioWaveform != nil)

        let sourceTensor = try await loadVideo(
            from: source, width: Self.width, height: Self.height, numFrames: Self.frames)
        let expected = MLX.clip(
            (sourceTensor.squeezed(axis: 0).transposed(1, 2, 3, 0) + 1.0) / 2.0, min: 0, max: 1)
        MLX.eval(expected)
        let maxDiff = MLX.abs(result.frames.asType(.float32) - expected.asType(.float32))
            .max().item(Float.self)
        #expect(maxDiff == 0.0, "picture drifted from the source (maxdiff=\(maxDiff))")
    }

    /// `.audioOnly` without the audio models is a configuration error, not a
    /// silent fallback to a plain video retake.
    @Test func audioOnlyWithoutAudioModelsThrows() async throws {
        let source = Self.sourceVideoPath
        try #require(FileManager.default.fileExists(atPath: source),
                     "Source video missing: \(source)")

        let pipeline = LTXPipeline(model: .distilled)
        try await pipeline.loadModels()  // no loadAudioModels

        let config = LTXVideoGenerationConfig(
            width: Self.width, height: Self.height, numFrames: Self.frames,
            videoPath: source,
            retakeModality: .audioOnly)

        await #expect(throws: LTXError.self) {
            _ = try await pipeline.generateRetake(
                prompt: "Rain on a roof.", config: config, upscalerWeightsPath: "")
        }
    }

    /// Lock-guarded sink for progress callbacks arriving off the test's thread.
    final class StepCollector: @unchecked Sendable {
        private let lock = NSLock()
        private var storage: Set<Int> = []

        func record(_ totalSteps: Int) {
            lock.lock(); defer { lock.unlock() }
            storage.insert(totalSteps)
        }

        var totals: Set<Int> {
            lock.lock(); defer { lock.unlock() }
            return storage
        }
    }
}
