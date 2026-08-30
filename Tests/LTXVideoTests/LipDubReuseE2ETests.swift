// LipDubReuseE2ETests.swift — Gated E2E for fused-LoRA reuse across consecutive runs (PR #36, B6)
// Copyright 2026
//
// The reuse behavior CANNOT be tested through the CLI: every invocation is a new
// process, so the model reloads regardless. This test drives two consecutive
// generateLipDub calls in one process and asserts the state machine:
//   run 1 fuses → run 2 reuses (no double-fusion) → other pipelines throw while
//   fused → switching LoRA throws.
//
// Gated behind LTX_E2E_LIPDUB=1 — it loads the full 22B + audio stack and runs
// two real (small) generations (~10-20 min, needs ≥ 48 GB RAM since the
// transformer must survive between runs: MemoryOptimizationConfig.disabled).
//
// Run:
//   LTX_E2E_LIPDUB=1 [LTX_E2E_LIPDUB_LORA=/path/to/ic-lora.safetensors] \
//   xcodebuild -scheme ltx-video-swift-mlx-Package -destination 'platform=macOS' \
//     -derivedDataPath .xcodebuild-tests -skipPackagePluginValidation \
//     -skipMacroValidation -configuration Release test \
//     -only-testing:LTXVideoTests

import Foundation
import Testing
@preconcurrency import MLX
@testable import LTXVideo

@Suite("LipDub fused-LoRA reuse E2E (gated: LTX_E2E_LIPDUB=1)",
       .enabled(if: ProcessInfo.processInfo.environment["LTX_E2E_LIPDUB"] == "1"),
       .serialized)
struct LipDubReuseE2ETests {

    /// Reference video shipped with the repo (5 s of French speech, 768x512).
    static var referenceVideoPath: String {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // LTXVideoTests
            .deletingLastPathComponent()   // Tests
            .deletingLastPathComponent()   // repo root
            .appendingPathComponent("docs/examples/lipdub/lipdub-teaser-french-ours-768x512-121f.mp4")
            .path
    }

    static func resolveLoRAPath() async throws -> String {
        if let provided = ProcessInfo.processInfo.environment["LTX_E2E_LIPDUB_LORA"] {
            try #require(FileManager.default.fileExists(atPath: provided),
                         "LTX_E2E_LIPDUB_LORA points to a missing file: \(provided)")
            return provided
        }
        let downloader = ModelDownloader(hfToken: ProcessInfo.processInfo.environment["HF_TOKEN"])
        return try await downloader.downloadLipDubLoRA { _ in }.path
    }

    @Test func consecutiveRunsReuseFusedTransformer() async throws {
        let refVideo = Self.referenceVideoPath
        try #require(FileManager.default.fileExists(atPath: refVideo),
                     "Reference video missing: \(refVideo)")
        let loraPath = try await Self.resolveLoRAPath()

        // unloadAfterUse == false is the reuse precondition: the fused transformer
        // must survive between runs.
        let pipeline = LTXPipeline(model: .distilled, memoryOptimization: .disabled)
        try await pipeline.loadModels()
        // includeEncoder: video-reference LipDub encodes the source audio track
        // (same as the CLI's lipdub command).
        try await pipeline.loadAudioModels(includeEncoder: true)
        let upscalerPath = try await pipeline.downloadUpscalerWeights()

        #expect(await pipeline.fusedLipDubLoRAPath == nil, "pristine after load")

        let config = LTXVideoGenerationConfig(width: 384, height: 256, numFrames: 33)
        let prompt = "A person speaking in French saying: \"Bonjour à tous, ceci est un test.\""

        // Run 1 — fuses the IC-LoRA.
        let t0 = Date()
        let result1 = try await pipeline.generateLipDub(
            prompt: prompt,
            referenceVideoPath: refVideo,
            lipdubLoraPath: loraPath,
            config: config,
            upscalerWeightsPath: upscalerPath
        )
        let run1s = Date().timeIntervalSince(t0)
        #expect(result1.frames.dim(0) == 33)
        #expect(await pipeline.fusedLipDubLoRAPath == loraPath, "LoRA tracked after run 1")
        #expect(await pipeline.isAudioLoaded, "transformer survived run 1 (.disabled preset)")

        // Run 2 — same LoRA: must reuse the fused transformer. Before PR #36 this
        // re-fused (double delta = corrupted output) after an app-side reload.
        let t1 = Date()
        let result2 = try await pipeline.generateLipDub(
            prompt: prompt,
            referenceVideoPath: refVideo,
            lipdubLoraPath: loraPath,
            config: config,
            upscalerWeightsPath: upscalerPath
        )
        let run2s = Date().timeIntervalSince(t1)
        #expect(result2.frames.dim(0) == 33)
        #expect(await pipeline.fusedLipDubLoRAPath == loraPath, "state unchanged after reuse")
        print("[E2E] run1=\(String(format: "%.1f", run1s))s run2=\(String(format: "%.1f", run2s))s (run2 skips fusion)")

        // While fused, regular generation must refuse (silent corruption before).
        await #expect(throws: LTXError.self) {
            _ = try await pipeline.generateVideo(
                prompt: "a red ball", config: config, upscalerWeightsPath: upscalerPath
            )
        }
        await #expect(throws: LTXError.self) {
            var retakeConfig = config
            retakeConfig.videoPath = refVideo
            _ = try await pipeline.generateRetake(
                prompt: "a red ball", config: retakeConfig, upscalerWeightsPath: upscalerPath
            )
        }

        // Switching LoRA without a reload must throw (no pristine weights kept).
        // The file must exist (existence is checked before the fusion guard).
        let dummyLoRA = FileManager.default.temporaryDirectory
            .appendingPathComponent("dummy-other-lora-\(UUID().uuidString).safetensors")
        try Data().write(to: dummyLoRA)
        defer { try? FileManager.default.removeItem(at: dummyLoRA) }
        await #expect(throws: LTXError.self) {
            _ = try await pipeline.generateLipDub(
                prompt: prompt,
                referenceVideoPath: refVideo,
                lipdubLoraPath: dummyLoRA.path,
                config: config,
                upscalerWeightsPath: upscalerPath
            )
        }

        // Same LoRA at a different scale must also throw: the fusion keeps no
        // originals, so the delta cannot be rescaled in place.
        #expect(await pipeline.fusedLipDubLoRAScale == 1.0)
        await #expect(throws: LTXError.self) {
            _ = try await pipeline.generateLipDub(
                prompt: prompt,
                referenceVideoPath: refVideo,
                lipdubLoraPath: loraPath,
                lipdubLoRAScale: 0.8,
                config: config,
                upscalerWeightsPath: upscalerPath
            )
        }
    }

    /// Reuse without keeping the prompt encoder resident.
    ///
    /// `.disabled` (above) is the only preset that used to reach the reuse path,
    /// and it holds the encoder through the whole run — 7.5 GB on LTX-2.3, 26 GB
    /// on LTX-2.5. `.keepingTransformer()` splits the two: the encoder is freed
    /// after each text encode and rebuilt on its own at the next segment, while
    /// the fused transformer survives. Before the split gating, segment 2 failed
    /// on "Gemma model not loaded" (the unconditional-unload regression this
    /// suite caught once already).
    @Test func encoderReloadsAloneWhileTheFusionSurvives() async throws {
        let refVideo = Self.referenceVideoPath
        try #require(FileManager.default.fileExists(atPath: refVideo),
                     "Reference video missing: \(refVideo)")
        let loraPath = try await Self.resolveLoRAPath()

        let pipeline = LTXPipeline(
            model: .distilled, memoryOptimization: .light.keepingTransformer())
        try await pipeline.loadModels()
        try await pipeline.loadAudioModels(includeEncoder: true)
        let upscalerPath = try await pipeline.downloadUpscalerWeights()
        #expect(await pipeline.isGemmaLoaded, "encoder resident after load")

        let config = LTXVideoGenerationConfig(width: 384, height: 256, numFrames: 33)
        let prompt = "A person speaking in French saying: \"Bonjour à tous, ceci est un test.\""

        let t0 = Date()
        let result1 = try await pipeline.generateLipDub(
            prompt: prompt,
            referenceVideoPath: refVideo,
            lipdubLoraPath: loraPath,
            config: config,
            upscalerWeightsPath: upscalerPath
        )
        let run1s = Date().timeIntervalSince(t0)
        #expect(result1.frames.dim(0) == 33)
        #expect(await !pipeline.isGemmaLoaded, "encoder freed after the text encode")
        #expect(await pipeline.fusedLipDubLoRAPath == loraPath,
                "the fusion survived the encoder unload")
        #expect(await pipeline.isAudioLoaded, "transformer kept by keepingTransformer()")

        // Segment 2: reloads the encoder alone, reuses the fused transformer.
        let t1 = Date()
        let result2 = try await pipeline.generateLipDub(
            prompt: prompt,
            referenceVideoPath: refVideo,
            lipdubLoraPath: loraPath,
            config: config,
            upscalerWeightsPath: upscalerPath
        )
        let run2s = Date().timeIntervalSince(t1)
        #expect(result2.frames.dim(0) == 33)
        #expect(await pipeline.fusedLipDubLoRAPath == loraPath, "still fused, still the same file")
        #expect(await !pipeline.isGemmaLoaded, "encoder freed again")
        print("[E2E] keepingTransformer: run1=\(String(format: "%.1f", run1s))s "
              + "run2=\(String(format: "%.1f", run2s))s (run2 reloads the encoder only)")
    }
}
