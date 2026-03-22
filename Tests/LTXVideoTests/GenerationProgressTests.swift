//
//  GenerationProgressTests.swift
//  ltx-video-swift-mlx
//

import Testing
import MLX
@testable import LTXVideo

// MARK: - GenerationProgress Tests

@Suite("GenerationProgress")
struct GenerationProgressTests {
    @Test func testProgressFraction() {
        let p = GenerationProgress(currentStep: 0, totalSteps: 8, sigma: 1.0, phase: .denoising)
        #expect(p.progress == 1.0 / 8.0)

        let p2 = GenerationProgress(currentStep: 7, totalSteps: 8, sigma: 0.0, phase: .denoising)
        #expect(p2.progress == 1.0)
    }

    @Test func testStatusString() {
        let p = GenerationProgress(currentStep: 2, totalSteps: 8, sigma: 0.725, phase: .denoising)
        #expect(p.status == "Step 3/8 [denoising] (σ=0.7250)")
    }

    @Test func testStatusFirstStep() {
        let p = GenerationProgress(currentStep: 0, totalSteps: 8, sigma: 1.0, phase: .denoising)
        #expect(p.status == "Step 1/8 [denoising] (σ=1.0000)")
    }

    @Test func testStatusLastStep() {
        let p = GenerationProgress(currentStep: 7, totalSteps: 8, sigma: 0.42, phase: .refinement)
        #expect(p.status == "Step 8/8 [refinement] (σ=0.4200)")
    }

    @Test func testPhaseStatus() {
        let p = GenerationProgress(currentStep: 8, totalSteps: 8, sigma: 0, phase: .upscaling)
        #expect(p.status == "[upscaling]")

        let p2 = GenerationProgress(currentStep: 8, totalSteps: 8, sigma: 0, phase: .decoding)
        #expect(p2.status == "[decoding]")

        let p3 = GenerationProgress(currentStep: 8, totalSteps: 8, sigma: 0, phase: .exporting)
        #expect(p3.status == "[exporting]")
    }
}

// MARK: - GenerationTimings Tests

@Suite("GenerationTimings")
struct GenerationTimingsTests {
    @Test func testEmptyTimings() {
        let t = GenerationTimings()
        #expect(t.textEncoding == 0)
        #expect(t.vaeDecode == 0)
        #expect(t.totalDenoise == 0)
        #expect(t.avgStepTime == 0)
        #expect(t.denoiseSteps.isEmpty)
        #expect(t.peakMemoryMB == 0)
        #expect(t.meanMemoryMB == 0)
    }

    @Test func testDenoiseStepTimings() {
        var t = GenerationTimings()
        t.denoiseSteps = [1.0, 2.0, 3.0]
        #expect(t.totalDenoise == 6.0)
        #expect(t.avgStepTime == 2.0)
    }

    @Test func testSingleStep() {
        var t = GenerationTimings()
        t.denoiseSteps = [5.0]
        #expect(t.totalDenoise == 5.0)
        #expect(t.avgStepTime == 5.0)
    }
}

// MARK: - VideoGenerationResult Tests

@Suite("VideoGenerationResult")
struct VideoGenerationResultTests {
    @Test func testBasicResult() {
        let frames = MLXArray.zeros([10, 512, 768, 3])
        let result = VideoGenerationResult(
            frames: frames,
            seed: 42,
            generationTime: 100.0
        )
        #expect(result.numFrames == 10)
        #expect(result.height == 512)
        #expect(result.width == 768)
        #expect(result.seed == 42)
        #expect(result.generationTime == 100.0)
        #expect(result.timings == nil)
        #expect(result.audioWaveform == nil)
        #expect(result.audioSampleRate == nil)
    }

    @Test func testResultWithAudio() {
        let frames = MLXArray.zeros([10, 512, 768, 3])
        let audio = MLXArray.zeros([24000])
        let result = VideoGenerationResult(
            frames: frames,
            seed: 42,
            generationTime: 100.0,
            audioWaveform: audio,
            audioSampleRate: 24000
        )
        #expect(result.audioWaveform != nil)
        #expect(result.audioSampleRate == 24000)
    }
}

// MARK: - LTXModelRegistry Tests

@Suite("LTXModelRegistry")
struct LTXModelRegistryTests {
    @Test func testRecommendedModel() {
        #expect(LTXModelRegistry.recommendedModel == .distilled)
    }

    @Test func testSystemRAM() {
        // Just verify it returns a positive value
        #expect(LTXModelRegistry.systemRAMGB > 0)
    }
}
