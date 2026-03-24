//
//  LTXConfigTests.swift
//  ltx-video-swift-mlx
//

import Testing
import Foundation
@testable import LTXVideo

// MARK: - LTXModel Tests

@Suite("LTXModel")
struct LTXModelTests {
    @Test func testDistilledProperties() {
        let model = LTXModel.distilled
        #expect(model.rawValue == "distilled")
        #expect(model.displayName == "LTX-2.3 Distilled (~46GB)")
        #expect(model.defaultSteps == 8)
        #expect(model.estimatedVRAM == 46)
        #expect(model.huggingFaceRepo == "Lightricks/LTX-2.3")
        #expect(model.unifiedWeightsFilename == "ltx-2.3-22b-distilled.safetensors")
    }

    @Test func testTransformerConfig() {
        let config = LTXModel.distilled.transformerConfig
        #expect(config.numLayers == 48)
        #expect(config.numAttentionHeads == 32)
        #expect(config.attentionHeadDim == 128)
        #expect(config.innerDim == 4096)
        #expect(config.gatedAttention == true)
        #expect(config.crossAttentionAdaLN == true)
        #expect(config.captionProjBeforeConnector == true)
    }

    @Test func testCaseIterable() {
        let allCases = LTXModel.allCases
        #expect(allCases.count == 2)
        #expect(allCases.contains(.distilled))
        #expect(allCases.contains(.dev))
    }

    @Test func testDevProperties() {
        let model = LTXModel.dev
        #expect(model.rawValue == "dev")
        #expect(model.displayName == "LTX-2.3 Dev (~46GB)")
        #expect(model.defaultSteps == 30)
        #expect(model.estimatedVRAM == 46)
        #expect(model.huggingFaceRepo == "Lightricks/LTX-2.3")
        #expect(model.unifiedWeightsFilename == "ltx-2.3-22b-dev.safetensors")
        #expect(model.transformerConfig.gatedAttention == true)
    }
}

// MARK: - LTXTransformerConfig Tests

@Suite("LTXTransformerConfig")
struct LTXTransformerConfigTests {
    @Test func testDefaultConfig() {
        let config = LTXTransformerConfig.default
        #expect(config.numLayers == 48)
        #expect(config.gatedAttention == false)
        #expect(config.crossAttentionAdaLN == false)
        #expect(config.captionProjBeforeConnector == false)
        #expect(config.captionChannels == 3840)
    }

    @Test func testLTX23Config() {
        let config = LTXTransformerConfig.ltx23
        #expect(config.numLayers == 48)
        #expect(config.gatedAttention == true)
        #expect(config.crossAttentionAdaLN == true)
        #expect(config.captionProjBeforeConnector == true)
        #expect(config.captionChannels == 4096)
    }

    @Test func testInnerDim() {
        let config = LTXTransformerConfig(numAttentionHeads: 16, attentionHeadDim: 64)
        #expect(config.innerDim == 1024)
    }

    @Test func testAudioDimensions() {
        let config = LTXTransformerConfig.ltx23
        #expect(config.audioNumAttentionHeads == 32)
        #expect(config.audioAttentionHeadDim == 64)
        #expect(config.audioInnerDim == 2048)
        #expect(config.audioCrossAttentionDim == 2048)
    }

    @Test func testDescription() {
        let config = LTXTransformerConfig.ltx23
        let desc = config.description
        #expect(desc.contains("layers: 48"))
        #expect(desc.contains("heads: 32"))
    }

    @Test func testRoPEConfig() {
        let config = LTXTransformerConfig.ltx23
        #expect(config.ropeTheta == 10000.0)
        #expect(config.maxPos == [20, 2048, 2048])
        #expect(config.timestepScaleMultiplier == 1000)
    }
}

// MARK: - LTXVideoGenerationConfig Tests

@Suite("LTXVideoGenerationConfig")
struct LTXVideoGenerationConfigTests {
    @Test func testDefaultValues() {
        let config = LTXVideoGenerationConfig()
        #expect(config.width == 704)
        #expect(config.height == 480)
        #expect(config.numFrames == 121)
        #expect(config.numSteps == 8)
        #expect(config.seed == nil)
        #expect(config.enhancePrompt == false)
        #expect(config.imagePath == nil)
        #expect(config.imageCondNoiseScale == 0.0)
        #expect(config.videoPath == nil)
        #expect(config.retakeStrength == 0.8)
        #expect(config.retakeStartTime == nil)
        #expect(config.retakeEndTime == nil)
    }

    @Test func testModelConvenienceInit() {
        let config = LTXVideoGenerationConfig(model: .distilled)
        #expect(config.numSteps == 8)
    }

    @Test func testModelConvenienceInitOverride() {
        let config = LTXVideoGenerationConfig(model: .distilled, numSteps: 4)
        #expect(config.numSteps == 4)
    }

    @Test func testLatentDimensions() {
        let config = LTXVideoGenerationConfig(width: 768, height: 512, numFrames: 121)
        #expect(config.latentWidth == 24)   // 768 / 32
        #expect(config.latentHeight == 16)  // 512 / 32
        #expect(config.latentFrames == 16)  // (121 - 1) / 8 + 1
        #expect(config.numLatentTokens == 24 * 16 * 16)
    }

    @Test func testLatentDimensions1024x576() {
        let config = LTXVideoGenerationConfig(width: 1024, height: 576, numFrames: 241)
        #expect(config.latentWidth == 32)   // 1024 / 32
        #expect(config.latentHeight == 18)  // 576 / 32
        #expect(config.latentFrames == 31)  // (241 - 1) / 8 + 1
    }

    // MARK: - Validation Tests

    @Test func testValidConfig() throws {
        let config = LTXVideoGenerationConfig(width: 768, height: 512, numFrames: 121)
        try config.validate()
    }

    @Test func testInvalidWidthNotDivisibleBy64() {
        let config = LTXVideoGenerationConfig(width: 700, height: 512)
        #expect(throws: LTXError.self) { try config.validate() }
    }

    @Test func testInvalidHeightNotDivisibleBy64() {
        let config = LTXVideoGenerationConfig(width: 768, height: 500)
        #expect(throws: LTXError.self) { try config.validate() }
    }

    @Test func testInvalidFrameCount() {
        let config = LTXVideoGenerationConfig(numFrames: 10)  // not 8n+1
        #expect(throws: LTXError.self) { try config.validate() }
    }

    @Test func testValidFrameCounts() throws {
        for n in [9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121, 241] {
            let config = LTXVideoGenerationConfig(width: 768, height: 512, numFrames: n)
            try config.validate()
        }
    }

    @Test func testWidthTooSmall() {
        let config = LTXVideoGenerationConfig(width: 32, height: 512)
        #expect(throws: LTXError.self) { try config.validate() }
    }

    @Test func testWidthTooLarge() {
        let config = LTXVideoGenerationConfig(width: 4096, height: 512)
        #expect(throws: LTXError.self) { try config.validate() }
    }

    @Test func testFramesTooFew() {
        let config = LTXVideoGenerationConfig(width: 768, height: 512, numFrames: 1)
        #expect(throws: LTXError.self) { try config.validate() }
    }

    @Test func testStepsTooFew() {
        let config = LTXVideoGenerationConfig(width: 768, height: 512, numSteps: 0)
        #expect(throws: LTXError.self) { try config.validate() }
    }

    @Test func testStepsTooMany() {
        let config = LTXVideoGenerationConfig(width: 768, height: 512, numSteps: 200)
        #expect(throws: LTXError.self) { try config.validate() }
    }

    @Test func testMissingImagePath() {
        let config = LTXVideoGenerationConfig(
            width: 768, height: 512, imagePath: "/nonexistent/image.png"
        )
        #expect(throws: LTXError.self) { try config.validate() }
    }

    @Test func testMissingVideoPath() {
        let config = LTXVideoGenerationConfig(
            width: 768, height: 512, videoPath: "/nonexistent/video.mp4"
        )
        #expect(throws: LTXError.self) { try config.validate() }
    }

    @Test func testInvalidRetakeStrengthZero() {
        // Create a temporary file for valid videoPath
        let tmpPath = "/tmp/ltx_test_video.mp4"
        FileManager.default.createFile(atPath: tmpPath, contents: nil)
        defer { try? FileManager.default.removeItem(atPath: tmpPath) }

        let config = LTXVideoGenerationConfig(
            width: 768, height: 512,
            videoPath: tmpPath,
            retakeStrength: 0.0
        )
        #expect(throws: LTXError.self) { try config.validate() }
    }

    @Test func testRetakeStrengthBoundaries() throws {
        let tmpPath = "/tmp/ltx_test_video2.mp4"
        FileManager.default.createFile(atPath: tmpPath, contents: nil)
        defer { try? FileManager.default.removeItem(atPath: tmpPath) }

        // strength = 1.0 should be valid
        let config1 = LTXVideoGenerationConfig(
            width: 768, height: 512,
            videoPath: tmpPath,
            retakeStrength: 1.0
        )
        try config1.validate()

        // strength = 0.5 should be valid
        let config2 = LTXVideoGenerationConfig(
            width: 768, height: 512,
            videoPath: tmpPath,
            retakeStrength: 0.5
        )
        try config2.validate()
    }

    @Test func testRetakeFieldsSet() {
        let config = LTXVideoGenerationConfig(
            videoPath: "/tmp/vid.mp4",
            retakeStrength: 0.7,
            retakeStartTime: 2.0,
            retakeEndTime: 5.0
        )
        #expect(config.videoPath == "/tmp/vid.mp4")
        #expect(config.retakeStrength == 0.7)
        #expect(config.retakeStartTime == 2.0)
        #expect(config.retakeEndTime == 5.0)
    }

    // MARK: - Retake Temporal Mask Tests

    @Test func testRetakePartialTimeRange() {
        // Partial retake: only start_time set → regenerate from start_time to end
        let config = LTXVideoGenerationConfig(
            numFrames: 121,  // 5s at 24fps
            videoPath: "/tmp/vid.mp4",
            retakeStartTime: 2.5
        )
        #expect(config.retakeStartTime == 2.5)
        #expect(config.retakeEndTime == nil)  // nil = end of video
    }

    @Test func testRetakeLatentFrameMapping() {
        // Verify the latent frame formula: latent_frames = (pixel_frames - 1) / 8 + 1
        let latent121 = (121 - 1) / 8 + 1
        let latent233 = (233 - 1) / 8 + 1
        let latent65 = (65 - 1) / 8 + 1
        #expect(latent121 == 16)
        #expect(latent233 == 30)
        #expect(latent65 == 9)
    }

    @Test func testRetakeMinimumDurationForVisibleChanges() {
        // Each latent frame covers ~0.33s at 24fps (8 pixel frames / 24fps)
        // Minimum recommended: 5 seconds (121 frames = 16 latent frames)
        let shortDuration = Float(9) / 24.0   // 0.37s
        let longDuration = Float(121) / 24.0  // 5.0s
        #expect(shortDuration < 1.0)
        #expect(longDuration >= 5.0)
    }

    @Test func testRetakeTemporalGranularity() {
        // start_time=0.3 → startPixel=7 → latentFrame = 7/8 = 0 (ALL frames regen)
        // start_time=0.4 → startPixel=9 → latentFrame = 9/8 = 1 (frame 0 kept)
        let startPixel1 = Int(0.3 * 24.0)
        let startPixel2 = Int(0.4 * 24.0)
        let latentFrame1 = startPixel1 / 8
        let latentFrame2 = startPixel2 / 8
        #expect(latentFrame1 == 0)
        #expect(latentFrame2 == 1)
    }
}
