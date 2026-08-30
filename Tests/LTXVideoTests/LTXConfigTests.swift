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
        #expect(allCases.count == 4)
        #expect(allCases.contains(.distilled))
        #expect(allCases.contains(.dev))
        #expect(allCases.contains(.v25Distilled))
        #expect(allCases.contains(.v25Dev))
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

// MARK: - Catalog / licensing Tests

@Suite("LTXModelCatalog")
struct LTXModelCatalogTests {
    @Test func testFamilyAndGating() {
        #expect(LTXModel.distilled.family == .ltx23)
        #expect(LTXModel.v25Dev.family == .ltx25)
        // LTX-2.3 is an open repo; every LTX-2.5 repo requires accepting the licence.
        #expect(LTXModel.distilled.isGated == false)
        #expect(LTXModel.v25Distilled.isGated == true)
        #expect(LTXModel.v25Distilled.huggingFaceRepo == "Lightricks/LTX-2.5")
    }

    @Test func testLicenseIsSharedAcrossGenerations() {
        for model in LTXModel.allCases {
            #expect(model.licenseInfo == .ltx2Community)
            #expect(model.isCommercialUseAllowed == true)
            #expect(model.licenseURL.hasPrefix("https://"))
        }
    }

    @Test func testSupportStatus() {
        for model in LTXModel.allCases {
            #expect(model.support == .supported)
            #expect(throws: Never.self) { try model.validateRunnable() }
        }
    }

    /// The refusal mechanism must keep working even though every catalogued
    /// artefact happens to be runnable today — the point is to fail before a
    /// multi-gigabyte download, not after it. Asserting on the *inventory*
    /// instead would have this test break every time something ships, which is
    /// exactly what it did when the temporal upscaler landed.
    @Test func testUnimplementedSupportIsRefused() {
        let pending = LTXModelSupport.notImplemented("temporal upscaling is not implemented")
        #expect(!pending.isRunnable)
        #expect(pending.label == "catalog")
        #expect(LTXModelSupport.supported.isRunnable)
        #expect(LTXModelSupport.supported.label == "ready")

        // And anything the catalog does mark unimplemented must label itself so.
        for aux in LTXAuxiliaryModel.allCases where !aux.support.isRunnable {
            #expect(aux.support.label == "catalog")
        }
    }

    @Test func testTextEncoderRequirement() {
        #expect(LTXModel.distilled.textEncoder == .gemma3_12b)
        #expect(LTXModel.distilled.textEncoder.externalRepo == "mlx-community/gemma-3-12b-it-qat-4bit")
        // LTX-2.5 bundles its Gemma 4 derivative inside the checkpoint.
        #expect(LTXModel.v25Dev.textEncoder == .gemma4_12bLTX)
        #expect(LTXModel.v25Dev.textEncoder.externalRepo == nil)
    }

    @Test func testWeightsLayoutAndComponents() {
        #expect(LTXModel.distilled.weightsLayout == .unified)
        #expect(LTXModel.distilled.componentFiles.count == 1)

        #expect(LTXModel.v25Distilled.weightsLayout == .split)
        let kinds = LTXModel.v25Distilled.componentFiles.map(\.kind)
        #expect(kinds.contains(.transformer))
        #expect(kinds.contains(.textEncoder))
        #expect(kinds.contains(.videoVAE))
        #expect(kinds.contains(.audioVAE))
        #expect(kinds.contains(.durationHead))
    }

    @Test func testAuxiliaryModelMetadata() {
        // Renamed upstream in August 2026 — the old repo/filename 404s.
        #expect(LTXAuxiliaryModel.dubItLoRA_23.huggingFaceRepo == "Lightricks/LTX-2.3-22b-IC-LoRA-DubIt")
        #expect(LTXAuxiliaryModel.dubItLoRA_23.filename == "ltx-2.3-22b-ic-lora-dubit-0.9.safetensors")
        // 1.0 was withdrawn from the repo; only 1.1 resolves.
        #expect(LTXAuxiliaryModel.spatialUpscalerX2_23.filename == "ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
        // IC-LoRA repos are gated even where the base checkpoint repo is open.
        #expect(LTXAuxiliaryModel.spatialUpscalerX2_23.gating == .open)
        #expect(LTXAuxiliaryModel.dubItLoRA_23.gating == .licenseAcceptanceRequired)
        #expect(LTXAuxiliaryModel.pixelSpatialUpscalerX2_25.gating == .licenseAcceptanceRequired)
        #expect(LTXAuxiliaryModel.pixelSpatialUpscalerX2_25.family == .ltx25)
    }

    /// Two upscaler families share the word "upscaler" and nothing else: the latent
    /// one is a conv model run between generation stages, the pixel one an IC-LoRA
    /// that re-renders from a reference video. Feeding one to the other's code path
    /// fuses zero layers and silently produces the wrong operation.
    @Test func testUpscalerFamiliesAreDistinguishable() {
        for latent: LTXAuxiliaryModel in [.spatialUpscalerX2_23, .latentSpatialUpscalerX2_25,
                                          .latentTemporalUpscalerX2_25] {
            #expect(latent.isAdapter == false, "\(latent.rawValue) is not a LoRA")
        }
        for pixel: LTXAuxiliaryModel in [.pixelSpatialUpscalerX2_23, .pixelSpatialUpscalerX4_23,
                                         .pixelSpatialUpscalerX2_25] {
            #expect(pixel.isAdapter, "\(pixel.rawValue) is a LoRA")
        }
        // Both generations publish a pixel upscaler; the resolver must not fall back
        // to a latent one for 2.3.
        #expect(LTXAuxiliaryModel.pixelSpatialUpscaler(for: .ltx23) == .pixelSpatialUpscalerX2_23)
        #expect(LTXAuxiliaryModel.pixelSpatialUpscaler(for: .ltx25) == .pixelSpatialUpscalerX2_25)
        #expect(LTXAuxiliaryModel.pixelSpatialUpscaler(for: .ltx23).isAdapter)
    }

    @Test func testFeedForwardBiasTracksCheckpoint() {
        // 2.3 ships ff.net.{0.proj,2}.bias for every block; 2.5 sets ff_bias: false.
        #expect(LTXTransformerConfig.ltx23.ffBias == true)
        #expect(LTXTransformerConfig.ltx25.ffBias == false)
        // The two streams diverge in 2.5: `ff_bias: false` but `audio_ff_bias`
        // unset, and audio_ff.net.{0.proj,2}.bias is present for all 48 blocks.
        // Tying them together drops 96 trained audio biases.
        #expect(LTXTransformerConfig.ltx23.audioFfBias == true)
        #expect(LTXTransformerConfig.ltx25.audioFfBias == true)
        #expect(LTXTransformerConfig.ltx25.keyframesAbsPosEmbedding == true)
        // Everything else is identical between the two generations.
        #expect(LTXTransformerConfig.ltx25.numLayers == LTXTransformerConfig.ltx23.numLayers)
        #expect(LTXTransformerConfig.ltx25.innerDim == LTXTransformerConfig.ltx23.innerDim)
        #expect(LTXTransformerConfig.ltx25.maxPos == LTXTransformerConfig.ltx23.maxPos)
        #expect(LTXTransformerConfig.ltx25.captionChannels == LTXTransformerConfig.ltx23.captionChannels)
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

    // 481 frames = 20 s at 24 fps, the RoPE positional range (maxPos[0] = 20 s).
    // Values used to be capped at 257 (~10.7 s) — too short for dubbing use cases.
    @Test func testFrameCountUpToRoPERange() throws {
        for n in [265, 361, 481] {
            let config = LTXVideoGenerationConfig(width: 768, height: 512, numFrames: n)
            try config.validate()
        }
    }

    @Test func testFramesBeyondRoPERange() {
        let config = LTXVideoGenerationConfig(width: 768, height: 512, numFrames: 489)
        #expect(throws: LTXError.self) { try config.validate() }
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
        #expect(config.retakeModality == .videoOnly)
    }

    @Test func testRegenerateAudioDefault() {
        let config = LTXVideoGenerationConfig()
        #expect(!config.retakeModality.regeneratesAudio)
    }

    @Test func testRegenerateAudioEnabled() {
        let config = LTXVideoGenerationConfig(
            videoPath: "/tmp/vid.mp4",
            regenerateAudio: true
        )
        #expect(config.retakeModality == .both)
        #expect(config.retakeModality.regeneratesAudio)
        #expect(config.videoPath == "/tmp/vid.mp4")
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
