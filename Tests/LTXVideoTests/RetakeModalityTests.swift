//
//  RetakeModalityTests.swift
//  ltx-video-swift-mlx
//
//  Per-modality retake: which stream is regenerated, and the legacy
//  `regenerateAudio` spelling it replaces.
//

import Testing
import Foundation
@testable import LTXVideo

@Suite("RetakeModality")
struct RetakeModalityTests {

    // MARK: - The enum itself

    @Test func testStreamsPerModality() {
        #expect(RetakeModality.videoOnly.regeneratesVideo)
        #expect(!RetakeModality.videoOnly.regeneratesAudio)

        #expect(RetakeModality.both.regeneratesVideo)
        #expect(RetakeModality.both.regeneratesAudio)

        #expect(!RetakeModality.audioOnly.regeneratesVideo)
        #expect(RetakeModality.audioOnly.regeneratesAudio)
    }

    @Test func testEveryModalityRegeneratesSomething() {
        // A modality that freezes both streams would denoise nothing and return
        // its input — there is no such case, and there should not be one.
        for modality in RetakeModality.allCases {
            #expect(modality.regeneratesVideo || modality.regeneratesAudio,
                    "\(modality) regenerates neither stream")
        }
    }

    // MARK: - Legacy spelling

    @Test func testDefaultIsVideoOnly() {
        let config = LTXVideoGenerationConfig(videoPath: "/tmp/vid.mp4")
        #expect(config.retakeModality == .videoOnly)
    }

    @Test func testRegenerateAudioParameterMapsToBoth() {
        let config = LTXVideoGenerationConfig(
            videoPath: "/tmp/vid.mp4", regenerateAudio: true)
        #expect(config.retakeModality == .both)
    }

    @Test func testExplicitModalityWinsOverLegacyParameter() {
        // A caller passing both is expressing the newer intent; .audioOnly cannot
        // be spelled with the boolean at all.
        let config = LTXVideoGenerationConfig(
            videoPath: "/tmp/vid.mp4",
            regenerateAudio: true,
            retakeModality: .audioOnly)
        #expect(config.retakeModality == .audioOnly)
    }

    @Test func testModelInitializerCarriesModality() {
        let config = LTXVideoGenerationConfig(
            model: .dev, videoPath: "/tmp/vid.mp4", retakeModality: .audioOnly)
        #expect(config.retakeModality == .audioOnly)
        #expect(config.numSteps == LTXModel.dev.defaultSteps)
    }

    // Deprecated itself, so reading the deprecated property below raises no
    // warning here — and none anywhere else, since this is its only use.
    @available(*, deprecated)
    @Test func testDeprecatedPropertyIsAViewOfTheModality() {
        var config = LTXVideoGenerationConfig(videoPath: "/tmp/vid.mp4")

        config.retakeModality = .both
        #expect(config.regenerateAudio)

        config.retakeModality = .audioOnly
        #expect(config.regenerateAudio,
                "audio-only regenerates audio, whatever the boolean can express")

        config.retakeModality = .videoOnly
        #expect(!config.regenerateAudio)

        // Writing it still selects one of the two modalities it can express.
        config.regenerateAudio = true
        #expect(config.retakeModality == .both)
        config.regenerateAudio = false
        #expect(config.retakeModality == .videoOnly)
    }

    // MARK: - Validation

    @Test func testAudioOnlyRejectsAPartialWindow() {
        let tmpPath = NSTemporaryDirectory() + "ltx_test_modality_window.mp4"
        FileManager.default.createFile(atPath: tmpPath, contents: nil)
        defer { try? FileManager.default.removeItem(atPath: tmpPath) }

        // The window masks video latent frames; .audioOnly regenerates none, so
        // accepting it would advertise an audio window that does not exist.
        let config = LTXVideoGenerationConfig(
            width: 768, height: 512,
            videoPath: tmpPath,
            retakeStartTime: 1.0,
            retakeEndTime: 3.0,
            retakeModality: .audioOnly)
        #expect(throws: LTXError.self) { try config.validate() }
    }

    @Test func testAudioOnlyWithoutAWindowValidates() throws {
        let tmpPath = NSTemporaryDirectory() + "ltx_test_modality_ok.mp4"
        FileManager.default.createFile(atPath: tmpPath, contents: nil)
        defer { try? FileManager.default.removeItem(atPath: tmpPath) }

        let config = LTXVideoGenerationConfig(
            width: 768, height: 512, videoPath: tmpPath, retakeModality: .audioOnly)
        try config.validate()
    }

    @Test func testWindowStaysValidForTheOtherModalities() throws {
        let tmpPath = NSTemporaryDirectory() + "ltx_test_modality_both.mp4"
        FileManager.default.createFile(atPath: tmpPath, contents: nil)
        defer { try? FileManager.default.removeItem(atPath: tmpPath) }

        for modality in [RetakeModality.videoOnly, .both] {
            let config = LTXVideoGenerationConfig(
                width: 768, height: 512,
                videoPath: tmpPath,
                retakeStartTime: 1.0,
                retakeEndTime: 3.0,
                retakeModality: modality)
            try config.validate()
        }
    }
}
