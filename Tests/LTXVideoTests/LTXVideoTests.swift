//
//  LTXVideoTests.swift
//  ltx-video-swift-mlx
//

import Testing
@testable import LTXVideo

// MARK: - LTXVideo Module Tests

@Suite("LTXVideo Module")
struct LTXVideoModuleTests {
    @Test func testVersion() {
        // Deliberately not pinned to a literal. This test asserted "0.1.0"
        // through both the 0.2.0 and 0.3.0 bumps, so it never verified the
        // version — it only broke when someone finally changed it.
        //
        // What actually matters, that the shipped binary's version matches the
        // release tag, cannot be checked from here: it is enforced by
        // scripts/package-release.sh, which refuses to package on a mismatch.
        // All this can usefully hold is the shape.
        let components = LTXVideo.version.split(separator: ".")
        #expect(components.count == 3, "not semver: \(LTXVideo.version)")
        #expect(components.allSatisfy { Int($0) != nil }, "non-numeric: \(LTXVideo.version)")
    }

    @Test func testName() {
        #expect(LTXVideo.name == "LTX-Video-Swift-MLX")
    }
}

// MARK: - LTXError Tests

@Suite("LTXError")
struct LTXErrorTests {
    @Test func testErrorDescriptions() {
        let errors: [(LTXError, String)] = [
            (.modelNotLoaded("transformer"), "Model component not loaded: transformer"),
            (.invalidConfiguration("bad"), "Invalid configuration: bad"),
            (.insufficientMemory(required: 64, available: 32), "Insufficient memory: required 64GB, available 32GB"),
            (.weightLoadingFailed("corrupt"), "Failed to load weights: corrupt"),
            (.downloadFailed("timeout"), "Download failed: timeout"),
            (.videoProcessingFailed("decode"), "Video processing failed: decode"),
            (.generationFailed("OOM"), "Generation failed: OOM"),
            (.generationCancelled, "Generation was cancelled"),
            (.invalidFrameCount(10), "Invalid frame count: 10. Must be 8n + 1 (e.g., 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97)"),
            (.invalidDimensions(width: 100, height: 100), "Invalid dimensions: 100x100. Both must be divisible by 32"),
            (.textEncodingFailed("no model"), "Text encoding failed: no model"),
            (.fileNotFound("/tmp/nope"), "File not found: /tmp/nope"),
            (.invalidLoRA("wrong rank"), "Invalid LoRA: wrong rank"),
            (.exportFailed("codec"), "Export failed: codec"),
        ]

        for (error, expected) in errors {
            #expect(error.errorDescription == expected)
        }
    }
}

// MARK: - LTXDebug Tests

@Suite("LTXDebug")
struct LTXDebugTests {
    @Test func testDebugEnableDisable() {
        LTXDebug.disable()
        #expect(!LTXDebug.isEnabled)
        #expect(!LTXDebug.isVerbose)

        LTXDebug.enableDebugMode()
        #expect(LTXDebug.isEnabled)
        #expect(!LTXDebug.isVerbose)

        LTXDebug.enableVerboseMode()
        #expect(LTXDebug.isEnabled)
        #expect(LTXDebug.isVerbose)

        LTXDebug.disable()
        #expect(!LTXDebug.isEnabled)
        #expect(!LTXDebug.isVerbose)
    }
}
