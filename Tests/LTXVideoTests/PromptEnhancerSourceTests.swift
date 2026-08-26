// PromptEnhancerSourceTests.swift — enhancer source resolution, offline
// Copyright 2026
//
// The point of PromptEnhancerSource is to stop a second multi-gigabyte Gemma
// from landing on disk, so what matters here is that `.localRoot` never
// downloads and that it refuses a root the loader would only reject later,
// with a worse message.

import Foundation
import Testing
@testable import LTXVideo

@Suite("Prompt enhancer source")
struct PromptEnhancerSourceTests {

    /// A scratch directory removed when the test ends.
    static func makeTempDir() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("ltx-enhancer-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    @Test func defaultIsTheReferencePrecision() {
        // bf16 is what the reference space runs; a quantized default would be a
        // silent quality change for every existing caller.
        #expect(PromptEnhancerSource.default == .managed(.bf16))
    }

    @Test func precisionsNameDistinctRepositories() {
        #expect(PromptEnhancerPrecision.bf16.repoID == "mlx-community/gemma-4-e2b-it-bf16")
        #expect(PromptEnhancerPrecision.sixBit.repoID == "mlx-community/gemma-4-e2b-it-6bit")
        #expect(PromptEnhancerPrecision.sixBit.rawValue == "6bit")
        // The whole reason 6-bit is offered: it is materially smaller.
        #expect(PromptEnhancerPrecision.sixBit.approximateSizeGB
                < PromptEnhancerPrecision.bf16.approximateSizeGB)
    }

    @Test func cacheDirectoriesDoNotCollideAcrossPrecisions() async throws {
        let root = try Self.makeTempDir()
        defer { try? FileManager.default.removeItem(at: root) }
        let downloader = ModelDownloader(cacheDir: root)

        let bf16 = await downloader.gemma4EnhancerCacheDir(.bf16)
        let sixBit = await downloader.gemma4EnhancerCacheDir(.sixBit)
        #expect(bf16 != sixBit)
        // Installs made before precisions existed wrote this exact name; changing
        // it would re-download 10 GB for everyone already holding the weights.
        #expect(bf16.lastPathComponent == "enhancer-gemma4-e2b-bf16")
    }

    @Test func localRootResolvesWithoutNetwork() async throws {
        let root = try Self.makeTempDir()
        defer { try? FileManager.default.removeItem(at: root) }
        let checkpoint = root.appendingPathComponent("my-gemma")
        try FileManager.default.createDirectory(at: checkpoint, withIntermediateDirectories: true)
        try #"{"model_type":"gemma4"}"#.write(
            to: checkpoint.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)

        let downloader = ModelDownloader(cacheDir: root)
        let resolved = try await downloader.resolveGemma4Enhancer(
            source: .localRoot(checkpoint.path))

        #expect(resolved.standardizedFileURL == checkpoint.standardizedFileURL)
        // Nothing may have been fetched into the managed location.
        let managed = await downloader.gemma4EnhancerCacheDir(.bf16)
        #expect(!FileManager.default.fileExists(atPath: managed.path))
    }

    @Test func localRootRejectsAMissingDirectory() async throws {
        let root = try Self.makeTempDir()
        defer { try? FileManager.default.removeItem(at: root) }
        let downloader = ModelDownloader(cacheDir: root)

        await #expect(throws: LTXError.self) {
            _ = try await downloader.resolveGemma4Enhancer(
                source: .localRoot(root.appendingPathComponent("nope").path))
        }
    }

    @Test func localRootRejectsADirectoryThatIsNotACheckpoint() async throws {
        let root = try Self.makeTempDir()
        defer { try? FileManager.default.removeItem(at: root) }
        let empty = root.appendingPathComponent("empty")
        try FileManager.default.createDirectory(at: empty, withIntermediateDirectories: true)
        let downloader = ModelDownloader(cacheDir: root)

        // Caught here rather than surfacing later as a model-parsing failure.
        await #expect(throws: LTXError.self) {
            _ = try await downloader.resolveGemma4Enhancer(source: .localRoot(empty.path))
        }
    }

    @Test func localRootRejectsAFile() async throws {
        let root = try Self.makeTempDir()
        defer { try? FileManager.default.removeItem(at: root) }
        let file = root.appendingPathComponent("weights.safetensors")
        try Data([0x00]).write(to: file)
        let downloader = ModelDownloader(cacheDir: root)

        await #expect(throws: LTXError.self) {
            _ = try await downloader.resolveGemma4Enhancer(source: .localRoot(file.path))
        }
    }
}
