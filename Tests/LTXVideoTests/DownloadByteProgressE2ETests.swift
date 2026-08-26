// DownloadByteProgressE2ETests.swift — byte progress against a real transfer
// Copyright 2026
//
// The unit tests pin the weighting arithmetic; only a real transfer proves the
// URLSession delegate is actually wired and firing. Uses a 33 MB ungated file
// so the test stays seconds, not minutes.
//
// Run:
//   LTX_NETWORK_TESTS=1 swift test --filter DownloadByteProgressE2ETests

import Foundation
import Testing
@testable import LTXVideo

@Suite("Download byte progress (network: LTX_NETWORK_TESTS)",
       .enabled(if: ProcessInfo.processInfo.environment["LTX_NETWORK_TESTS"] != nil))
struct DownloadByteProgressE2ETests {

    /// Ungated, stable, and big enough to span many chunks.
    static let repoID = "mlx-community/gemma-4-e2b-it-6bit"
    static let file = "tokenizer.json"
    static let approximateGB = 0.031

    @Test func aRealTransferReportsClimbingByteCounts() async throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ltx-dl-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let destination = dir.appendingPathComponent(Self.file)
        let downloader = ModelDownloader(cacheDir: dir)

        // The callback arrives on URLSession's delegate queue, so collect under
        // a lock rather than into a bare array.
        let collected = Collector()
        try await downloader.runDownloadPlan(
            WeightedDownloadPlan(items: [.init(
                label: Self.file, repoPath: Self.file,
                sizeGB: Self.approximateGB, destination: destination)]),
            repoId: Self.repoID,
            progress: { collected.append($0) })

        let samples = collected.samples
        #expect(FileManager.default.fileExists(atPath: destination.path))

        // More than the single "starting" event means the delegate fired.
        #expect(samples.count > 1, "only \(samples.count) progress events — delegate not wired")

        let withBytes = samples.filter { $0.bytesDownloaded > 0 }
        #expect(!withBytes.isEmpty, "no event carried a byte count")
        #expect(withBytes.allSatisfy { $0.totalBytes > 0 }, "byte events carried no total")

        // Byte counts and the aggregate both only climb.
        for (a, b) in zip(withBytes, withBytes.dropFirst()) {
            #expect(b.bytesDownloaded >= a.bytesDownloaded)
            #expect(b.progress >= a.progress)
        }
        #expect(samples.allSatisfy { $0.progress >= 0 && $0.progress <= 1.0 })
    }

    @Test func aFailedRequestLeavesNothingCached() async throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ltx-dl-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let destination = dir.appendingPathComponent("nope.safetensors")
        let downloader = ModelDownloader(cacheDir: dir)

        // The coordinator moves the body into place before anyone inspects the
        // status, so a 404 page could otherwise be cached as if it were weights
        // — and every later run would "find" it and skip the download.
        await #expect(throws: LTXError.self) {
            try await downloader.runDownloadPlan(
                WeightedDownloadPlan(items: [.init(
                    label: "nope", repoPath: "definitely-not-a-file.safetensors",
                    sizeGB: 0.001, destination: destination)]),
                repoId: Self.repoID,
                progress: nil)
        }
        #expect(!FileManager.default.fileExists(atPath: destination.path))
    }

    /// Lock-guarded sink for callbacks arriving off the test's thread.
    final class Collector: @unchecked Sendable {
        private let lock = NSLock()
        private var storage: [DownloadProgress] = []

        func append(_ p: DownloadProgress) {
            lock.lock(); defer { lock.unlock() }
            storage.append(p)
        }

        var samples: [DownloadProgress] {
            lock.lock(); defer { lock.unlock() }
            return storage
        }
    }
}
