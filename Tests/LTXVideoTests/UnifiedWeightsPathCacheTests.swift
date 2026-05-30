//
//  UnifiedWeightsPathCacheTests.swift
//  ltx-video-swift-mlx
//

import Testing
import Foundation
@testable import LTXVideo

@Suite("UnifiedWeightsPathCache")
struct UnifiedWeightsPathCacheTests {

    @Test func cacheMissReturnsNil() {
        var cache = UnifiedWeightsPathCache(fileExists: { _ in true })
        #expect(cache.cachedPath(for: .distilled) == nil)
        #expect(cache.cachedPath(for: .dev) == nil)
    }

    @Test func storeThenHitReturnsPathWhenFileExists() {
        var cache = UnifiedWeightsPathCache(fileExists: { _ in true })
        cache.store("/tmp/ltx-distilled.safetensors", for: .distilled)
        #expect(cache.cachedPath(for: .distilled) == "/tmp/ltx-distilled.safetensors")
    }

    @Test func cacheEvictsAndReturnsNilWhenFileVanishes() {
        var present = true
        var cache = UnifiedWeightsPathCache(fileExists: { _ in present })
        cache.store("/tmp/ltx-distilled.safetensors", for: .distilled)
        #expect(cache.cachedPath(for: .distilled) == "/tmp/ltx-distilled.safetensors")

        present = false
        #expect(cache.cachedPath(for: .distilled) == nil)

        // The eviction is sticky: even if the file reappears under the same
        // path, the entry has been cleared, so the next read still misses.
        present = true
        #expect(cache.cachedPath(for: .distilled) == nil)
    }

    @Test func storeOverwritesPreviousEntryForSameModel() {
        var cache = UnifiedWeightsPathCache(fileExists: { _ in true })
        cache.store("/tmp/first.safetensors", for: .distilled)
        cache.store("/tmp/second.safetensors", for: .distilled)
        #expect(cache.cachedPath(for: .distilled) == "/tmp/second.safetensors")
    }

    @Test func entriesAreIsolatedPerModel() {
        var cache = UnifiedWeightsPathCache(fileExists: { _ in true })
        cache.store("/tmp/distilled.safetensors", for: .distilled)
        cache.store("/tmp/dev.safetensors", for: .dev)
        #expect(cache.cachedPath(for: .distilled) == "/tmp/distilled.safetensors")
        #expect(cache.cachedPath(for: .dev) == "/tmp/dev.safetensors")
    }

    @Test func evictionOfOneModelLeavesOthersIntact() {
        var existingPaths: Set<String> = ["/tmp/dev.safetensors"]
        var cache = UnifiedWeightsPathCache(fileExists: { existingPaths.contains($0) })
        cache.store("/tmp/distilled.safetensors", for: .distilled)
        cache.store("/tmp/dev.safetensors", for: .dev)

        // Only the distilled file disappears.
        #expect(cache.cachedPath(for: .distilled) == nil)
        #expect(cache.cachedPath(for: .dev) == "/tmp/dev.safetensors")

        // And bringing it back via store() works as expected.
        existingPaths.insert("/tmp/distilled-v2.safetensors")
        cache.store("/tmp/distilled-v2.safetensors", for: .distilled)
        #expect(cache.cachedPath(for: .distilled) == "/tmp/distilled-v2.safetensors")
    }

    @Test func clearEmptiesAllEntries() {
        var cache = UnifiedWeightsPathCache(fileExists: { _ in true })
        cache.store("/tmp/distilled.safetensors", for: .distilled)
        cache.store("/tmp/dev.safetensors", for: .dev)

        cache.clear()

        #expect(cache.cachedPath(for: .distilled) == nil)
        #expect(cache.cachedPath(for: .dev) == nil)
    }

    @Test func defaultFileExistsProbeUsesRealFilesystem() throws {
        let tempDir = FileManager.default.temporaryDirectory
        let realFile = tempDir.appendingPathComponent("ltx-cache-real-\(UUID().uuidString).safetensors")
        let fakeFile = tempDir.appendingPathComponent("ltx-cache-missing-\(UUID().uuidString).safetensors")

        try Data([0x42]).write(to: realFile)
        defer { try? FileManager.default.removeItem(at: realFile) }

        var cache = UnifiedWeightsPathCache()
        cache.store(realFile.path, for: .distilled)
        cache.store(fakeFile.path, for: .dev)

        #expect(cache.cachedPath(for: .distilled) == realFile.path)
        #expect(cache.cachedPath(for: .dev) == nil)
    }
}
