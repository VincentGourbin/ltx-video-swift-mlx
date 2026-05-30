import Foundation

/// In-memory cache of resolved unified safetensors paths, keyed by model.
///
/// Used by `LTXPipeline` so that when a caller supplies a local unified
/// weights file (via `loadModels(..., ltxWeightsPath:)`), later lazy loads
/// — e.g. the I2V VAE encoder or the audio transformer — reuse that same
/// file instead of falling back to the downloader.
///
/// Cache entries are validated on read: if the underlying file has
/// disappeared since it was stored, the stale entry is evicted and
/// `cachedPath(for:)` returns `nil` so the caller re-resolves.
///
/// The `fileExists` probe is injectable so the cache can be unit-tested
/// without touching the real filesystem.
struct UnifiedWeightsPathCache {
    private var paths: [LTXModel: String] = [:]
    private let fileExists: (String) -> Bool

    init(fileExists: @escaping (String) -> Bool = { FileManager.default.fileExists(atPath: $0) }) {
        self.fileExists = fileExists
    }

    /// Returns the cached path for `model` if one is stored AND still exists on disk.
    /// Evicts and returns `nil` if the cached file is missing.
    mutating func cachedPath(for model: LTXModel) -> String? {
        guard let cached = paths[model] else { return nil }
        if fileExists(cached) {
            return cached
        }
        paths[model] = nil
        return nil
    }

    mutating func store(_ path: String, for model: LTXModel) {
        paths[model] = path
    }

    mutating func clear() {
        paths.removeAll()
    }
}
