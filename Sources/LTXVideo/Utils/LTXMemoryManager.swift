// LTXMemoryManager.swift - Centralized memory management for LTX-2
// Copyright 2025

import Foundation
@preconcurrency import MLX

/// Centralized GPU memory management for the LTX-2 generation pipeline.
///
/// Provides structured memory monitoring, cache clearing, and phase-based
/// cache limit control. Use this to inspect or manually manage GPU memory
/// between generation runs.
///
/// ## Monitoring
/// ```swift
/// LTXMemoryManager.logMemoryState("before generation")
/// // ... generate video ...
/// LTXMemoryManager.logMemoryState("after generation")
/// ```
///
/// ## Manual Cleanup
/// ```swift
/// LTXMemoryManager.fullCleanup()  // clear cache + synchronize
/// ```
///
/// - Note: The pipeline automatically calls these methods at phase transitions
///   when ``MemoryOptimizationConfig`` is configured.
public enum LTXMemoryManager {

    /// Clear the GPU buffer cache
    public static func clearCache() {
        Memory.clearCache()
    }

    /// Full cleanup: clear cache and force synchronization
    public static func fullCleanup() {
        Memory.clearCache()
        eval([MLXArray]())
    }

    /// Log current GPU memory state
    public static func logMemoryState(_ label: String = "") {
        let snapshot = Memory.snapshot()
        let prefix = label.isEmpty ? "[MEM]" : "[MEM] \(label):"
        LTXDebug.log("\(prefix) active=\(snapshot.activeMemory / (1024*1024))MB peak=\(snapshot.peakMemory / (1024*1024))MB cache=\(snapshot.cacheMemory / (1024*1024))MB")
    }

    /// Get current memory snapshot for comparison
    public static func snapshot() -> Memory.Snapshot {
        Memory.snapshot()
    }

    /// Log memory delta between two snapshots
    public static func logDelta(from start: Memory.Snapshot, label: String) {
        let end = Memory.snapshot()
        let delta = start.delta(end)
        LTXDebug.log("[MEM] \(label) delta: \(delta.description)")
    }

    // MARK: - Phase-Based Cache Limits

    /// Pipeline phases with recommended cache limits
    public enum Phase {
        case textEncoding
        case denoising
        case vaeDecode
        case idle

        /// Recommended cache limit in bytes for this phase
        var recommendedCacheLimit: Int {
            switch self {
            case .textEncoding: return 512 * 1024 * 1024    // 512MB
            case .denoising:    return 2048 * 1024 * 1024   // 2GB
            case .vaeDecode:    return 512 * 1024 * 1024    // 512MB
            case .idle:         return 0                     // No limit (default)
            }
        }
    }

    /// Set cache limit for a specific pipeline phase
    ///
    /// - Parameter phase: The pipeline phase to configure for
    public static func setPhase(_ phase: Phase) {
        let limit = phase.recommendedCacheLimit
        if limit > 0 {
            Memory.cacheLimit = limit
            LTXDebug.log("[MEM] Phase \(phase): cache limit set to \(limit / (1024*1024))MB")
        } else {
            // Reset to default (no explicit limit)
            Memory.cacheLimit = 0
            LTXDebug.log("[MEM] Phase \(phase): cache limit reset to default")
        }
    }

    /// Reset cache limit to default
    public static func resetCacheLimit() {
        Memory.cacheLimit = 0
    }
}
