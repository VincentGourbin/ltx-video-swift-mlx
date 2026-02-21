// MemoryOptimizationConfig.swift - Memory optimization for LTX-2 generation
// Copyright 2025

import Foundation

/// Memory optimization configuration for LTX-2 generation
///
/// Controls how aggressively the pipeline manages GPU memory during generation.
/// Higher optimization levels trade speed for lower peak memory usage.
///
/// Follows the pattern from flux-2-swift-mlx.
public struct MemoryOptimizationConfig: Sendable {
    /// How often to evaluate lazy computation graphs (every N transformer blocks)
    /// Lower values = more frequent eval = lower peak memory but slower
    public var evalFrequency: Int

    /// Whether to call Memory.clearCache() after evaluation
    public var clearCacheOnEval: Bool

    /// Whether to unload each component after use in the pipeline
    /// (e.g., unload text encoder before loading transformer)
    public var unloadAfterUse: Bool

    /// Sleep duration (seconds) after unloading a component, to allow GPU memory reclaim
    public var unloadSleepSeconds: Double

    public init(
        evalFrequency: Int = 4,
        clearCacheOnEval: Bool = false,
        unloadAfterUse: Bool = true,
        unloadSleepSeconds: Double = 0.5
    ) {
        self.evalFrequency = evalFrequency
        self.clearCacheOnEval = clearCacheOnEval
        self.unloadAfterUse = unloadAfterUse
        self.unloadSleepSeconds = unloadSleepSeconds
    }

    // MARK: - Presets

    /// No optimization — keep everything in memory, eval every 8 blocks
    public static let disabled = MemoryOptimizationConfig(
        evalFrequency: 8,
        clearCacheOnEval: false,
        unloadAfterUse: false,
        unloadSleepSeconds: 0
    )

    /// Light optimization — eval every 4 blocks, unload after use
    public static let light = MemoryOptimizationConfig(
        evalFrequency: 4,
        clearCacheOnEval: false,
        unloadAfterUse: true,
        unloadSleepSeconds: 0.3
    )

    /// Moderate optimization — eval every 2 blocks, clear cache
    public static let moderate = MemoryOptimizationConfig(
        evalFrequency: 2,
        clearCacheOnEval: true,
        unloadAfterUse: true,
        unloadSleepSeconds: 0.5
    )

    /// Aggressive optimization — eval every block, clear cache
    public static let aggressive = MemoryOptimizationConfig(
        evalFrequency: 1,
        clearCacheOnEval: true,
        unloadAfterUse: true,
        unloadSleepSeconds: 1.0
    )

    /// Default preset
    public static let `default` = MemoryOptimizationConfig.light

    /// Auto-select preset based on available system RAM
    public static func recommended(forRAMGB ram: Int) -> MemoryOptimizationConfig {
        switch ram {
        case ...32:
            return .aggressive
        case 33...64:
            return .moderate
        case 65...96:
            return .light
        default:
            return .disabled
        }
    }
}
