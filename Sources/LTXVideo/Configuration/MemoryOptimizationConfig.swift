// MemoryOptimizationConfig.swift - Memory optimization for LTX-2 generation
// Copyright 2025

import Foundation

/// Controls how aggressively the pipeline manages GPU memory during generation.
///
/// Higher optimization levels trade speed for lower peak memory usage. Choose
/// a preset based on your machine's available RAM, or use
/// ``recommended(forRAMGB:)`` for automatic selection.
///
/// ## Presets
/// | Preset | Eval Freq | Cache Clear | Unload | VAE Tiling | Best For |
/// |--------|-----------|-------------|--------|------------|----------|
/// | ``disabled`` | 8 | No | No | No | 128+ GB |
/// | ``light`` | 4 | No | Yes | No | 64-96 GB |
/// | ``moderate`` | 2 | Yes | Yes | 8 frames | 32-64 GB |
/// | ``aggressive`` | 1 | Yes | Yes | 6 frames | 16-32 GB |
///
/// ## Example
/// ```swift
/// let pipeline = LTXPipeline(
///     model: .distilled,
///     memoryOptimization: .moderate
/// )
/// ```
public struct MemoryOptimizationConfig: Sendable {
    /// How often to evaluate lazy computation graphs (every N transformer blocks)
    /// Lower values = more frequent eval = lower peak memory but slower
    public var evalFrequency: Int

    /// Whether to call Memory.clearCache() after evaluation
    public var clearCacheOnEval: Bool

    /// Whether to unload each component after use in the pipeline
    /// (e.g., unload text encoder before loading transformer)
    ///
    /// This is the coarse switch. ``unloadTextEncoderAfterUse`` and
    /// ``unloadTransformerAfterUse`` override it per component when set.
    public var unloadAfterUse: Bool

    /// Whether to drop the prompt encoder once the text is encoded.
    /// `nil` (default) follows ``unloadAfterUse``.
    ///
    /// Worth setting independently on LTX-2.5: its Gemma 4 encoder is 26 GB,
    /// against 7.5 GB for the Gemma 3 of LTX-2.3, so keeping it resident to make
    /// consecutive runs cheap is a very different bargain than it used to be.
    /// See ``keepingTransformer()``.
    public var unloadTextEncoderAfterUse: Bool?

    /// Whether to drop the transformer before the VAE decode.
    /// `nil` (default) follows ``unloadAfterUse``.
    ///
    /// Setting this to `false` is what makes LipDub fusion reuse reachable: the
    /// unload also clears the fusion record, so the next segment has to reload
    /// **and** re-fuse the 22B.
    public var unloadTransformerAfterUse: Bool?

    /// Whether the prompt encoder is dropped after text encoding.
    public var unloadsTextEncoder: Bool { unloadTextEncoderAfterUse ?? unloadAfterUse }

    /// Whether the transformer is dropped before the VAE decode.
    public var unloadsTransformer: Bool { unloadTransformerAfterUse ?? unloadAfterUse }

    /// Sleep duration (seconds) after unloading a component, to allow GPU memory reclaim
    public var unloadSleepSeconds: Double

    /// VAE temporal tile size (latent frames per chunk). 0 = disabled (decode all at once).
    /// For long videos, tiling reduces peak VAE memory by ~75%.
    /// Recommended: 8 for videos > 97 frames, 0 for shorter videos.
    public var vaeTemporalTileSize: Int

    /// VAE temporal tile overlap (latent frames). Blended with linear interpolation.
    public var vaeTemporalTileOverlap: Int

    public init(
        evalFrequency: Int = 4,
        clearCacheOnEval: Bool = false,
        unloadAfterUse: Bool = true,
        unloadSleepSeconds: Double = 0.5,
        vaeTemporalTileSize: Int = 0,
        vaeTemporalTileOverlap: Int = 1,
        unloadTextEncoderAfterUse: Bool? = nil,
        unloadTransformerAfterUse: Bool? = nil
    ) {
        self.evalFrequency = evalFrequency
        self.clearCacheOnEval = clearCacheOnEval
        self.unloadAfterUse = unloadAfterUse
        self.unloadSleepSeconds = unloadSleepSeconds
        self.vaeTemporalTileSize = vaeTemporalTileSize
        self.vaeTemporalTileOverlap = vaeTemporalTileOverlap
        self.unloadTextEncoderAfterUse = unloadTextEncoderAfterUse
        self.unloadTransformerAfterUse = unloadTransformerAfterUse
    }

    // MARK: - Derived presets

    /// The same preset, but keeping the transformer (and any LoRA fused into it)
    /// resident across runs.
    ///
    /// This is the setting for consecutive segments sharing one adapter — LipDub
    /// storyboards, chained dialogue. Everything else keeps unloading, so the
    /// 26 GB LTX-2.5 prompt encoder is still freed after each text encode; the
    /// pipeline reloads it on its own, encoder-only, at the next segment.
    ///
    /// ```swift
    /// let pipeline = LTXPipeline(
    ///     model: .v25Distilled,
    ///     memoryOptimization: .recommended(forRAMGB: ram).keepingTransformer())
    /// ```
    public func keepingTransformer() -> MemoryOptimizationConfig {
        var copy = self
        copy.unloadTransformerAfterUse = false
        return copy
    }

    /// The same preset, but keeping the prompt encoder resident after encoding.
    ///
    /// Saves a reload per run at the cost of 26 GB resident on LTX-2.5 — only
    /// worth it on a machine with headroom to spare.
    public func keepingTextEncoder() -> MemoryOptimizationConfig {
        var copy = self
        copy.unloadTextEncoderAfterUse = false
        return copy
    }

    // MARK: - Presets

    /// No optimization — keep everything in memory, eval every 8 blocks
    public static let disabled = MemoryOptimizationConfig(
        evalFrequency: 8,
        clearCacheOnEval: false,
        unloadAfterUse: false,
        unloadSleepSeconds: 0,
        vaeTemporalTileSize: 0
    )

    /// Light optimization — eval every 4 blocks, unload after use
    public static let light = MemoryOptimizationConfig(
        evalFrequency: 4,
        clearCacheOnEval: false,
        unloadAfterUse: true,
        unloadSleepSeconds: 0.3,
        vaeTemporalTileSize: 0
    )

    /// Moderate optimization — eval every 2 blocks, clear cache, VAE tiling
    public static let moderate = MemoryOptimizationConfig(
        evalFrequency: 2,
        clearCacheOnEval: true,
        unloadAfterUse: true,
        unloadSleepSeconds: 0.5,
        vaeTemporalTileSize: 8,
        vaeTemporalTileOverlap: 1
    )

    /// Aggressive optimization — eval every block, clear cache, VAE tiling
    public static let aggressive = MemoryOptimizationConfig(
        evalFrequency: 1,
        clearCacheOnEval: true,
        unloadAfterUse: true,
        unloadSleepSeconds: 1.0,
        vaeTemporalTileSize: 6,
        vaeTemporalTileOverlap: 1
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
