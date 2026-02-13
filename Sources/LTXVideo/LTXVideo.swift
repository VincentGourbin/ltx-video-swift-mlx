// LTXVideo.swift - LTX-2 Video Generation Framework for Apple Silicon
// Copyright 2025

import Foundation
@preconcurrency import MLX
import MLXNN

/// LTX-2 Video Generation Framework
///
/// This framework provides text-to-video generation using the LTX-2 model
/// optimized for Apple Silicon via MLX.
///
/// ## Usage
/// ```swift
/// let pipeline = LTXPipeline(model: .distilledFP8)
/// try await pipeline.loadModels()
/// let video = try await pipeline.generateVideo(
///     prompt: "A cat walking in the garden",
///     width: 512,
///     height: 512,
///     numFrames: 25
/// )
/// try pipeline.exportVideo(video, to: "output.mp4")
/// ```
public enum LTXVideo {
    /// Framework version
    public static let version = "0.1.0"

    /// Framework name
    public static let name = "LTX-Video-Swift-MLX"
}

// MARK: - Errors

/// Errors that can occur during LTX-2 operations
public enum LTXError: Error, LocalizedError, Sendable {
    /// A required model component is not loaded
    case modelNotLoaded(String)

    /// Invalid configuration provided
    case invalidConfiguration(String)

    /// Insufficient memory for the operation
    case insufficientMemory(required: Int, available: Int)

    /// Failed to load weights from file
    case weightLoadingFailed(String)

    /// Failed to download model from HuggingFace
    case downloadFailed(String)

    /// Video processing failed
    case videoProcessingFailed(String)

    /// Generation failed
    case generationFailed(String)

    /// Generation was cancelled by user
    case generationCancelled

    /// Invalid frame count (must be 8n + 1)
    case invalidFrameCount(Int)

    /// Invalid dimensions (must be divisible by 32)
    case invalidDimensions(width: Int, height: Int)

    /// Text encoding failed
    case textEncodingFailed(String)

    /// File not found
    case fileNotFound(String)

    /// Invalid LoRA configuration
    case invalidLoRA(String)

    /// Export failed
    case exportFailed(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotLoaded(let component):
            return "Model component not loaded: \(component)"
        case .invalidConfiguration(let message):
            return "Invalid configuration: \(message)"
        case .insufficientMemory(let required, let available):
            return "Insufficient memory: required \(required)GB, available \(available)GB"
        case .weightLoadingFailed(let message):
            return "Failed to load weights: \(message)"
        case .downloadFailed(let message):
            return "Download failed: \(message)"
        case .videoProcessingFailed(let message):
            return "Video processing failed: \(message)"
        case .generationFailed(let message):
            return "Generation failed: \(message)"
        case .generationCancelled:
            return "Generation was cancelled"
        case .invalidFrameCount(let count):
            return "Invalid frame count: \(count). Must be 8n + 1 (e.g., 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97)"
        case .invalidDimensions(let width, let height):
            return "Invalid dimensions: \(width)x\(height). Both must be divisible by 32"
        case .textEncodingFailed(let message):
            return "Text encoding failed: \(message)"
        case .fileNotFound(let path):
            return "File not found: \(path)"
        case .invalidLoRA(let message):
            return "Invalid LoRA: \(message)"
        case .exportFailed(let message):
            return "Export failed: \(message)"
        }
    }
}

// MARK: - Debug Logging

/// Debug logging utility for LTX-2
public enum LTXDebug {
    /// Whether debug mode is enabled
    public nonisolated(unsafe) static var isEnabled = false

    /// Whether verbose mode is enabled
    public nonisolated(unsafe) static var isVerbose = false

    /// Enable debug mode
    public static func enableDebugMode() {
        isEnabled = true
    }

    /// Enable verbose mode (includes debug)
    public static func enableVerboseMode() {
        isEnabled = true
        isVerbose = true
    }

    /// Disable debug mode
    public static func disable() {
        isEnabled = false
        isVerbose = false
    }

    /// Log a debug message
    public static func log(_ message: String) {
        if isEnabled {
            print("[LTX] \(message)")
            fflush(stdout)
        }
    }

    /// Log a verbose message
    public static func verbose(_ message: String) {
        if isVerbose {
            print("[LTX-V] \(message)")
        }
    }
}

// MARK: - Profiler

/// Performance profiler for LTX-2 operations
public actor LTXProfiler {
    /// Shared profiler instance
    public static let shared = LTXProfiler()

    private var isEnabled = false
    private var timings: [(name: String, duration: TimeInterval)] = []
    private var startTimes: [String: Date] = [:]

    private init() {}

    /// Enable profiling
    public func enable() {
        isEnabled = true
        timings.removeAll()
        startTimes.removeAll()
    }

    /// Disable profiling
    public func disable() {
        isEnabled = false
    }

    /// Start timing an operation
    public func start(_ name: String) {
        guard isEnabled else { return }
        startTimes[name] = Date()
    }

    /// End timing an operation
    public func end(_ name: String) {
        guard isEnabled, let startTime = startTimes[name] else { return }
        let duration = Date().timeIntervalSince(startTime)
        timings.append((name: name, duration: duration))
        startTimes.removeValue(forKey: name)
    }

    /// Generate a profiling report
    public func generateReport() -> String {
        guard !timings.isEmpty else {
            return "No profiling data available"
        }

        var report = "=== LTX Performance Report ===\n"
        var total: TimeInterval = 0

        for (name, duration) in timings {
            report += String(format: "  %@: %.2fs\n", name, duration)
            total += duration
        }

        report += String(format: "  Total: %.2fs\n", total)
        return report
    }

    /// Clear all profiling data
    public func clear() {
        timings.removeAll()
        startTimes.removeAll()
    }
}

// MARK: - Video Generation Result

/// Result of video generation
/// Detailed timing breakdown for generation profiling
public struct GenerationTimings: Sendable {
    /// Text encoding time (Gemma + Feature Extractor + Connector)
    public var textEncoding: TimeInterval = 0
    /// Time per denoising step
    public var denoiseSteps: [TimeInterval] = []
    /// VAE decoding time
    public var vaeDecode: TimeInterval = 0
    /// Total denoising time
    public var totalDenoise: TimeInterval { denoiseSteps.reduce(0, +) }
    /// Average per-step denoising time
    public var avgStepTime: TimeInterval { denoiseSteps.isEmpty ? 0 : totalDenoise / Double(denoiseSteps.count) }
}

public struct VideoGenerationResult: @unchecked Sendable {
    /// Generated frames as MLX array (F, H, W, C) in [0, 255] uint8
    public let frames: MLXArray

    /// Number of frames
    public var numFrames: Int { frames.dim(0) }

    /// Frame height
    public var height: Int { frames.dim(1) }

    /// Frame width
    public var width: Int { frames.dim(2) }

    /// Generation seed used
    public let seed: UInt64

    /// Total generation time in seconds
    public let generationTime: TimeInterval

    /// Detailed timing breakdown (populated when profiling)
    public let timings: GenerationTimings?

    public init(frames: MLXArray, seed: UInt64, generationTime: TimeInterval, timings: GenerationTimings? = nil) {
        self.frames = frames
        self.seed = seed
        self.generationTime = generationTime
        self.timings = timings
    }
}

// MARK: - Re-exports

// Re-export configuration types
public typealias Model = LTXModel
public typealias TransformerConfig = LTXTransformerConfig
public typealias VideoConfig = LTXVideoGenerationConfig
public typealias ModelRegistry = LTXModelRegistry
public typealias QuantizationConfig = LTXQuantizationConfig
