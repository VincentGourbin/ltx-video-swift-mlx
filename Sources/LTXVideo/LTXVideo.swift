// LTXVideo.swift - LTX-2 Video Generation Framework for Apple Silicon
// Copyright 2025

import Foundation
@preconcurrency import MLX
import MLXNN

/// LTX-2 Video Generation Framework for Apple Silicon
///
/// Provides text-to-video generation using the LTX-2 model from Lightricks,
/// optimized for Apple Silicon via the MLX framework.
///
/// ## Quick Start
/// ```swift
/// import LTXVideo
///
/// let pipeline = LTXPipeline(model: .distilled)
/// try await pipeline.loadModels()
/// let upscalerPath = try await pipeline.downloadUpscalerWeights()
///
/// let config = LTXVideoGenerationConfig(
///     width: 768, height: 512, numFrames: 121
/// )
/// let result = try await pipeline.generateVideo(
///     prompt: "A cat walking in a garden",
///     config: config,
///     upscalerWeightsPath: upscalerPath
/// )
/// try await VideoExporter.exportVideo(
///     frames: result.frames, width: 768, height: 512, to: URL(fileURLWithPath: "output.mp4")
/// )
/// ```
///
/// ## Constraints
/// - **Frame count**: Must be `8n + 1` (9, 17, 25, 33, 41, ...)
/// - **Resolution**: Width and height must be divisible by 64
public enum LTXVideo {
    /// Framework version
    public static let version = "0.3.1"

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

// MARK: - Profiler (re-exported from MLXProfiler package)

@_exported import MLXProfiler

/// Convenience alias for backward compatibility
public typealias LTXVideoProfiler = MLXProfiler

// MARK: - Video Generation Result

/// The output of a video generation run.
///
/// Contains the generated frames as an MLX tensor, the seed used for
/// reproducibility, timing information, and convenience accessors for
/// frame count and dimensions.
///
/// ## Exporting
/// Use ``VideoExporter/exportVideo(frames:width:height:fps:to:)`` to save
/// the result as an MP4 file:
/// ```swift
/// try await VideoExporter.exportVideo(
///     frames: result.frames,
///     width: result.width,
///     height: result.height,
///     to: URL(fileURLWithPath: "output.mp4")
/// )
/// ```
public struct VideoGenerationResult: @unchecked Sendable {
    /// Generated video frames as an MLX array.
    ///
    /// Shape: `(F, H, W, C)` where `F` = frame count, `H` = height,
    /// `W` = width, `C` = 3 (RGB). Values are float32 in `[0, 1]`.
    public let frames: MLXArray

    /// Number of generated frames
    public var numFrames: Int { frames.dim(0) }

    /// Frame height in pixels
    public var height: Int { frames.dim(1) }

    /// Frame width in pixels
    public var width: Int { frames.dim(2) }

    /// The random seed used for generation (useful for reproducibility)
    public let seed: UInt64

    /// Total wall-clock generation time in seconds (excludes model loading)
    public let generationTime: TimeInterval

    /// Audio waveform (optional). Present when audio generation is enabled.
    /// Shape: `(samples,)` float32. Sample rate given by ``audioSampleRate``.
    public let audioWaveform: MLXArray?

    /// Audio sample rate in Hz (e.g. 24000). Only set when ``audioWaveform`` is present.
    public let audioSampleRate: Int?

    /// The prompt actually used for generation.
    /// When prompt enhancement is enabled, this is the VLM-enhanced version.
    /// When disabled, this equals the original input prompt.
    public let effectivePrompt: String?

    /// Path to the source audio file for passthrough export (retake only).
    ///
    /// When set, use ``VideoExporter/exportVideo(frames:width:height:fps:sourceAudioURL:config:to:)``
    /// with this path to copy the audio track directly from the source file
    /// (no re-encoding, no quality loss). This avoids AAC encoder issues with
    /// non-standard sample rates.
    ///
    /// - Note: Prefer `sourceAudioPath` over `audioWaveform` for retake results.
    public let sourceAudioPath: String?

    public init(
        frames: MLXArray,
        seed: UInt64,
        generationTime: TimeInterval,
        audioWaveform: MLXArray? = nil,
        audioSampleRate: Int? = nil,
        effectivePrompt: String? = nil,
        sourceAudioPath: String? = nil
    ) {
        self.frames = frames
        self.seed = seed
        self.generationTime = generationTime
        self.audioWaveform = audioWaveform
        self.audioSampleRate = audioSampleRate
        self.effectivePrompt = effectivePrompt
        self.sourceAudioPath = sourceAudioPath
    }
}

