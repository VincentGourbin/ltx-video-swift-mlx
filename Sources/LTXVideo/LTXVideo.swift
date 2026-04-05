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

/// Performance profiler for LTX-2 operations.
///
/// Thread-safe singleton that bridges with `ProfilingSession` for rich
/// Chrome Trace export and memory timeline tracking.
public class LTXVideoProfiler: @unchecked Sendable {
    public static let shared = LTXVideoProfiler()

    public var isEnabled: Bool = false

    /// Optional profiling session for rich data collection (Chrome Trace, memory timeline)
    public var activeSession: ProfilingSession? = nil

    private var timings: [TimingEntry] = []
    private var stepTimes: [TimeInterval] = []
    private var stepCount: Int = 0
    private var totalSteps: Int = 0
    private var activeTimers: [String: Date] = [:]
    private let queue = DispatchQueue(label: "com.ltxvideo.profiler", attributes: .concurrent)

    private init() {}

    /// Enable profiling
    public func enable() {
        isEnabled = true
        reset()
    }

    /// Disable profiling
    public func disable() {
        isEnabled = false
    }

    /// Set the total step count for denoising
    public func setTotalSteps(_ total: Int) {
        queue.async(flags: .barrier) {
            self.totalSteps = total
            self.stepCount = 0
        }
    }

    /// Start timing an operation
    public func start(_ name: String) {
        guard isEnabled else { return }
        let category = ProfilingSession.inferCategory(name)
        activeSession?.beginPhase(name, category: category)
        queue.async(flags: .barrier) {
            self.activeTimers[name] = Date()
        }
    }

    /// End timing an operation and record it
    public func end(_ name: String) {
        guard isEnabled else { return }
        let endTime = Date()
        let category = ProfilingSession.inferCategory(name)
        activeSession?.endPhase(name, category: category)
        queue.async(flags: .barrier) {
            guard let startTime = self.activeTimers[name] else { return }
            self.activeTimers.removeValue(forKey: name)

            let entry = TimingEntry(
                name: name,
                duration: endTime.timeIntervalSince(startTime),
                startTime: startTime,
                endTime: endTime
            )
            self.timings.append(entry)
        }
    }

    /// Record a timing entry directly
    public func record(_ name: String, duration: TimeInterval) {
        guard isEnabled else { return }
        let now = Date()
        queue.async(flags: .barrier) {
            let entry = TimingEntry(
                name: name,
                duration: duration,
                startTime: now.addingTimeInterval(-duration),
                endTime: now
            )
            self.timings.append(entry)
        }
    }

    /// Record a denoising step time
    public func recordStep(duration: TimeInterval) {
        guard isEnabled else { return }
        queue.async(flags: .barrier) {
            self.stepTimes.append(duration)
            self.stepCount += 1
            let currentStep = self.stepCount
            let total = self.totalSteps
            self.activeSession?.recordDenoisingStep(
                index: currentStep,
                total: total,
                durationUs: UInt64(duration * 1_000_000)
            )
        }
    }

    /// Measure a synchronous operation
    @discardableResult
    public func measure<T>(_ name: String, _ operation: () throws -> T) rethrows -> T {
        guard isEnabled else { return try operation() }

        let startTime = Date()
        let result = try operation()
        let endTime = Date()

        queue.async(flags: .barrier) {
            let entry = TimingEntry(
                name: name,
                duration: endTime.timeIntervalSince(startTime),
                startTime: startTime,
                endTime: endTime
            )
            self.timings.append(entry)
        }

        return result
    }

    /// Clear all recorded timings
    public func reset() {
        queue.async(flags: .barrier) {
            self.timings.removeAll()
            self.activeTimers.removeAll()
            self.stepTimes.removeAll()
            self.stepCount = 0
            self.totalSteps = 0
        }
    }

    /// Get all recorded timings (synchronous read)
    public func getTimings() -> [TimingEntry] {
        var result: [TimingEntry] = []
        queue.sync {
            result = self.timings
        }
        return result
    }

    /// Get step times (synchronous read)
    public func getStepTimes() -> [TimeInterval] {
        var result: [TimeInterval] = []
        queue.sync {
            result = self.stepTimes
        }
        return result
    }

    /// Generate a formatted report
    public func generateReport() -> String {
        var currentTimings: [TimingEntry] = []
        var currentStepTimes: [TimeInterval] = []

        queue.sync {
            currentTimings = self.timings
            currentStepTimes = self.stepTimes
        }

        guard !currentTimings.isEmpty else {
            return "No timing data recorded."
        }

        let totalTime = currentTimings.map(\.duration).reduce(0, +)

        var report = "\n"
        report += "  PHASE TIMINGS:\n"
        report += "  \(String(repeating: "\u{2500}", count: 60))\n"

        for entry in currentTimings {
            let percentage = totalTime > 0 ? (entry.duration / totalTime) * 100 : 0
            let bar = String(repeating: "\u{2588}", count: min(20, Int(percentage / 5)))
            let namePadded = entry.name.padding(toLength: 30, withPad: " ", startingAt: 0)
            let durationPadded = entry.durationFormatted.leftPadding(toLength: 10, withPad: " ")
            report += "  \(namePadded) \(durationPadded)  \(String(format: "%5.1f", percentage))% \(bar)\n"
        }

        report += "  \(String(repeating: "\u{2500}", count: 60))\n"
        let totalPadded = "TOTAL".padding(toLength: 30, withPad: " ", startingAt: 0)
        let totalDurPadded = formatDuration(totalTime).leftPadding(toLength: 10, withPad: " ")
        report += "  \(totalPadded) \(totalDurPadded)  100.0%\n"

        if !currentStepTimes.isEmpty {
            let total = currentStepTimes.reduce(0, +)
            let average = total / Double(currentStepTimes.count)
            let minStep = currentStepTimes.min() ?? 0
            let maxStep = currentStepTimes.max() ?? 0

            report += "\n  DENOISING STEP STATISTICS:\n"
            report += "  \(String(repeating: "\u{2500}", count: 60))\n"
            report += "  Steps:              \(currentStepTimes.count)\n"
            report += "  Total denoising:    \(formatDuration(total))\n"
            report += "  Average per step:   \(formatDuration(average))\n"
            report += "  Fastest step:       \(formatDuration(minStep))\n"
            report += "  Slowest step:       \(formatDuration(maxStep))\n"

            report += "\n  Estimated times for different step counts:\n"
            for stepCount in [8, 15, 30, 40] {
                let estimated = average * Double(stepCount)
                report += "    \(String(format: "%2d", stepCount)) steps: \(formatDuration(estimated))\n"
            }
        }

        // Find bottleneck
        if let slowest = currentTimings.max(by: { $0.duration < $1.duration }) {
            let percentage = totalTime > 0 ? (slowest.duration / totalTime) * 100 : 0
            report += "\n  Bottleneck: \(slowest.name) (\(String(format: "%.1f", percentage))% of total)\n"
        }

        return report
    }

    private func formatDuration(_ duration: TimeInterval) -> String {
        if duration < 1 {
            return String(format: "%.1fms", duration * 1000)
        } else if duration < 60 {
            return String(format: "%.2fs", duration)
        } else {
            let minutes = Int(duration / 60)
            let seconds = duration.truncatingRemainder(dividingBy: 60)
            return String(format: "%dm %.1fs", minutes, seconds)
        }
    }
}

/// Timing entry for a profiled operation
public struct TimingEntry: Sendable {
    public let name: String
    public let duration: TimeInterval
    public let startTime: Date
    public let endTime: Date

    public var durationMs: Double {
        duration * 1000
    }

    public var durationFormatted: String {
        if duration < 1 {
            return String(format: "%.1fms", durationMs)
        } else if duration < 60 {
            return String(format: "%.2fs", duration)
        } else {
            let minutes = Int(duration / 60)
            let seconds = duration.truncatingRemainder(dividingBy: 60)
            return String(format: "%dm %.1fs", minutes, seconds)
        }
    }
}

private extension String {
    func leftPadding(toLength length: Int, withPad character: String) -> String {
        let stringLength = self.count
        if stringLength >= length {
            return self
        }
        return String(repeating: character, count: length - stringLength) + self
    }
}

/// Legacy profiler API — forwards to LTXVideoProfiler
public actor LTXProfiler {
    public static let shared = LTXProfiler()

    private init() {}

    public func enable() {
        LTXVideoProfiler.shared.enable()
    }

    public func disable() {
        LTXVideoProfiler.shared.disable()
    }

    public func start(_ name: String) {
        LTXVideoProfiler.shared.start(name)
    }

    public func end(_ name: String) {
        LTXVideoProfiler.shared.end(name)
    }

    public func generateReport() -> String {
        LTXVideoProfiler.shared.generateReport()
    }

    public func clear() {
        LTXVideoProfiler.shared.reset()
    }
}

// MARK: - Video Generation Result

/// Detailed timing breakdown for each phase of the generation pipeline.
///
/// Populated when profiling is enabled via `profile: true` on generation methods.
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

    /// Peak GPU memory usage in MB (sampled during generation)
    public var peakMemoryMB: Int = 0
    /// Mean GPU memory usage in MB (averaged over denoising steps)
    public var meanMemoryMB: Int = 0
    /// Memory samples collected during denoising (in bytes)
    internal var memorySamples: [Int] = []

    /// Record a memory sample and update peak/mean
    internal mutating func sampleMemory() {
        let snapshot = Memory.snapshot()
        let activeBytes = snapshot.activeMemory
        memorySamples.append(activeBytes)
        let activeMB = activeBytes / (1024 * 1024)
        if activeMB > peakMemoryMB {
            peakMemoryMB = activeMB
        }
        if !memorySamples.isEmpty {
            let totalBytes = memorySamples.reduce(0, +)
            meanMemoryMB = totalBytes / memorySamples.count / (1024 * 1024)
        }
    }

    /// Update peak from Memory.peakMemory (captures GPU-level peak)
    internal mutating func capturePeakMemory() {
        let snapshot = Memory.snapshot()
        let peakMB = snapshot.peakMemory / (1024 * 1024)
        if peakMB > peakMemoryMB {
            peakMemoryMB = peakMB
        }
    }
}

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
    /// `W` = width, `C` = 3 (RGB). Values are uint8 in `[0, 255]`.
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

    /// Per-phase timing breakdown. Only populated when `profile: true`
    /// is passed to the generation method.
    public let timings: GenerationTimings?

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
        timings: GenerationTimings? = nil,
        audioWaveform: MLXArray? = nil,
        audioSampleRate: Int? = nil,
        effectivePrompt: String? = nil,
        sourceAudioPath: String? = nil
    ) {
        self.frames = frames
        self.seed = seed
        self.generationTime = generationTime
        self.timings = timings
        self.audioWaveform = audioWaveform
        self.audioSampleRate = audioSampleRate
        self.effectivePrompt = effectivePrompt
        self.sourceAudioPath = sourceAudioPath
    }
}

