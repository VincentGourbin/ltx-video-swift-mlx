// LTXVideoCLI.swift - Command-line interface for LTX-2.3 video generation
// Copyright 2025

import ArgumentParser
import Foundation
import LTXVideo

@main
struct LTXVideoCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "ltx-video",
        abstract: "LTX-2.3 video generation on Mac with MLX",
        version: "0.1.0",
        subcommands: [Generate.self, Retake.self, Profile.self, ExportQuantized.self, Download.self, Train.self, TrainingControl.self, Models.self, Info.self],
        defaultSubcommand: Info.self
    )
}

// MARK: - Keyframe Spec Parsing

/// Parse a `--keyframe` argument of the form `PATH:FRAME_IDX[:STRENGTH]`.
///
/// The path may contain colons (e.g. on Windows-style paths) — only the last one
/// or two colon-separated tokens are interpreted as numeric. We rsplit at most
/// twice and try to parse the trailing 1–2 tokens as `FRAME_IDX[, STRENGTH]`.
func parseKeyframeSpec(_ raw: String) throws -> KeyframeInput {
    let parts = raw.split(separator: ":", omittingEmptySubsequences: false).map(String.init)
    guard parts.count >= 2 else {
        throw ValidationError("Invalid --keyframe spec '\(raw)'. Expected PATH:FRAME_IDX[:STRENGTH]")
    }

    // Try 3-token form first (path may contain ':' so we also try a fallback)
    if parts.count >= 3,
       let frameIdx = Int(parts[parts.count - 2]),
       let strength = Float(parts[parts.count - 1]) {
        let path = parts[0..<(parts.count - 2)].joined(separator: ":")
        return KeyframeInput(path: path, pixelFrameIndex: frameIdx, strength: strength)
    }

    // 2-token form: PATH:FRAME_IDX
    if let frameIdx = Int(parts[parts.count - 1]) {
        let path = parts[0..<(parts.count - 1)].joined(separator: ":")
        return KeyframeInput(path: path, pixelFrameIndex: frameIdx)
    }

    throw ValidationError(
        "Invalid --keyframe spec '\(raw)'. Could not parse FRAME_IDX (and optional STRENGTH). " +
        "Expected PATH:FRAME_IDX[:STRENGTH], e.g. first.png:0 or last.png:120:1.0"
    )
}

// MARK: - Generate Command

struct Generate: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Generate a video from a text prompt (always two-stage distilled)"
    )

    @Argument(help: "The text prompt describing the video to generate")
    var prompt: String

    @Option(name: .shortAndLong, help: "Output file path (default: output.mp4)")
    var output: String = "output.mp4"

    @Option(name: .shortAndLong, help: "Video width in pixels (must be divisible by 64 for two-stage)")
    var width: Int = 768

    @Option(name: .shortAndLong, help: "Video height in pixels (must be divisible by 64 for two-stage)")
    var height: Int = 512

    @Option(name: .shortAndLong, help: "Number of frames (must be 8n+1, e.g., 9, 17, 25, 33...)")
    var frames: Int = 121

    @Option(name: .long, help: "Random seed for reproducibility")
    var seed: UInt64?

    @Option(name: .long, help: "Input image for image-to-video generation (shorthand for --keyframe PATH:0)")
    var image: String?

    @Option(name: .long, parsing: .singleValue,
            help: "Keyframe constraint, format PATH:FRAME_IDX[:STRENGTH]. Repeatable. Example: --keyframe first.png:0 --keyframe last.png:120. Latent stride is 8, so two keyframes within the same 8-frame group cannot coexist.")
    var keyframe: [String] = []

    @Flag(name: .long, help: "Generate audio alongside video (dual video/audio denoising)")
    var audio: Bool = false

    @Option(name: .long, help: "Audio gain (linear). 1.0=unchanged, 0.5=-6dB, 0.25=-12dB (default: 1.0)")
    var audioGain: Float = 1.0

    @Flag(name: .long, help: "Enhance prompt using Gemma before generation")
    var enhancePrompt: Bool = false

    @Option(name: .long, help: "Path to LoRA .safetensors file to apply during generation")
    var lora: String?

    @Option(name: .long, help: "LoRA scale factor (default: 1.0)")
    var loraScale: Float = 1.0

    @Option(name: .long, help: "Transformer quantization: bf16, qint8, int4, nvfp4, mxfp8")
    var transformerQuant: String = "bf16"

    @Flag(name: .long, help: "Mixed precision: first/last 6 blocks qint8, middle blocks int4")
    var mixedPrecision: Bool = false

    @Option(name: .long, help: "Video bitrate in kbps (e.g., 1000 for 1 Mbps). Default: quality-based encoding")
    var bitrate: Int?

    @Flag(name: .long, help: "Enable debug output")
    var debug: Bool = false

    @Flag(name: .long, help: "Enable performance profiling")
    var profile: Bool = false

    @Option(name: .long, help: "HuggingFace token for gated models")
    var hfToken: String?

    @Option(name: .long, help: "Custom directory for model storage")
    var modelsDir: String?

    @Option(name: .long, help: "Path to local Gemma model directory")
    var gemmaPath: String?

    @Option(name: .long, help: "Path to local LTX unified weights file")
    var ltxWeights: String?

    mutating func run() async throws {
        // Configure custom models directory if specified
        if let dir = modelsDir {
            LTXModelRegistry.customModelsDirectory = URL(fileURLWithPath: dir)
        }

        // Enable debug mode if requested
        if debug {
            LTXDebug.enableDebugMode()
        }

        // Attach profiling session when --profile is enabled
        var profilingSession: ProfilingSession? = nil
        if profile {
            let session = ProfilingSession(config: ProfilingConfig(trackMemory: true))
            session.title = "LTX-2.3 PROFILING REPORT"
            session.metadata["model"] = "distilled"
            session.metadata["quant"] = mixedPrecision ? "mixed" : transformerQuant
            session.metadata["resolution"] = "\(width)x\(height)"
            session.metadata["frames"] = String(frames)
            session.metadata["steps"] = String(8)
            profilingSession = session
            LTXVideoProfiler.shared.enable()
            LTXVideoProfiler.shared.activeSession = session
        }
        defer {
            if profile {
                LTXVideoProfiler.shared.activeSession = nil
                LTXVideoProfiler.shared.disable()
            }
        }

        // Parse keyframe specs from CLI strings (PATH:FRAME_IDX[:STRENGTH])
        let parsedKeyframes = try keyframe.map { try parseKeyframeSpec($0) }
        if !parsedKeyframes.isEmpty && image != nil {
            throw ValidationError("--image and --keyframe are mutually exclusive (use --keyframe PATH:0 for first-frame conditioning)")
        }
        let isI2V = image != nil || !parsedKeyframes.isEmpty

        print("LTX-2.3 Video Generation (Two-Stage Distilled)")
        print("================================================")
        if !parsedKeyframes.isEmpty {
            print("Mode: keyframe interpolation (\(parsedKeyframes.count) keyframe\(parsedKeyframes.count == 1 ? "" : "s"))")
            for kf in parsedKeyframes {
                print("  Keyframe @ pixel frame \(kf.pixelFrameIndex): \(kf.path)")
            }
        } else {
            print("Mode: \(isI2V ? "image-to-video" : "text-to-video")")
            if let imagePath = image {
                print("Input image: \(imagePath)")
            }
        }
        print("Prompt: \(prompt)")
        print("Output: \(output)")
        print("Resolution: \(width)x\(height) (stage 1: \(width/2)x\(height/2))")
        print("Frames: \(frames)")
        if let seed = seed {
            print("Seed: \(seed)")
        }
        if enhancePrompt {
            if image != nil {
                print("Prompt enhancement: enabled (multimodal I2V)")
            } else {
                print("Prompt enhancement: enabled (text-only T2V)")
            }
        }
        if let loraPath = lora {
            print("LoRA: \(loraPath) (scale: \(loraScale))")
        }
        if audio { print("Audio: enabled (dual video/audio generation)") }
        if transformerQuant != "bf16" { print("Transformer quantization: \(transformerQuant)") }
        print()

        // Validate frame count (must be 8n+1)
        guard (frames - 1) % 8 == 0 else {
            throw ValidationError("Frame count must be 8n+1 (e.g., 9, 17, 25, 33, ...). Got \(frames)")
        }

        // Validate dimensions (must be divisible by 64 for two-stage)
        guard width % 64 == 0 && height % 64 == 0 else {
            throw ValidationError("Width and height must be divisible by 64. Got \(width)x\(height)")
        }

        // Parse transformer quantization
        let quantConfig: LTXQuantizationConfig
        if mixedPrecision {
            quantConfig = .mixedDefault
            print("Quantization: mixed precision (first/last 6 blocks qint8, middle int4)")
        } else {
            guard let quantOption = TransformerQuantization(rawValue: transformerQuant) else {
                throw ValidationError("Invalid transformer quantization: \(transformerQuant). Use: bf16, qint8, or int4")
            }
            quantConfig = LTXQuantizationConfig(transformer: quantOption)
        }

        // Create pipeline (always distilled)
        print("Creating pipeline...")
        fflush(stdout)
        let pipeline = LTXPipeline(
            model: .distilled,
            quantization: quantConfig,
            hfToken: hfToken
        )
        print("Pipeline created")
        fflush(stdout)

        // Load models
        print("Loading models (this may take a while)...")
        fflush(stdout)
        let startLoad = Date()
        try await pipeline.loadModels(
            progressCallback: { progress in
                print("  \(progress.message) (\(Int(progress.progress * 100))%)")
            },
            gemmaModelPath: gemmaPath,
            ltxWeightsPath: ltxWeights
        )
        let loadTime = Date().timeIntervalSince(startLoad)
        print("Models loaded in \(String(format: "%.1f", loadTime))s")

        // Load audio models if requested
        if audio {
            print("Loading audio models...")
            fflush(stdout)
            try await pipeline.loadAudioModels { progress in
                print("  \(progress.message) (\(Int(progress.progress * 100))%)")
            }
            print("Audio models loaded")
        }

        // Fuse LoRA if specified (after model loading, before generation)
        if let loraPath = lora {
            print("Fusing LoRA weights...")
            fflush(stdout)
            let fusedCount = try await pipeline.fuseLoRA(from: loraPath, scale: loraScale)
            print("LoRA fused (\(fusedCount) weight updates)")
        }

        // Download upscaler (always needed for two-stage)
        print("Downloading upscaler weights (if needed)...")
        fflush(stdout)
        let upscalerPath = try await pipeline.downloadUpscalerWeights()
        print("Upscaler weights ready")

        // Build config (distilled defaults: 8 steps, no CFG, no STG)
        let config = LTXVideoGenerationConfig(
            width: width,
            height: height,
            numFrames: frames,
            numSteps: 8,
            seed: seed,
            enhancePrompt: enhancePrompt,
            imagePath: parsedKeyframes.isEmpty ? image : nil,
            keyframes: parsedKeyframes
        )

        // Generate — ONE API call
        print("\nGenerating video...")
        fflush(stdout)
        let startGen = Date()

        let result = try await pipeline.generateVideo(
            prompt: prompt,
            config: config,
            upscalerWeightsPath: upscalerPath,
            onProgress: { progress in
                print("  \(progress.status)")
            },
        )

        let genTime = Date().timeIntervalSince(startGen)
        print("Generation completed in \(String(format: "%.1f", genTime))s")
        fflush(stdout)

        // Export video
        print("\nExporting to \(output)...")
        fflush(stdout)
        let outputURL = URL(fileURLWithPath: output)

        // Configure video export with optional bitrate
        var exportConfig = VideoExportConfig.default
        if let kbps = bitrate {
            exportConfig.averageBitRate = kbps * 1000
            print("Using target bitrate: \(kbps) kbps")
        }

        if let audioWaveform = result.audioWaveform, let audioSampleRate = result.audioSampleRate {
            // Export video with muxed audio
            let videoURL = try await VideoExporter.exportVideo(
                frames: result.frames,
                width: width,
                height: height,
                fps: 24.0,
                audioWaveform: audioWaveform,
                audioSampleRate: audioSampleRate,
                audioGain: audioGain,
                config: exportConfig,
                to: outputURL
            )
            print("Video+audio saved to: \(videoURL.path)")

            // Also export standalone WAV
            let wavPath = output.replacingOccurrences(of: ".mp4", with: ".wav")
            try AudioExporter.exportToWAV(
                waveform: audioWaveform,
                sampleRate: audioSampleRate,
                audioGain: audioGain,
                path: wavPath
            )
            print("Audio WAV saved to: \(wavPath)")
        } else {
            let videoURL = try await VideoExporter.exportVideo(
                frames: result.frames,
                width: width,
                height: height,
                fps: 24.0,
                config: exportConfig,
                to: outputURL
            )
            print("Video saved to: \(videoURL.path)")
        }

        // Print summary
        print("\n--- Summary ---")
        print("Seed: \(result.seed)")
        print("Generation time: \(String(format: "%.1f", result.generationTime))s")
        if result.audioWaveform != nil {
            print("Audio sample rate: \(result.audioSampleRate ?? 0) Hz")
        }

        // Print profiling report + export Chrome Trace
        if let session = profilingSession {
            print(session.generateReport())
            let traceData = ChromeTraceExporter.export(session: session)
            let tracePath = output.replacingOccurrences(of: ".mp4", with: "_trace.json")
            try traceData.write(to: URL(fileURLWithPath: tracePath))
            print("Chrome Trace: \(tracePath)  (open in https://ui.perfetto.dev/)")
        }
    }
}

// MARK: - Retake Command

struct Retake: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Retake (video-to-video): regenerate a video with a new prompt"
    )

    @Argument(help: "The new text prompt for the retaken video")
    var prompt: String

    @Option(name: .long, help: "Source video path (required)")
    var video: String

    @Option(name: .long, help: "Retake strength: 0.0=keep original, 1.0=full regeneration (default: 0.8)")
    var strength: Float = 0.8

    @Option(name: .long, help: "Start time in seconds for partial retake (default: retake all frames)")
    var startTime: Float?

    @Option(name: .long, help: "End time in seconds for partial retake (default: retake all frames)")
    var endTime: Float?

    @Option(name: .shortAndLong, help: "Output file path (default: retake.mp4)")
    var output: String = "retake.mp4"

    @Option(name: .shortAndLong, help: "Video width in pixels (must be divisible by 64)")
    var width: Int = 768

    @Option(name: .shortAndLong, help: "Video height in pixels (must be divisible by 64)")
    var height: Int = 512

    @Option(name: .shortAndLong, help: "Number of frames (must be 8n+1, e.g., 9, 17, 25, 33...)")
    var frames: Int = 121

    @Option(name: .long, help: "Random seed for reproducibility")
    var seed: UInt64?

    @Flag(name: .long, help: "Enhance prompt using Gemma before generation")
    var enhancePrompt: Bool = false

    @Flag(name: .long, help: "Use distilled model (8 steps, fast) instead of dev (30 steps + CFG)")
    var distilled: Bool = false

    @Flag(name: .long, help: "Regenerate audio (instead of preserving source audio)")
    var regenerateAudio: Bool = false

    @Option(name: .long, help: "Transformer quantization: bf16, qint8, int4, nvfp4, mxfp8")
    var transformerQuant: String = "bf16"

    @Flag(name: .long, help: "Mixed precision: first/last 6 blocks qint8, middle blocks int4")
    var mixedPrecision: Bool = false

    @Option(name: .long, help: "Video bitrate in kbps (e.g., 1000 for 1 Mbps). Default: quality-based encoding")
    var bitrate: Int?

    @Flag(name: .long, help: "Enable debug output")
    var debug: Bool = false

    @Flag(name: .long, help: "Enable performance profiling")
    var profile: Bool = false

    @Option(name: .long, help: "HuggingFace token for gated models")
    var hfToken: String?

    @Option(name: .long, help: "Custom directory for model storage")
    var modelsDir: String?

    @Option(name: .long, help: "Path to local Gemma model directory")
    var gemmaPath: String?

    @Option(name: .long, help: "Path to local LTX unified weights file")
    var ltxWeights: String?

    mutating func run() async throws {
        if let dir = modelsDir {
            LTXModelRegistry.customModelsDirectory = URL(fileURLWithPath: dir)
        }

        if debug {
            LTXDebug.enableDebugMode()
        }

        // Attach profiling session when --profile is enabled
        var profilingSession: ProfilingSession? = nil
        if profile {
            let session = ProfilingSession(config: ProfilingConfig(trackMemory: true))
            session.title = "LTX-2.3 PROFILING REPORT"
            session.metadata["model"] = distilled ? "distilled" : "dev"
            session.metadata["quant"] = mixedPrecision ? "mixed" : transformerQuant
            session.metadata["resolution"] = "\(width)x\(height)"
            session.metadata["frames"] = String(frames)
            session.metadata["steps"] = String(distilled ? 8 : 30)
            profilingSession = session
            LTXVideoProfiler.shared.enable()
            LTXVideoProfiler.shared.activeSession = session
        }
        defer {
            if profile {
                LTXVideoProfiler.shared.activeSession = nil
                LTXVideoProfiler.shared.disable()
            }
        }

        print("LTX-2.3 Video Retake (Single-Stage)")
        print("====================================")
        print("Source video: \(video)")
        print("Prompt: \(prompt)")
        if let st = startTime, let et = endTime {
            print("Partial retake: \(st)s - \(et)s")
        } else if let st = startTime {
            print("Partial retake: \(st)s - end")
        } else if let et = endTime {
            print("Partial retake: start - \(et)s")
        }
        print("Output: \(output)")
        print("Resolution: \(width)x\(height)")
        print("Frames: \(frames)")
        if let seed = seed {
            print("Seed: \(seed)")
        }
        if enhancePrompt { print("Prompt enhancement: enabled") }
        if transformerQuant != "bf16" { print("Transformer quantization: \(transformerQuant)") }
        print()

        // Validate
        guard (frames - 1) % 8 == 0 else {
            throw ValidationError("Frame count must be 8n+1 (e.g., 9, 17, 25, 33, ...). Got \(frames)")
        }
        guard width % 64 == 0 && height % 64 == 0 else {
            throw ValidationError("Width and height must be divisible by 64. Got \(width)x\(height)")
        }
        guard FileManager.default.fileExists(atPath: video) else {
            throw ValidationError("Source video not found: \(video)")
        }
        // Parse quantization
        let quantConfig: LTXQuantizationConfig
        if mixedPrecision {
            quantConfig = .mixedDefault
        } else {
            guard let quantOption = TransformerQuantization(rawValue: transformerQuant) else {
                throw ValidationError("Invalid transformer quantization: \(transformerQuant). Use: bf16, qint8, or int4")
            }
            quantConfig = LTXQuantizationConfig(transformer: quantOption)
        }

        // Create pipeline
        let retakeModel: LTXModel = distilled ? .distilled : .dev
        print("Creating pipeline...")
        fflush(stdout)
        let pipeline = LTXPipeline(
            model: retakeModel,
            quantization: quantConfig,
            hfToken: hfToken
        )
        print("Pipeline created (\(retakeModel.rawValue) model)")
        fflush(stdout)

        // Load models
        print("Loading models (this may take a while)...")
        fflush(stdout)
        let startLoad = Date()
        try await pipeline.loadModels(
            progressCallback: { progress in
                print("  \(progress.message) (\(Int(progress.progress * 100))%)")
            },
            gemmaModelPath: gemmaPath,
            ltxWeightsPath: ltxWeights
        )
        let loadTime = Date().timeIntervalSince(startLoad)
        print("Models loaded in \(String(format: "%.1f", loadTime))s")

        // Load audio models if regenerating audio
        if regenerateAudio {
            print("Loading audio models (with encoder)...")
            fflush(stdout)
            try await pipeline.loadAudioModels(includeEncoder: true) { progress in
                print("  \(progress.message) (\(Int(progress.progress * 100))%)")
            }
            print("Audio models loaded")
        }

        // Build config
        let config = LTXVideoGenerationConfig(
            width: width,
            height: height,
            numFrames: frames,
            numSteps: 8,
            seed: seed,
            enhancePrompt: enhancePrompt,
            videoPath: video,
            retakeStrength: strength,
            retakeStartTime: startTime,
            retakeEndTime: endTime,
            regenerateAudio: regenerateAudio
        )

        // Generate retake (single-stage, no upscaler needed)
        print("\nGenerating retake...")
        fflush(stdout)
        let startGen = Date()

        let result = try await pipeline.generateRetake(
            prompt: prompt,
            config: config,
            upscalerWeightsPath: "",  // unused in single-stage retake
            onProgress: { progress in
                print("  \(progress.status)")
            },
        )

        let genTime = Date().timeIntervalSince(startGen)
        print("Retake completed in \(String(format: "%.1f", genTime))s")
        fflush(stdout)

        // Export video
        print("\nExporting to \(output)...")
        fflush(stdout)
        let outputURL = URL(fileURLWithPath: output)

        var exportConfig = VideoExportConfig.default
        if let kbps = bitrate {
            exportConfig.averageBitRate = kbps * 1000
            print("Using target bitrate: \(kbps) kbps")
        }

        if let audioWaveform = result.audioWaveform, let audioSampleRate = result.audioSampleRate {
            // Regenerated audio: mux video + new audio
            let videoURL = try await VideoExporter.exportVideo(
                frames: result.frames,
                width: width,
                height: height,
                fps: 24.0,
                audioWaveform: audioWaveform,
                audioSampleRate: audioSampleRate,
                config: exportConfig,
                to: outputURL
            )
            print("Video saved to: \(videoURL.path)")
            print("Audio: regenerated")

            // Also export standalone WAV
            let wavPath = output.replacingOccurrences(of: ".mp4", with: ".wav")
            try AudioExporter.exportToWAV(
                waveform: audioWaveform,
                sampleRate: audioSampleRate,
                path: wavPath
            )
            print("Audio WAV saved to: \(wavPath)")
        } else {
            // Source audio passthrough (copy audio track directly, no re-encode)
            let sourceAudioURL = URL(fileURLWithPath: video)
            let videoURL = try await VideoExporter.exportVideo(
                frames: result.frames,
                width: width,
                height: height,
                fps: 24.0,
                sourceAudioURL: sourceAudioURL,
                config: exportConfig,
                to: outputURL
            )
            print("Video saved to: \(videoURL.path)")
            print("Audio: source audio preserved")
        }

        // Print summary
        print("\n--- Summary ---")
        print("Seed: \(result.seed)")
        print("Strength: \(strength)")
        print("Generation time: \(String(format: "%.1f", result.generationTime))s")

        if let session = profilingSession {
            print(session.generateReport())
            let traceData = ChromeTraceExporter.export(session: session)
            let tracePath = output.replacingOccurrences(of: ".mp4", with: "_trace.json")
            try traceData.write(to: URL(fileURLWithPath: tracePath))
            print("Chrome Trace: \(tracePath)  (open in https://ui.perfetto.dev/)")
        }
    }
}

// MARK: - Export Quantized Command

struct ExportQuantized: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "export-quantized",
        abstract: "Export a quantized transformer to safetensors (nvfp4, mxfp8, qint8, int4)"
    )

    @Option(name: .long, help: "Path to input unified weights (bf16 safetensors)")
    var input: String

    @Option(name: .shortAndLong, help: "Output safetensors file path")
    var output: String

    @Option(name: .long, help: "Quantization mode: nvfp4, mxfp8, qint8, int4")
    var mode: String = "nvfp4"

    mutating func run() async throws {
        guard let quantOption = TransformerQuantization(rawValue: mode) else {
            throw ValidationError("Invalid mode: \(mode). Use: nvfp4, mxfp8, qint8, int4")
        }
        guard quantOption.needsQuantization else {
            throw ValidationError("Mode bf16 does not need quantization — just copy the file")
        }

        print("Export Quantized Transformer")
        print("===========================")
        print("Input: \(input)")
        print("Output: \(output)")
        print("Mode: \(quantOption.displayName) (groupSize=\(quantOption.groupSize), mode=\(quantOption.quantizationMode))")
        print()

        // Use pipeline to handle weight loading (it knows the mapping)
        let quantConfig = LTXQuantizationConfig(transformer: quantOption)
        let pipeline = LTXPipeline(model: .distilled, quantization: quantConfig)

        print("Loading and quantizing transformer...")
        fflush(stdout)
        let startLoad = Date()
        try await pipeline.loadModels(
            progressCallback: { progress in
                print("  \(progress.message) (\(Int(progress.progress * 100))%)")
            },
            ltxWeightsPath: input
        )
        print("Loaded and quantized in \(String(format: "%.1f", Date().timeIntervalSince(startLoad)))s")

        // Export quantized parameters
        print("Exporting to \(output)...")
        fflush(stdout)
        try await pipeline.exportQuantizedTransformer(to: output)

        let fileSize = try FileManager.default.attributesOfItem(atPath: output)[.size] as? Int64 ?? 0
        let fileSizeGB = Double(fileSize) / (1024 * 1024 * 1024)
        print("Exported: \(output) (\(String(format: "%.1f", fileSizeGB)) GB)")
        print("Done!")
    }
}

// MARK: - Download Command

struct Download: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Download model weights from HuggingFace"
    )

    @Option(name: .long, help: "HuggingFace token for gated models")
    var hfToken: String?

    @Option(name: .long, help: "Custom directory for model storage (default: ~/Library/Caches/ltx-video-mlx)")
    var modelsDir: String?

    @Flag(name: .long, help: "Force re-download even if files exist")
    var force: Bool = false

    mutating func run() async throws {
        // Configure custom models directory if specified
        if let dir = modelsDir {
            LTXModelRegistry.customModelsDirectory = URL(fileURLWithPath: dir)
        }

        let modelVariant = LTXModel.distilled

        print("LTX-2.3 Model Download")
        print("====================")
        print("Model: \(modelVariant.displayName)")
        print("Repository: \(modelVariant.huggingFaceRepo)")
        print("Estimated RAM: ~\(modelVariant.estimatedVRAM)GB")
        print()

        let downloader = ModelDownloader(hfToken: hfToken)

        // Download all components
        print("Downloading all components for \(modelVariant.displayName)...")
        let paths = try await downloader.downloadAllComponents(model: modelVariant) { progress in
            if let file = progress.currentFile {
                print("  [\(Int(progress.progress * 100))%] \(file)")
            } else {
                print("  \(progress.message)")
            }
        }
        print()
        print("Text encoder: \(paths.textEncoderDir.path)")
        print("Tokenizer: \(paths.tokenizerDir.path)")
        print("Unified weights: \(paths.unifiedWeightsPath.path)")
    }
}

// MARK: - Models Command

struct Models: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "List available models and their capabilities"
    )

    mutating func run() throws {
        LTXModel.printModelList()
    }
}

// MARK: - Info Command

struct Info: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Show information about LTX-2.3 implementation"
    )

    mutating func run() throws {
        print(
            """
            LTX-2.3 Video Generation for Apple Silicon
            =========================================

            Version: \(LTXVideo.version)
            Platform: macOS (Apple Silicon with MLX)

            Pipeline: Two-Stage Distilled (matching HF Space)
              Stage 1: Half resolution, 8 distilled Euler steps
              Upscale: 2x spatial upscaler
              Stage 2: Full resolution, 3 distilled refinement steps
              Audio: Optional dual video/audio denoising

            Constraints:
              Frame count: Must be 8n+1 (9, 17, 25, 33, 41, 49, ...)
              Resolution: Width and height must be divisible by 64
              Recommended: 768x512, 1024x576, 832x480

            Commands:
              generate      - Generate video (T2V or I2V, optional audio)
              retake        - Retake: regenerate a video with a new prompt (V2V)
              download      - Download model weights
              train         - Fine-tune models with LoRA
              info          - Show this information

            Examples:
              # Quick test (small video)
              ltx-video generate "A red ball bouncing" -w 256 -h 256 -f 9

              # Standard quality (two-stage distilled)
              ltx-video generate "Ocean waves at sunset" -w 768 -h 512 -f 121

              # With audio
              ltx-video generate "Birds singing in a forest" -w 768 -h 512 -f 121 --audio

              # Image-to-video
              ltx-video generate "Make this come alive" --image photo.jpg -w 768 -h 512 -f 25

              # Retake: change a beaver to a cat
              ltx-video retake "A cat playing in a forest stream" \
                --video beaver.mp4 --strength 0.8 -w 768 -h 512 -f 121

              # With prompt enhancement
              ltx-video generate "A beaver" -w 768 -h 512 -f 121 --enhance-prompt

              # With quantization (lower memory)
              ltx-video generate "A sunset" -w 768 -h 512 -f 121 --transformer-quant qint8
            """)
    }
}
