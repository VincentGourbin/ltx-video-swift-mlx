// LTXVideoCLI.swift - Command-line interface for LTX-2 video generation
// Copyright 2025

import ArgumentParser
import Foundation
import LTXVideo

@main
struct LTXVideoCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "ltx-video",
        abstract: "LTX-2 video generation on Mac with MLX (LTX-2.3 and LTX-2.5)",
        version: "0.3.1",
        subcommands: [Generate.self, Retake.self, LipDub.self, Upscale.self, Interpolate.self, Profile.self, ExportQuantized.self, Download.self, Train.self, TrainingControl.self, Models.self, Info.self],
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

    @Option(name: .shortAndLong, help: "Number of frames (8n+1, e.g. 9, 17, 25...), a duration ('15s'), or 'auto' to predict it from the prompt (LTX-2.5 only)")
    var frames: String = "121"

    @Option(name: .long, help: "Inference steps — dev models only (default 30); distilled models run their fixed 8-step schedule")
    var steps: Int?

    @Option(name: .long, help: "Random seed for reproducibility")
    var seed: UInt64?

    @Option(name: .long, help: "Input image for image-to-video generation (shorthand for --keyframe PATH:0)")
    var image: String?

    @Option(name: .long, parsing: .singleValue,
            help: "Keyframe constraint, format PATH:FRAME_IDX[:STRENGTH]. Repeatable. Example: --keyframe first.png:0 --keyframe last.png:120. Latent stride is 8, so two keyframes within the same 8-frame group cannot coexist.")
    var keyframe: [String] = []

    @Option(name: .long, parsing: .singleValue,
            help: "Generate a keyframe at this pixel frame and keep it as an anchor. Repeatable, strictly increasing. LTX-2.5 only. Example: --keyframe-slot 96 --keyframe-slot 192")
    var keyframeSlot: [Int] = []

    @Option(name: .long, help: "Write the generated keyframes (latents) here, for `interpolate --anchors` to reuse")
    var slotsOut: String?

    @Flag(name: .long, help: "Generate audio alongside video (dual video/audio denoising)")
    var audio: Bool = false

    @Option(name: .long, help: "Audio gain (linear). 1.0=unchanged, 0.5=-6dB, 0.25=-12dB (default: 1.0)")
    var audioGain: Float = 1.0

    @Flag(name: .long, help: "Enhance prompt using Gemma before generation")
    var enhancePrompt: Bool = false

    @OptionGroup var promptEnhancer: PromptEnhancerOptions

    @Flag(name: .long, help: "Decode with LTX-2.5's diffusion video VAE instead of the convolutional one (+1.5 GB download, slower, finer detail)")
    var diffvae: Bool = false

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

    @Flag(name: .long, help: "Advertise activity to external monitors (writes a transient manifest in ~/Library/Application Support/ai-runtime-beacons/)")
    var beacon: Bool = false

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

    @Option(name: .long, help: "Model variant: distilled, dev, 2.5-distilled, 2.5-dev (default: distilled)")
    var model: String = "distilled"

    mutating func run() async throws {
        // Resolved up front so every message names the variant actually running.
        let parsedModelVariant = try parseModelVariant(model)

        // Configure custom models directory if specified
        if let dir = modelsDir {
            LTXModelRegistry.customModelsDirectory = URL(fileURLWithPath: dir)
        }

        // Enable debug mode if requested
        if debug {
            LTXDebug.enableDebugMode()
        }

        RuntimeBeacon.isEnabled = beacon

        // Attach profiling session when --profile is enabled
        var profilingSession: ProfilingSession? = nil
        if profile {
            let session = ProfilingSession(config: ProfilingConfig(trackMemory: true))
            session.title = "\(parsedModelVariant.displayName) PROFILING REPORT"
            session.metadata["model"] = parsedModelVariant.rawValue
            session.metadata["quant"] = mixedPrecision ? "mixed" : transformerQuant
            session.metadata["resolution"] = "\(width)x\(height)"
            session.metadata["frames"] = String(frames)
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

        print("\(parsedModelVariant.displayName) — Video Generation (Two-Stage Distilled)")
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
        let autoDuration = frames.lowercased() == "auto"
        if autoDuration && parsedModelVariant.family == .ltx23 {
            throw ValidationError(
                "--frames auto needs a duration head, which ships from LTX-2.5 onward; "
                + "\(parsedModelVariant.displayName) has none. Pass an explicit frame count (8n+1).")
        }
        var frameCount = 121
        if !autoDuration {
            let parsed: (spec: FrameCountSpec, note: String?)
            do {
                parsed = try FrameCountSpec.parse(frames)
            } catch {
                throw ValidationError("\(error.localizedDescription)")
            }
            guard case .frames(let count) = parsed.spec else {
                throw ValidationError("Frames must be a count, a duration, or 'auto'. Got \(frames)")
            }
            if let note = parsed.note { print("Note: \(note)") }
            frameCount = count
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
            // The 2.5 text encoder ships in bf16 only (~24 GB), so it follows the
            // transformer's level rather than staying at full precision.
            quantConfig = LTXQuantizationConfig(transformer: quantOption, textEncoder: quantOption)
        }

        let modelVariant = parsedModelVariant

        print("Creating pipeline...")
        fflush(stdout)
        let pipeline = LTXPipeline(
            model: modelVariant,
            quantization: quantConfig,
            hfToken: hfToken,
            promptEnhancer: try promptEnhancer.resolve()
        )
        print("Pipeline created")
        fflush(stdout)

        // Enhance before anything heavy is resident. The enhancer needs no LTX
        // model, and sharing unified memory with a loaded checkpoint is brutal:
        // measured 3 min standalone vs ~85 min with the video+audio stack
        // (~48 GB) already in memory — same prompt, same settings. Upstream
        // ordering is also enhance-then-duration, so this satisfies both.
        var effectivePrompt = prompt
        if enhancePrompt {
            print("Enhancing prompt...")
            fflush(stdout)
            let promptImage = parsedKeyframes.first?.path ?? image
            effectivePrompt = try await pipeline.enhancePromptWithVLM(prompt, imagePath: promptImage)
        }

        // Load models
        print("Loading models (this may take a while)...")
        fflush(stdout)
        let startLoad = Date()
        try await pipeline.loadModels(
            progressCallback: { progress in
                downloadPrinter.report(progress)
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
                downloadPrinter.report(progress)
            }
            print("Audio models loaded")
        }

        if diffvae {
            print("Loading the diffusion video decoder...")
            fflush(stdout)
            try await pipeline.loadDiffusionDecoder { progress in
                if let file = progress.currentFile { print("  \(file)"); fflush(stdout) }
            }
            print("Diffusion decoder ready")
        }

        // Fuse LoRA if specified (after model loading, before generation)
        if let loraPath = lora {
            print("Fusing LoRA weights...")
            fflush(stdout)
            let fusedCount = try await pipeline.fuseLoRA(from: loraPath, scale: loraScale)
            print("LoRA fused (\(fusedCount) layer-pairs)")
        }

        // Auto duration: ask the duration head before committing to a frame count
        if autoDuration {
            print("Predicting duration from the prompt...")
            fflush(stdout)
            let predicted = try await pipeline.predictFrameCount(for: effectivePrompt)
            frameCount = predicted.frames
            let clampNote = predicted.wasClamped ? " (clamped from \(String(format: "%.1f", predicted.seconds))s)" : ""
            print("Duration: \(String(format: "%.2f", Float(frameCount) / 24.0))s → \(frameCount) frames\(clampNote)")

            // The duration head regresses a length from connector tokens; it never
            // sees the text, so a duration written in the prompt is dropped without
            // a word. Say so rather than substituting it: `auto` means "let the
            // model choose the natural length", and quietly overriding that with a
            // parsed number would change what the mode means.
            //
            // Checked against the prompt the user wrote, not the enhanced one — the
            // enhancer's rewrite tends to drop the duration outright.
            if let asked = PromptDuration.find(in: prompt) {
                let got = Double(frameCount - 1) / 24.0
                if abs(asked - got) > 0.5 {
                    let (wanted, _) = FrameCountSpec.frames(forSeconds: asked)
                    let want = String(format: "%.3g", asked)
                    let have = String(format: "%.3g", got)
                    print()
                    print("Note: the prompt asks for \(want)s, but --frames auto predicted \(have)s.")
                    print("      The duration head reads the scene, not written durations.")
                    print("      Pass --frames \(wanted) (or \(want)s) to get \(want)s.")
                }
            }
        }

        // Routing a dev checkpoint:
        //  - no LoRA                  → guided single-stage (the two-stage 8-step
        //    schedule is a distilled-training property);
        //  - LoRA + explicit --steps  → guided single-stage with the LoRA fused
        //    (a style/camera adapter on dev);
        //  - LoRA, no --steps         → two-stage 8 steps, ASSUMING the LoRA is
        //    the distilled one — stated out loud, since a style LoRA here would
        //    silently produce the degraded no-distillation result.
        let useDevPath = parsedModelVariant.isForTraining && (lora == nil || steps != nil)
        if useDevPath {
            print("Dev checkpoint → full-quality single-stage "
                + "(\(steps ?? parsedModelVariant.defaultSteps) steps, CFG 3.0, STG)"
                + (lora != nil ? " with the LoRA fused" : ""))
        } else if parsedModelVariant.isForTraining {
            print("Dev checkpoint + LoRA, no --steps: assuming a distilled LoRA and "
                + "running the two-stage 8-step schedule. For a style/camera LoRA, "
                + "pass --steps to run the guided single-stage path instead.")
        }
        if let steps, !useDevPath {
            throw ValidationError(
                "--steps \(steps) only applies to dev checkpoints on the single-stage path; "
                + "\(parsedModelVariant.displayName)'s two-stage schedule is fixed at 8 steps.")
        }

        // Download upscaler (needed for the two-stage path)
        print("Downloading upscaler weights (if needed)...")
        fflush(stdout)
        let upscalerPath = try await pipeline.downloadUpscalerWeights()
        print("Upscaler weights ready")

        // Build config (distilled defaults: 8 steps, no CFG, no STG)
        let config = LTXVideoGenerationConfig(
            width: width,
            height: height,
            numFrames: frameCount,
            numSteps: useDevPath ? (steps ?? parsedModelVariant.defaultSteps) : 8,
            seed: seed,
            enhancePrompt: false,   // already enhanced above, before duration prediction
            imagePath: parsedKeyframes.isEmpty ? image : nil,
            keyframes: parsedKeyframes
        )

        profilingSession?.metadata["steps"] = String(config.numSteps)

        // Generate — ONE API call
        print("\nGenerating video...")
        fflush(stdout)
        let startGen = Date()

        let result: VideoGenerationResult
        if useDevPath {
            // Dev checkpoint without a distilled LoRA: the fixed 8-step two-stage
            // schedule is a distilled-training property, so run the full-quality
            // single-stage path (CFG 3.0 + STG + rescale) instead.
            result = try await pipeline.generateVideoDev(
                prompt: effectivePrompt,
                config: config,
                onProgress: { progress in
                    print("  \(progress.status)")
                    fflush(stdout)
                })
        } else {
            result = try await pipeline.generateVideo(
                prompt: effectivePrompt,
                config: config,
                upscalerWeightsPath: upscalerPath,
                onProgress: { progress in
                    print("  \(progress.status)")
                    fflush(stdout)
                },
                keyframeSlots: keyframeSlot,
                slotsOutputPath: slotsOut
            )
        }

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

    @Option(name: .long, help: "Path to LoRA .safetensors file to apply during retake")
    var lora: String?

    @Option(name: .long, help: "LoRA scale factor (default: 1.0)")
    var loraScale: Float = 1.0

    @Option(name: .shortAndLong, help: "Output file path (default: retake.mp4)")
    var output: String = "retake.mp4"

    @Option(name: .shortAndLong, help: "Video width in pixels (must be divisible by 64)")
    var width: Int = 768

    @Option(name: .shortAndLong, help: "Video height in pixels (must be divisible by 64)")
    var height: Int = 512

    @Option(name: .shortAndLong, help: "Number of frames (8n+1, e.g. 9, 17, 25...) or a duration ('15s')")
    var frames: String = "121"

    @Option(name: .long, help: "Random seed for reproducibility")
    var seed: UInt64?

    @Flag(name: .long, help: "Enhance prompt using Gemma before generation")
    var enhancePrompt: Bool = false

    @OptionGroup var promptEnhancer: PromptEnhancerOptions

    @Flag(name: .long, help: "Use distilled model (8 steps, fast) instead of dev (30 steps + CFG)")
    var distilled: Bool = false

    @Option(name: .long, help: "Inference steps (dev model only — the distilled model runs a fixed trained 8-step schedule; default: 30)")
    var steps: Int?

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

    @Flag(name: .long, help: "Advertise activity to external monitors (writes a transient manifest in ~/Library/Application Support/ai-runtime-beacons/)")
    var beacon: Bool = false

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

        RuntimeBeacon.isEnabled = beacon

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
        if let loraPath = lora { print("LoRA: \(loraPath) (scale: \(loraScale))") }
        if transformerQuant != "bf16" { print("Transformer quantization: \(transformerQuant)") }
        print()

        // Validate
        let frameCount = try resolveFrames(frames)
        guard width % 64 == 0 && height % 64 == 0 else {
            throw ValidationError("Width and height must be divisible by 64. Got \(width)x\(height)")
        }
        guard FileManager.default.fileExists(atPath: video) else {
            throw ValidationError("Source video not found: \(video)")
        }
        if let loraPath = lora {
            guard FileManager.default.fileExists(atPath: loraPath) else {
                throw ValidationError("LoRA file not found: \(loraPath)")
            }
        }
        guard loraScale.isFinite else {
            throw ValidationError("LoRA scale must be finite. Got \(loraScale)")
        }
        guard loraScale >= 0 else {
            throw ValidationError("LoRA scale must be >= 0. Got \(loraScale)")
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
            hfToken: hfToken,
            promptEnhancer: try promptEnhancer.resolve()
        )
        print("Pipeline created (\(retakeModel.rawValue) model)")
        fflush(stdout)

        // Load models
        print("Loading models (this may take a while)...")
        fflush(stdout)
        let startLoad = Date()
        try await pipeline.loadModels(
            progressCallback: { progress in
                downloadPrinter.report(progress)
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
                downloadPrinter.report(progress)
            }
            print("Audio models loaded")
        }

        // Fuse LoRA if specified (after loading final transformer variant)
        if let loraPath = lora {
            print("Fusing LoRA weights...")
            fflush(stdout)
            let fusedCount = try await pipeline.fuseLoRA(from: loraPath, scale: loraScale)
            print("LoRA fused (\(fusedCount) layer-pairs)")
        }

        // Resolve step count: configurable on dev (proper schedule for any
        // count); the distilled model runs its fixed trained 8-step schedule
        // (issue #33 — arbitrary counts there produce artifacts).
        if steps != nil && distilled {
            throw ValidationError(
                "--steps requires the dev model: the distilled model runs a fixed " +
                "trained 8-step sigma schedule and custom counts produce artifacts.")
        }
        let numSteps = steps ?? retakeModel.defaultSteps
        if let s = steps { print("Inference steps: \(s) (dev)") }

        // Build config
        let config = LTXVideoGenerationConfig(
            width: width,
            height: height,
            numFrames: frameCount,
            numSteps: numSteps,
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

// MARK: - LipDub Command

struct LipDub: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "lipdub",
        abstract: "Lip-sync a reference video to a new prompt using the LipDub IC-LoRA"
    )

    @Argument(help: "The text prompt describing what is being said / shown")
    var prompt: String

    @Option(name: .long, help: "Reference video path (.mp4) — frames AND (default) audio are extracted from this file. Mutually exclusive with --reference-image.")
    var referenceVideo: String?

    @Option(name: .long, help: "Reference image path (.png/.jpg) used as a frame-0 I2V keyframe (same path as `generate --image`). The image anchors identity at frame 0 while the rest of the timeline animates from the prompt; the LipDub LoRA + --target-audio drive lip-sync. Requires --target-audio (the still image has no audio track).")
    var referenceImage: String?

    @Option(name: .long, help: "Target audio path (.wav/.m4a/.mp4) to lip-sync to. With --reference-video, the audio is auto-aligned (silence-aware time-stretch, pitch preserved) to the source's speech window and replaces the source audio. REQUIRED with --reference-image (used directly without alignment).")
    var targetAudio: String?

    @Option(name: .long, help: "Segment chaining (image mode): path to the PREVIOUS segment's video. Its last 9 frames are read natively — no clip preparation needed. Replaces the still image as the frame-0 anchor, preserving position AND motion across the cut. Requires --target-audio; trim 1 frame at concatenation. Mutually exclusive with --reference-video.")
    var continuationTail: String?

    @Option(name: .shortAndLong, help: "Output file path (default: lipdub.mp4)")
    var output: String = "lipdub.mp4"

    @Option(name: .shortAndLong, help: "Video width in pixels (must be divisible by 64)")
    var width: Int = 768

    @Option(name: .shortAndLong, help: "Video height in pixels (must be divisible by 64)")
    var height: Int = 512

    @Option(name: .shortAndLong, help: "Number of frames (8n+1, e.g. 121, 241) or a duration ('10s'). Should match the reference video length.")
    var frames: String = "121"

    @Option(name: .long, help: "Random seed for reproducibility")
    var seed: UInt64?

    @Option(name: .long, help: "Path to LipDub IC-LoRA .safetensors (default: auto-download from HuggingFace)")
    var lora: String?

    @Option(name: .long, help: "LipDub IC-LoRA scale (default: 1.0). Experimental — this LoRA carries the reference-token conditioning itself, not a style, so values away from 1.0 weaken lip-sync rather than soften an effect. Outside 0.5...1.5 warns; must be > 0.")
    var loraScale: Float = 1.0

    @Option(name: .long, help: "Video bitrate in kbps (e.g., 1000 for 1 Mbps). Default: quality-based encoding")
    var bitrate: Int?

    @Flag(name: .long, help: "Enable debug output")
    var debug: Bool = false

    @Flag(name: .long, help: "Advertise activity to external monitors (writes a transient manifest in ~/Library/Application Support/ai-runtime-beacons/)")
    var beacon: Bool = false

    @Flag(name: .long, help: "Enhance the prompt via the VLM (Gemma) by analyzing --reference-image. Generates a richer scene description while preserving the `speaking in <LANG> saying: \"...\"` LipDub signature. Image mode only.")
    var enhancePrompt: Bool = false

    @OptionGroup var promptEnhancer: PromptEnhancerOptions

    @Option(name: .long, help: "HuggingFace token for gated models (the IC-LoRAs and every LTX-2.5 repo are gated)")
    var hfToken: String?

    @Option(name: .long, help: "Custom directory for model storage")
    var modelsDir: String?

    @Option(name: .long, help: "Path to local Gemma model directory")
    var gemmaPath: String?

    @Option(name: .long, help: "Path to local LTX unified weights file")
    var ltxWeights: String?

    @Option(name: .long, help: "Model variant: distilled or 2.5-distilled (default: distilled)")
    var model: String = "distilled"

    mutating func run() async throws {
        if let dir = modelsDir {
            LTXModelRegistry.customModelsDirectory = URL(fileURLWithPath: dir)
        }
        if debug { LTXDebug.enableDebugMode() }

        RuntimeBeacon.isEnabled = beacon

        let variant = try parseModelVariant(model)
        print("\(variant.displayName) — LipDub")
        print("==============")
        if let tail = continuationTail {
            guard referenceVideo == nil else {
                throw ValidationError("--continuation-tail is for image-mode segment chaining; with --reference-video, segment the source video instead.")
            }
            guard FileManager.default.fileExists(atPath: tail) else {
                throw ValidationError("Continuation tail clip not found: \(tail)")
            }
            guard targetAudio != nil else {
                throw ValidationError("--continuation-tail requires --target-audio.")
            }
            print("Continuation tail: \(tail) (last latent frame anchors frame 0 — trim 1 frame at concatenation)")
        }
        switch (referenceVideo, referenceImage) {
        case (nil, nil):
            guard continuationTail != nil else {
                throw ValidationError("Must provide one of --reference-video, --reference-image or --continuation-tail.")
            }
        case (.some, .some):
            throw ValidationError("--reference-video and --reference-image are mutually exclusive.")
        case (.some(let vp), nil):
            if enhancePrompt {
                throw ValidationError("--enhance-prompt requires --reference-image (the VLM analyzes the still image); it is not supported in video-reference mode.")
            }
            print("Reference video: \(vp)")
            if let ta = targetAudio { print("Target audio: \(ta) (silence-aware auto-align enabled)") }
        case (nil, .some(let ip)):
            print("Reference image: \(ip) (single I2V keyframe at frame 0, rest animated from the prompt)")
            guard let ta = targetAudio else {
                throw ValidationError("--reference-image requires --target-audio (the still image has no audio track).")
            }
            print("Target audio: \(ta) (used directly, no alignment)")
            if enhancePrompt {
                print("Prompt enhancement: on (multimodal Gemma VLM analyzes the reference image)")
            }
        }
        print("Prompt: \(prompt)")
        print("Output: \(output)")
        print("Resolution: \(width)x\(height) (stage 1: \(width / 2)x\(height / 2))")
        print("Frames: \(frames)")
        if let seed = seed { print("Seed: \(seed)") }
        print()

        // Validate
        let frameCount = try resolveFrames(frames)
        guard width % 64 == 0 && height % 64 == 0 else {
            throw ValidationError("Width and height must be divisible by 64. Got \(width)x\(height)")
        }
        if let vp = referenceVideo {
            guard FileManager.default.fileExists(atPath: vp) else {
                throw ValidationError("Reference video not found: \(vp)")
            }
        }
        if let ip = referenceImage {
            guard FileManager.default.fileExists(atPath: ip) else {
                throw ValidationError("Reference image not found: \(ip)")
            }
        }
        if let ta = targetAudio {
            guard FileManager.default.fileExists(atPath: ta) else {
                throw ValidationError("Target audio not found: \(ta)")
            }
        }

        print("Creating pipeline (\(variant.displayName), audio enabled)...")
        fflush(stdout)
        let pipeline = LTXPipeline(
            model: variant,
            quantization: LTXQuantizationConfig(transformer: .bf16),
            hfToken: hfToken,
            promptEnhancer: try promptEnhancer.resolve()
        )
        print("Pipeline created")

        print("Loading models (this may take a while)...")
        fflush(stdout)
        let startLoad = Date()
        try await pipeline.loadModels(
            progressCallback: { progress in
                downloadPrinter.report(progress)
            },
            gemmaModelPath: gemmaPath,
            ltxWeightsPath: ltxWeights
        )

        // LipDub needs the AudioVAE encoder to encode the reference audio.
        print("Loading audio models (with encoder)...")
        fflush(stdout)
        try await pipeline.loadAudioModels(includeEncoder: true) { progress in
            downloadPrinter.report(progress)
        }
        let loadTime = Date().timeIntervalSince(startLoad)
        print("Models loaded in \(String(format: "%.1f", loadTime))s")

        // Resolve LoRA path: --lora overrides; otherwise auto-download.
        let loraPath: String
        if let provided = lora {
            guard FileManager.default.fileExists(atPath: provided) else {
                throw ValidationError("LipDub LoRA not found: \(provided)")
            }
            loraPath = provided
            print("Using LipDub LoRA: \(provided)")
        } else {
            print("Downloading LipDub IC-LoRA (gated, requires HF token)...")
            let downloader = ModelDownloader(hfToken: hfToken)
            let url = try await downloader.downloadLipDubLoRA { progress in
                downloadPrinter.report(progress)
            }
            loraPath = url.path
        }
        if loraScale != 1.0 {
            print("LipDub LoRA scale: \(loraScale) (published value is 1.0)")
        }

        // Download upscaler (always needed for two-stage)
        print("Downloading upscaler weights (if needed)...")
        let upscalerPath = try await pipeline.downloadUpscalerWeights()
        print("Upscaler weights ready")

        let config = LTXVideoGenerationConfig(
            width: width,
            height: height,
            numFrames: frameCount,
            numSteps: 8,
            seed: seed
        )

        print("\nGenerating LipDub...")
        fflush(stdout)
        let startGen = Date()
        let result = try await pipeline.generateLipDub(
            prompt: prompt,
            referenceVideoPath: referenceVideo,
            referenceImagePath: referenceImage,
            continuationTailPath: continuationTail,
            lipdubLoraPath: loraPath,
            lipdubLoRAScale: loraScale,
            config: config,
            upscalerWeightsPath: upscalerPath,
            targetAudioPath: targetAudio,
            enhancePrompt: enhancePrompt,
            onProgress: { progress in
                print("  \(progress.status)")
                fflush(stdout)
            }
        )
        let genTime = Date().timeIntervalSince(startGen)
        print("LipDub completed in \(String(format: "%.1f", genTime))s")
        fflush(stdout)

        // Export
        print("\nExporting to \(output)...")
        fflush(stdout)
        let outputURL = URL(fileURLWithPath: output)
        var exportConfig = VideoExportConfig.default
        if let kbps = bitrate {
            exportConfig.averageBitRate = kbps * 1000
        }

        guard let audioWaveform = result.audioWaveform, let audioSampleRate = result.audioSampleRate else {
            throw ValidationError("LipDub pipeline returned no audio — this should never happen")
        }
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

        let wavPath = output.replacingOccurrences(of: ".mp4", with: ".wav")
        try AudioExporter.exportToWAV(
            waveform: audioWaveform,
            sampleRate: audioSampleRate,
            path: wavPath
        )
        print("Audio WAV saved to: \(wavPath)")

        print("\n--- Summary ---")
        print("Seed: \(result.seed)")
        print("Generation time: \(String(format: "%.1f", result.generationTime))s")
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
                downloadPrinter.report(progress)
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
        abstract: "Show information about this LTX implementation"
    )

    mutating func run() throws {
        print(
            """
            LTX-2 Video Generation for Apple Silicon (LTX-2.3 + LTX-2.5)
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

              # LipDub: lip-sync a reference video to a new prompt
              ltx-video lipdub "a person speaking the dialogue" \
                --reference-video source.mp4 -w 768 -h 512 -f 121

              # LipDub from a still image (image replicated to F frames; out-of-distribution)
              ltx-video lipdub "a man speaking in English saying: \"Hello world\"" \
                --reference-image portrait.png --target-audio tts.wav -w 768 -h 512 -f 121

              # With prompt enhancement
              ltx-video generate "A beaver" -w 768 -h 512 -f 121 --enhance-prompt

              # With quantization (lower memory)
              ltx-video generate "A sunset" -w 768 -h 512 -f 121 --transformer-quant qint8
            """)
    }
}
