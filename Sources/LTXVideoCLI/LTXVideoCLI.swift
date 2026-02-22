// LTXVideoCLI.swift - Command-line interface for LTX-2 video generation
// Copyright 2025

import ArgumentParser
import Foundation
import LTXVideo

@main
struct LTXVideoCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "ltx-video",
        abstract: "LTX-2 video generation on Mac with MLX",
        version: "0.1.0",
        subcommands: [Generate.self, Download.self, Info.self],
        defaultSubcommand: Info.self
    )
}

// MARK: - Generate Command

struct Generate: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Generate a video from a text prompt"
    )

    @Argument(help: "The text prompt describing the video to generate")
    var prompt: String

    @Option(name: .shortAndLong, help: "Output file path (default: output.mp4)")
    var output: String = "output.mp4"

    @Option(name: .shortAndLong, help: "Video width in pixels (must be divisible by 32)")
    var width: Int = 512

    @Option(name: .shortAndLong, help: "Video height in pixels (must be divisible by 32)")
    var height: Int = 512

    @Option(name: .shortAndLong, help: "Number of frames (must be 8n+1, e.g., 9, 17, 25, 33...)")
    var frames: Int = 25

    @Option(name: .shortAndLong, help: "Number of inference steps (default: 8 for distilled)")
    var steps: Int?

    @Option(name: .shortAndLong, help: "CFG guidance scale (1.0 = no CFG, default for distilled)")
    var guidance: Float?

    @Option(name: .long, help: "Random seed for reproducibility")
    var seed: UInt64?

    @Option(name: .shortAndLong, help: "Model variant: distilled or dev")
    var model: String = "distilled"

    @Option(name: .long, help: "Path to LoRA weights (.safetensors)")
    var lora: String?

    @Option(name: .long, help: "LoRA scale factor")
    var loraScale: Float = 1.0

    @Option(name: .long, help: "HuggingFace token for gated models")
    var hfToken: String?

    @Option(name: .long, help: "Path to local Gemma3 model directory")
    var gemmaPath: String?

    @Option(name: .long, help: "Path to unified LTX-2 weights file (.safetensors)")
    var ltxWeights: String?

    @Option(name: .long, help: "Negative prompt for CFG (default: detailed quality-negative prompt)")
    var negativePrompt: String?

    @Option(name: .long, help: "Guidance rescale (phi). 0.0=off, 0.7=recommended with CFG")
    var guidanceRescale: Float = 0.0

    @Option(name: .long, help: "Cross-attention scale. 1.0=default, >1=stronger prompt adherence")
    var crossAttnScale: Float = 1.0

    @Option(name: .long, help: "GE velocity correction gamma. 0.0=off")
    var geGamma: Float = 0.0

    @Option(name: .long, help: "STG (Spatio-Temporal Guidance) scale. 0.0=off, 0.5=recommended")
    var stgScale: Float = 0.0

    @Option(name: .long, help: "STG block indices (comma-separated, e.g. \"29\" or \"28,29\")")
    var stgBlocks: String = "29"

    @Option(name: .long, help: "Transformer quantization: bf16 (default), qint8 (8-bit), int4 (4-bit)")
    var transformerQuant: String = "bf16"

    @Flag(name: .long, help: "Use two-stage generation: half resolution then upscale 2x and refine")
    var twoStage: Bool = false

    @Flag(name: .long, help: "Apply distilled LoRA (auto-downloads, forces dev model, 8 steps, no CFG)")
    var distilledLora: Bool = false

    @Flag(name: .long, help: "Enhance prompt using Gemma before generation")
    var enhancePrompt: Bool = false

    @Flag(name: .long, help: "Enable debug output")
    var debug: Bool = false

    @Flag(name: .long, help: "Enable performance profiling")
    var profile: Bool = false

    @Flag(name: .long, help: "Skip model loading (dry run)")
    var dryRun: Bool = false

    mutating func run() async throws {
        // Enable debug mode if requested
        if debug {
            LTXDebug.enableDebugMode()
        }

        // Parse STG blocks
        let parsedStgBlocks = stgBlocks.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }

        print("LTX-2 Video Generation")
        print("======================")
        print("Prompt: \(prompt)")
        print("Output: \(output)")
        print("Resolution: \(width)x\(height)")
        print("Frames: \(frames)")
        print("Model: \(distilledLora ? "dev" : model)")
        if distilledLora { print("Distilled LoRA: enabled") }
        if let seed = seed {
            print("Seed: \(seed)")
        }
        if guidanceRescale > 0 { print("Guidance rescale: \(guidanceRescale)") }
        if crossAttnScale != 1.0 { print("Cross-attention scale: \(crossAttnScale)") }
        if geGamma > 0 { print("GE gamma: \(geGamma)") }
        if stgScale > 0 { print("STG scale: \(stgScale), blocks: \(parsedStgBlocks)") }
        if twoStage { print("Two-stage: enabled") }
        if enhancePrompt { print("Prompt enhancement: enabled") }
        if transformerQuant != "bf16" { print("Transformer quantization: \(transformerQuant)") }
        print()

        // Validate frame count (must be 8n+1)
        guard (frames - 1) % 8 == 0 else {
            throw ValidationError("Frame count must be 8n+1 (e.g., 9, 17, 25, 33, ...). Got \(frames)")
        }

        // Validate dimensions (must be divisible by 32)
        guard width % 32 == 0 && height % 32 == 0 else {
            throw ValidationError("Width and height must be divisible by 32. Got \(width)x\(height)")
        }

        // Parse transformer quantization
        guard let quantOption = TransformerQuantization(rawValue: transformerQuant) else {
            throw ValidationError("Invalid transformer quantization: \(transformerQuant). Use: bf16, qint8, or int4")
        }
        let quantConfig = LTXQuantizationConfig(transformer: quantOption)

        // Parse model variant
        // --distilled-lora forces dev model
        let effectiveModel = distilledLora ? "dev" : model
        guard let modelVariant = LTXModel(rawValue: effectiveModel) else {
            throw ValidationError("Invalid model: \(model). Use: distilled or dev")
        }

        if twoStage {
            // Two-stage validation: width/height must be divisible by 64
            guard width % 64 == 0 && height % 64 == 0 else {
                throw ValidationError("Two-stage requires width and height divisible by 64. Got \(width)x\(height)")
            }
            let stageDesc = distilledLora || (effectiveModel == "dev" && lora == nil) ? "dev + distilled LoRA" : effectiveModel
            print("Two-stage pipeline: \(width/2)x\(height/2) -> upscale 2x -> \(width)x\(height)")
            print("  Base model: \(stageDesc)")
        }

        if distilledLora {
            print("Distilled LoRA: will fuse into dev model (8 steps, no CFG)")
        }

        if dryRun {
            print("Validation passed (dry run mode)")
            return
        }

        // Create pipeline
        print("Creating pipeline...")
        fflush(stdout)
        let pipeline = LTXPipeline(
            model: modelVariant,
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

        // Apply LoRA
        // --distilled-lora: always download and fuse distilled LoRA
        // --two-stage without --distilled-lora: no auto LoRA (user controls model/steps/CFG)
        let needsDistilledLoRA = distilledLora
        if needsDistilledLoRA {
            // Download and fuse distilled LoRA
            print("Downloading distilled LoRA (if needed)...")
            fflush(stdout)
            let loraPath = try await pipeline.downloadDistilledLoRA()
            print("Fusing distilled LoRA into transformer...")
            fflush(stdout)
            let fusedCount = try await pipeline.fuseLoRA(from: loraPath, scale: loraScale)
            print("  Fused \(fusedCount) layers (scale=\(loraScale))")
        } else if let loraPath = lora {
            // Custom LoRA specified
            print("Applying LoRA from \(loraPath)...")
            let result = try await pipeline.applyLoRA(from: loraPath, scale: loraScale)
            print("  Modified \(result.modifiedLayerCount) layers")
        }

        // Generate video
        print("\nGenerating video...")
        let startGen = Date()

        // Determine steps and CFG:
        // --distilled-lora: default to 8 steps, no CFG (can be overridden with --steps/--guidance)
        // Otherwise: use model defaults or explicit overrides
        let effectiveSteps: Int
        let effectiveCFG: Float
        if distilledLora {
            effectiveSteps = steps ?? 8
            effectiveCFG = guidance ?? 1.0
        } else {
            effectiveSteps = steps ?? modelVariant.defaultSteps
            effectiveCFG = guidance ?? modelVariant.defaultGuidance
        }

        let config = LTXVideoGenerationConfig(
            width: width,
            height: height,
            numFrames: frames,
            numSteps: effectiveSteps,
            cfgScale: effectiveCFG,
            seed: seed,
            guidanceRescale: guidanceRescale,
            crossAttentionScale: crossAttnScale,
            geGamma: geGamma,
            stgScale: stgScale,
            stgBlocks: parsedStgBlocks,
            twoStage: twoStage,
            enhancePrompt: enhancePrompt
        )

        let result: VideoGenerationResult
        if twoStage {
            // Two-stage: half-res -> upscale 2x -> refine at full-res
            print("Downloading upscaler weights (if needed)...")
            fflush(stdout)
            let upscalerPath = try await pipeline.downloadUpscalerWeights()
            print("Upscaler weights ready")

            result = try await pipeline.generateVideoTwoStage(
                prompt: prompt,
                config: config,
                upscalerWeightsPath: upscalerPath,
                onProgress: { progress in
                    print("  \(progress.status)")
                },
                profile: profile
            )
        } else {
            result = try await pipeline.generateVideo(
                prompt: prompt,
                negativePrompt: negativePrompt,
                config: config,
                onProgress: { progress in
                    print("  \(progress.status)")
                },
                profile: profile
            )
        }

        let genTime = Date().timeIntervalSince(startGen)
        print("Generation completed in \(String(format: "%.1f", genTime))s")

        // Export video
        print("\nExporting to \(output)...")
        let outputURL = URL(fileURLWithPath: output)

        let videoURL = try await VideoExporter.exportVideo(
            frames: result.frames,
            width: width,
            height: height,
            fps: 24.0,
            to: outputURL
        )
        print("Video saved to: \(videoURL.path)")

        // Print summary
        print("\n--- Summary ---")
        print("Frames: \(result.numFrames)")
        print("Resolution: \(result.width)x\(result.height)")
        print("Seed: \(result.seed)")
        print("Generation time: \(String(format: "%.1f", result.generationTime))s")

        // Print detailed profiling if enabled
        if profile, let t = result.timings {
            let f = { (v: Double) -> String in String(format: "%.1f", v) }
            print("\n--- Profiling ---")
            print("Text Encoding (Gemma + FE + Connector): \(f(t.textEncoding))s")
            print("Denoising (\(t.denoiseSteps.count) steps):                 \(f(t.totalDenoise))s")
            for (i, stepTime) in t.denoiseSteps.enumerated() {
                print("  Step \(i): \(f(stepTime))s")
            }
            print("  Average per step:                      \(f(t.avgStepTime))s")
            print("VAE Decoding:                            \(f(t.vaeDecode))s")
            print("Model Loading:                           \(f(loadTime))s")
            let pipelineTotal = t.textEncoding + t.totalDenoise + t.vaeDecode
            print("Pipeline total (excl. loading/export):   \(f(pipelineTotal))s")
            print("\n--- Memory ---")
            print("Peak GPU memory:                         \(t.peakMemoryMB) MB")
            print("Mean GPU memory (denoising):              \(t.meanMemoryMB) MB")
        }
    }
}

// MARK: - Download Command

struct Download: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Download model weights from HuggingFace"
    )

    @Option(name: .shortAndLong, help: "Model variant: distilled or dev")
    var model: String = "distilled"

    @Option(name: .long, help: "HuggingFace token for gated models")
    var hfToken: String?

    @Flag(name: .long, help: "Force re-download even if files exist")
    var force: Bool = false

    mutating func run() async throws {
        print("LTX-2 Model Download")
        print("====================")

        // Parse model variant
        guard let modelVariant = LTXModel(rawValue: model) else {
            throw ValidationError("Invalid model: \(model). Use: distilled or dev")
        }

        print("Model: \(modelVariant.displayName)")
        print("Repository: \(modelVariant.huggingFaceRepo)")
        print("Estimated RAM: ~\(modelVariant.estimatedVRAM)GB")
        print()

        let downloader = ModelDownloader(hfToken: hfToken)

        // Download all components (Diffusers per-component format)
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
        print("Connector: \(paths.connectorPath.path)")
        print("VAE: \(paths.vaePath.path)")
        print("Unified weights: \(paths.unifiedWeightsPath.path)")
    }
}

// MARK: - Info Command

struct Info: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Show information about LTX-2 implementation"
    )

    mutating func run() throws {
        print(
            """
            LTX-2 Video Generation for Apple Silicon
            =========================================

            Version: \(LTXVideo.version)
            Platform: macOS (Apple Silicon with MLX)

            Model Variants:
              distilled     - ~16GB RAM, 8 steps
              dev           - ~25GB RAM, 40 steps (best quality)

            Constraints:
              Frame count: Must be 8n+1 (9, 17, 25, 33, 41, 49, ...)
              Resolution: Width and height must be divisible by 32
              Recommended: 512x512, 768x512, 832x480, 1024x576

            Usage:
              ltx-video generate "A cat walking" --output cat.mp4
              ltx-video generate "Ocean sunset" --two-stage -w 768 -h 512 -f 241
              ltx-video download --model distilled
              ltx-video info

            Examples:
              # Quick test (small video)
              ltx-video generate "A red ball bouncing" -w 256 -h 256 -f 9

              # Standard quality
              ltx-video generate "Ocean waves at sunset" -w 512 -h 512 -f 25

              # High quality with two-stage pipeline (10 seconds)
              ltx-video generate "A forest with falling leaves" -w 768 -h 512 -f 241 --two-stage
            """)
    }
}
