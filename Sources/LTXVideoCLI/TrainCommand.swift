// TrainCommand.swift - CLI command for LoRA training
// Copyright 2026

import ArgumentParser
import Foundation
import LTXVideo

struct Train: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Train a LoRA adapter on a dataset of videos + captions"
    )

    @Argument(help: "Path to dataset directory (video.mp4 + video.txt pairs)")
    var dataset: String

    @Option(name: .shortAndLong, help: "Output directory for LoRA weights and checkpoints")
    var output: String = "./lora-output"

    @Option(name: .shortAndLong, help: "Model variant: dev (required for training) or distilled")
    var model: String = "dev"

    @Option(name: .long, help: "LoRA rank (default: 16)")
    var rank: Int = 16

    @Option(name: .long, help: "LoRA alpha (default: same as rank)")
    var alpha: Float?

    @Option(name: .long, help: "Learning rate (default: 2e-4)")
    var lr: Float = 2e-4

    @Option(name: .long, help: "Weight decay (default: 0.01)")
    var weightDecay: Float = 0.01

    @Option(name: .long, help: "Maximum training steps (default: 2000)")
    var steps: Int = 2000

    @Option(name: .long, help: "Save checkpoint every N steps (default: 250)")
    var saveEvery: Int = 250

    @Option(name: .shortAndLong, help: "Video width (must be divisible by 32)")
    var width: Int = 256

    @Option(name: .shortAndLong, help: "Video height (must be divisible by 32)")
    var height: Int = 256

    @Option(name: .shortAndLong, help: "Number of frames (must be 8n+1)")
    var frames: Int = 9

    @Option(name: .long, help: "Transformer quantization: bf16 (default), qint8, int4")
    var transformerQuant: String = "bf16"

    @Option(name: .long, help: "Number of last transformer blocks to apply LoRA to (0 = all, default)")
    var loraBlocks: Int = 0

    @Option(name: .long, help: "Gradient accumulation steps (default: 1)")
    var gradAccum: Int = 1

    @Option(name: .long, help: "Max gradient norm for clipping (default: 1.0, 0=disabled)")
    var maxGradNorm: Float = 1.0

    @Option(name: .long, help: "LR warmup steps (default: 100)")
    var warmupSteps: Int = 100

    @Option(name: .long, help: "Trigger word to prepend to all captions (e.g., CAKEIFY)")
    var triggerWord: String?

    @Option(name: .long, help: "Random seed")
    var seed: UInt64?

    @Option(name: .long, help: "HuggingFace token")
    var hfToken: String?

    @Option(name: .long, help: "Custom models directory")
    var modelsDir: String?

    @Option(name: .long, help: "Path to local Gemma3 model directory")
    var gemmaPath: String?

    @Option(name: .long, help: "Path to unified LTX-2 weights file")
    var ltxWeights: String?

    @Flag(name: .long, help: "Train on audio+video (not yet supported)")
    var audio: Bool = false

    @Option(name: .long, help: "Audio loss weight (default: 0.5)")
    var audioLossWeight: Float = 0.5

    @Flag(name: .long, inversion: .prefixedNo, help: "Include FFN layers in LoRA targets (default: no)")
    var includeFFN: Bool = false

    @Option(name: .long, help: "Memory preset: compact (32GB), balanced (64GB), quality (96GB), max (192GB)")
    var preset: String?

    @Flag(name: .long, help: "Resume from latest checkpoint in output directory")
    var resume: Bool = false

    @Flag(name: .long, help: "Advertise activity to external monitors (writes a transient manifest in ~/Library/Application Support/ai-runtime-beacons/)")
    var beacon: Bool = false

    @Option(name: .long, help: "Resume from a specific checkpoint step")
    var resumeStep: Int?

    @Option(
        name: .long,
        help: "Generate a preview video at each checkpoint using this prompt. Runs the distilled model as a subprocess (fast, 8 steps). Omit to disable preview generation."
    )
    var previewPrompt: String?

    @Option(
        name: .long,
        help: "Input image for I2V preview generation at checkpoints (used with --preview-prompt)."
    )
    var previewImage: String?

    mutating func run() async throws {
        print("LTX-2 LoRA Training")
        print("===================")

        RuntimeBeacon.isEnabled = beacon

        // Guard audio flag
        if audio {
            print("Error: Audio LoRA training is not yet supported.")
            print("Audio latent caching will be added in a future version.")
            throw ExitCode.failure
        }

        // Apply preset if specified
        var config: LoRATrainingConfig
        if let presetName = preset {
            switch presetName {
            case "compact": config = .compact
            case "balanced": config = .balanced
            case "quality": config = .quality
            case "max": config = .max
            default:
                throw ValidationError("Unknown preset: \(presetName). Use: compact, balanced, quality, max")
            }
            print("Preset: \(presetName)")
        } else {
            config = LoRATrainingConfig()
        }

        // Override with explicit flags
        config.rank = rank
        config.alpha = alpha ?? Float(rank)
        config.learningRate = lr
        config.weightDecay = weightDecay
        config.maxSteps = steps
        config.saveEvery = saveEvery
        config.width = width
        config.height = height
        config.numFrames = frames
        config.model = model
        config.includeAudio = audio
        config.audioLossWeight = audioLossWeight
        config.includeFFN = includeFFN
        config.transformerQuant = transformerQuant
        config.loraBlocks = loraBlocks
        config.gradientAccumulationSteps = gradAccum
        config.maxGradNorm = maxGradNorm
        config.warmupSteps = warmupSteps
        config.triggerWord = triggerWord
        config.seed = seed
        config.hfToken = hfToken
        config.modelsDir = modelsDir
        config.gemmaPath = gemmaPath
        config.ltxWeightsPath = ltxWeights
        config.previewPrompt = previewPrompt
        config.previewImage = previewImage

        // Print training plan summary
        print()
        print("Training Plan:")
        print("  Dataset: \(dataset)")
        print("  Output: \(output)")
        print("  Model: \(model)")
        print("  Resolution: \(config.width)x\(config.height), Frames: \(config.numFrames)")
        print("  Rank: \(config.rank), Alpha: \(config.alpha), LR: \(config.learningRate)")
        print("  Steps: \(config.maxSteps), Save every: \(config.saveEvery)")
        print("  Quant: \(config.transformerQuant)")
        print("  Targets: attention\(config.includeFFN ? " + FFN" : " only")")
        if config.loraBlocks > 0 {
            print("  LoRA blocks: last \(config.loraBlocks) of 48")
        }
        print("  Grad accumulation: \(config.gradientAccumulationSteps)")
        print("  Grad clip norm: \(config.maxGradNorm > 0 ? String(format: "%.1f", config.maxGradNorm) : "disabled")")
        if let trigger = config.triggerWord {
            print("  Trigger word: \(trigger)")
        }
        if let preview = config.previewPrompt {
            print("  Preview prompt: \(preview)")
            if let img = config.previewImage {
                print("  Preview image: \(img)")
            }
        }
        print()

        let trainer = LoRATrainer(
            config: config,
            datasetPath: dataset,
            outputDir: output
        )

        // Set up training controller for pause/resume/stop
        let controller = TrainingController(outputDir: output)
        trainer.controller = controller

        // Handle Ctrl+C for graceful stop
        signal(SIGINT) { _ in
            print("\n  Ctrl+C received, stopping gracefully...")
            // Use file-based signaling since we can't capture the controller
            let outputDir = CommandLine.arguments.dropFirst().first { !$0.hasPrefix("-") } ?? "."
            TrainingController.stopTraining(outputDir: outputDir)
        }

        // Determine resume behavior
        let resumeFromStep: Int?
        if let step = resumeStep {
            resumeFromStep = step
        } else if resume {
            resumeFromStep = nil  // auto-detect latest
        } else {
            resumeFromStep = 0  // force fresh start
        }

        try await trainer.train(resumeFromStep: resumeFromStep) { progress in
            if progress.step % 10 == 0 || progress.step == 1 {
                print(progress.status)
                fflush(stdout)
            }
        }
    }
}
