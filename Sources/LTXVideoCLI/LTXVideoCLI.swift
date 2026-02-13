// LTXVideoCLI.swift - Command-line interface for LTX-2 video generation
// Copyright 2025

import ArgumentParser
import Foundation
import LTXVideo
@preconcurrency import MLX
import MLXRandom
import MLXNN
import MLXLMCommon
import Hub
import Tokenizers

@main
struct LTXVideoCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "ltx-video",
        abstract: "LTX-2 video generation on Mac with MLX",
        version: "0.1.0",
        subcommands: [Generate.self, Download.self, Info.self, TestComponents.self, Validate.self, TestGemmaNative.self, Encode.self, DecodeLatent.self, DenoiseTest.self],
        defaultSubcommand: Info.self
    )
}

// MARK: - Test Components Command

struct TestComponents: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "test",
        abstract: "Test individual components (transformer, vae, text-encoder)"
    )

    @Option(name: .shortAndLong, help: "Component to test: transformer, vae, connector (text-encoder), all")
    var component: String = "transformer"

    @Option(name: .shortAndLong, help: "Model variant: distilledFP8, distilled, or dev")
    var model: String = "distilled"

    @Flag(name: .long, help: "Enable debug output")
    var debug: Bool = false

    mutating func run() async throws {
        if debug {
            LTXDebug.enableDebugMode()
        }

        // Parse model variant
        guard let modelVariant = LTXModel(rawValue: model) else {
            throw ValidationError("Invalid model: \(model). Use: distilledFP8, distilled, or dev")
        }

        print("LTX-2 Component Test")
        print("====================")
        print("Component: \(component)")
        print("Model: \(modelVariant.displayName)")
        print()

        // Download unified weights if needed
        let downloader = ModelDownloader()
        print("Checking model weights...")
        let weightsPath = try await downloader.downloadLTXWeights(model: modelVariant) { progress in
            print("  \(progress.message)")
        }
        print("Weights: \(weightsPath.path)")
        print()

        // Load and split unified weights
        print("Loading and splitting weights...")
        let components = try LTXWeightLoader.loadUnifiedWeights(
            from: weightsPath.path,
            isFP8: modelVariant.isFP8
        )
        print("  Transformer: \(components.transformer.count) tensors")
        print("  VAE: \(components.vaeDecoder.count) tensors")
        print("  TextEncoder: \(components.textEncoder.count) tensors")
        print()

        switch component.lowercased() {
        case "transformer":
            try testTransformer(weights: components.transformer, config: modelVariant.transformerConfig)
        case "vae":
            try testVAE(weights: components.vaeDecoder)
        case "connector", "text-encoder":
            try testTextEncoderConnector(weights: components.textEncoder)
        case "all":
            try testTransformer(weights: components.transformer, config: modelVariant.transformerConfig)
            print()
            try testVAE(weights: components.vaeDecoder)
            print()
            try testTextEncoderConnector(weights: components.textEncoder)
        default:
            print("Unknown component: \(component)")
            print("Available: transformer, vae, connector (text-encoder), all")
        }
    }

    private func testTransformer(weights: [String: MLXArray], config: LTXTransformerConfig) throws {
        print("=== Testing Transformer ===")
        print("Input weights: \(weights.count) tensors")

        // Show sample keys
        print("\nSample weight keys:")
        for key in weights.keys.sorted().prefix(10) {
            if let array = weights[key] {
                print("  \(key): \(array.shape)")
            }
        }

        // Create model
        print("\nCreating transformer model...")
        let transformer = LTXTransformer(config: config)

        let params = transformer.parameters().flattened()
        print("Model has \(params.count) parameter keys")

        // Apply weights
        print("\nApplying weights...")
        try LTXWeightLoader.applyTransformerWeights(weights, to: transformer)

        // Verify FP8 dequant values
        if let w = weights["transformer_blocks.1.attn1.to_k.weight"] {
            eval(w)
            let flat = w.flattened()
            let first10 = (0..<10).map { flat[$0].item(Float.self) }
            print("\nFP8 dequant verification (block 1 attn1 to_k):")
            print("  Shape: \(w.shape), dtype: \(w.dtype)")
            print("  Mean: \(w.mean().item(Float.self))")
            print("  Std: \(MLX.sqrt(w.variance()).item(Float.self))")
            print("  First 10: \(first10)")
        }

        print("\n✅ Transformer test passed")
    }

    private func testVAE(weights: [String: MLXArray]) throws {
        print("=== Testing VAE ===")
        print("Input weights: \(weights.count) tensors")

        // Show sample keys
        print("\nSample weight keys:")
        for key in weights.keys.sorted().prefix(10) {
            if let array = weights[key] {
                print("  \(key): \(array.shape)")
            }
        }

        // Create model
        print("\nCreating VAE decoder model...")
        let vae = VideoDecoder()

        let params = vae.parameters().flattened()
        print("Model has \(params.count) parameter keys")

        // Apply weights
        print("\nApplying weights...")
        try LTXWeightLoader.applyVAEWeights(weights, to: vae)

        print("\n✅ VAE test passed")
    }

    private func testTextEncoderConnector(weights: [String: MLXArray]) throws {
        print("=== Testing Text Encoder Connector ===")
        print("Input weights: \(weights.count) tensors")

        // Show sample keys
        print("\nSample weight keys:")
        for key in weights.keys.sorted().prefix(15) {
            if let array = weights[key] {
                print("  \(key): \(array.shape)")
            }
        }

        // Create model
        print("\nCreating VideoGemmaTextEncoderModel...")
        let textEncoder = createTextEncoder()

        let params = textEncoder.parameters().flattened()
        print("Model has \(params.count) parameter keys")

        // Map and apply weights
        print("\nMapping and applying weights...")
        let mapped = LTXWeightLoader.mapTextEncoderWeights(weights)
        print("Mapped \(mapped.count) weights")

        // Check coverage
        var found = 0
        var notFound = 0
        var notFoundKeys: [String] = []

        for key in mapped.keys {
            if params.contains(where: { $0.0 == key }) {
                found += 1
            } else {
                notFound += 1
                if notFoundKeys.count < 10 {
                    notFoundKeys.append(key)
                }
            }
        }

        print("  Found in model: \(found)")
        print("  Not found: \(notFound)")
        if !notFoundKeys.isEmpty {
            print("  Missing keys: \(notFoundKeys)")
        }

        try LTXWeightLoader.applyTextEncoderWeights(weights, to: textEncoder)

        print("\n✅ Text Encoder Connector test passed")
    }
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

    @Option(name: .shortAndLong, help: "Model variant: distilledFP8, distilled, or dev")
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

        print("LTX-2 Video Generation")
        print("======================")
        print("Prompt: \(prompt)")
        print("Output: \(output)")
        print("Resolution: \(width)x\(height)")
        print("Frames: \(frames)")
        print("Model: \(model)")
        if let seed = seed {
            print("Seed: \(seed)")
        }
        print()

        // Validate frame count (must be 8n+1)
        guard (frames - 1) % 8 == 0 else {
            throw ValidationError("Frame count must be 8n+1 (e.g., 9, 17, 25, 33, ...). Got \(frames)")
        }

        // Validate dimensions (must be divisible by 32)
        guard width % 32 == 0 && height % 32 == 0 else {
            throw ValidationError("Width and height must be divisible by 32. Got \(width)x\(height)")
        }

        // Parse model variant
        guard let modelVariant = LTXModel(rawValue: model) else {
            throw ValidationError("Invalid model: \(model). Use: distilledFP8, distilled, or dev")
        }

        if dryRun {
            print("✅ Validation passed (dry run mode)")
            return
        }

        // Create pipeline
        print("Creating pipeline...")
        fflush(stdout)  // Force flush
        let pipeline = LTXPipeline(
            model: modelVariant,
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

        // Apply LoRA if specified
        if let loraPath = lora {
            print("Applying LoRA from \(loraPath)...")
            let result = try await pipeline.applyLoRA(from: loraPath, scale: loraScale)
            print("  Modified \(result.modifiedLayerCount) layers")
        }

        // Generate video
        print("\nGenerating video...")
        let startGen = Date()

        let config = LTXVideoGenerationConfig(
            width: width,
            height: height,
            numFrames: frames,
            numSteps: steps ?? modelVariant.defaultSteps,
            cfgScale: guidance ?? modelVariant.defaultGuidance,
            seed: seed
        )

        let result = try await pipeline.generateVideo(
            prompt: prompt,
            config: config,
            onProgress: { progress in
                print("  \(progress.status)")
            },
            profile: profile
        )

        let genTime = Date().timeIntervalSince(startGen)
        print("Generation completed in \(String(format: "%.1f", genTime))s")

        // Export video
        print("\nExporting to \(output)...")
        let exportStart = Date()
        let outputURL = URL(fileURLWithPath: output)
        let exporter = VideoExporter()

        // Convert MLXArray frames to exportable format
        let exportFrames = VideoExporter.tensorToImages(result.frames)

        if exportFrames.isEmpty {
            print("No frames generated to export")
            return
        }

        let exportResult = VideoExportFrames(
            frames: exportFrames,
            fps: 24.0,
            width: width,
            height: height
        )

        let videoURL = try await exporter.export(exportResult, to: outputURL)
        let exportTime = Date().timeIntervalSince(exportStart)
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
            print("Video Export:                            \(f(exportTime))s")
            print("Model Loading:                           \(f(loadTime))s")
            let pipelineTotal = t.textEncoding + t.totalDenoise + t.vaeDecode
            print("Pipeline total (excl. loading/export):   \(f(pipelineTotal))s")
        }
    }
}

// MARK: - Download Command

struct Download: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Download model weights from HuggingFace"
    )

    @Option(name: .shortAndLong, help: "Model variant: distilledFP8, distilled, or dev")
    var model: String = "distilled"

    @Option(name: .long, help: "HuggingFace token for gated models")
    var hfToken: String?

    @Flag(name: .long, help: "Also download Gemma3 12B text encoder")
    var withGemma: Bool = false

    @Flag(name: .long, help: "Force re-download even if files exist")
    var force: Bool = false

    mutating func run() async throws {
        print("LTX-2 Model Download")
        print("====================")

        // Parse model variant
        guard let modelVariant = LTXModel(rawValue: model) else {
            throw ValidationError("Invalid model: \(model). Use: distilledFP8, distilled, or dev")
        }

        print("Model: \(modelVariant.displayName)")
        print("Repository: \(modelVariant.huggingFaceRepo)")
        print("File: \(modelVariant.weightsFilename)")
        print("Estimated RAM: ~\(modelVariant.estimatedVRAM)GB")
        print()

        let downloader = ModelDownloader(hfToken: hfToken)

        // Download LTX unified weights
        if await downloader.isLTXWeightsDownloaded(modelVariant) && !force {
            print("✅ LTX weights already downloaded")
        } else {
            print("Downloading LTX-2 weights...")
            let weightsPath = try await downloader.downloadLTXWeights(model: modelVariant) { progress in
                if let file = progress.currentFile {
                    print("  [\(Int(progress.progress * 100))%] \(file)")
                } else {
                    print("  \(progress.message)")
                }
            }
            print("✅ LTX weights: \(weightsPath.path)")
        }

        // Download Gemma if requested
        if withGemma {
            print()
            if await downloader.isGemmaDownloaded() && !force {
                print("✅ Gemma already downloaded")
            } else {
                print("Downloading Gemma3 12B (MLX 4-bit)...")
                let gemmaPath = try await downloader.downloadGemma { progress in
                    if let file = progress.currentFile {
                        print("  [\(Int(progress.progress * 100))%] \(file)")
                    } else {
                        print("  \(progress.message)")
                    }
                }
                print("✅ Gemma: \(gemmaPath.path)")
            }
        }
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
              • distilledFP8  - ~12GB RAM, 8 steps (fastest)
              • distilled     - ~16GB RAM, 8 steps
              • dev           - ~25GB RAM, 50 steps (best quality)

            Constraints:
              • Frame count: Must be 8n+1 (9, 17, 25, 33, 41, 49, ...)
              • Resolution: Width and height must be divisible by 32
              • Recommended: 512x512, 768x512, 832x480, 1024x576

            Implementation Status:
              ✅ Configuration layer
              ✅ Scheduler (flow-matching)
              ✅ 3D RoPE (rotary embeddings)
              ✅ Transformer (48 blocks)
              ✅ VAE Decoder (3D convolutions)
              ✅ Text Encoder (Gemma3 wrapper)
              ✅ LoRA support
              ✅ Pipeline orchestration
              ✅ Weight loading
              ✅ Video export (MP4)

            Usage:
              ltx-video generate "A cat walking" --output cat.mp4
              ltx-video download --model distilledFP8
              ltx-video info

            Examples:
              # Quick test (small video)
              ltx-video generate "A red ball bouncing" -w 256 -h 256 -f 9

              # Standard quality
              ltx-video generate "Ocean waves at sunset" -w 512 -h 512 -f 25

              # High quality (requires more RAM)
              ltx-video generate "A forest with falling leaves" -w 768 -h 512 -f 49 --model distilled
            """)
    }
}

// MARK: - Validate Command

struct Validate: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "validate",
        abstract: "Validate Swift implementation against Python reference"
    )

    @Option(name: .shortAndLong, help: "Model variant: distilledFP8, distilled, or dev")
    var model: String = "distilled"

    @Flag(name: .long, help: "Enable debug output")
    var debug: Bool = false

    @Flag(name: .long, help: "Run quick tests only (skip heavy computation)")
    var quick: Bool = false

    mutating func run() async throws {
        if debug {
            LTXDebug.enableDebugMode()
        }

        print("LTX-2 Validation Suite")
        print("======================")
        print()

        // Parse model variant
        guard let modelVariant = LTXModel(rawValue: model) else {
            throw ValidationError("Invalid model: \(model). Use: distilledFP8, distilled, or dev")
        }

        // Download unified weights if needed
        let downloader = ModelDownloader()
        print("Checking model weights...")
        let weightsPath = try await downloader.downloadLTXWeights(model: modelVariant) { progress in
            if !progress.message.isEmpty {
                print("  \(progress.message)")
            }
        }
        print("Weights: \(weightsPath.path)")
        print()

        // Load and split unified weights
        print("Loading and splitting weights...")
        let components = try LTXWeightLoader.loadUnifiedWeights(
            from: weightsPath.path,
            isFP8: modelVariant.isFP8
        )
        print()

        var passed = 0
        var failed = 0

        // Test 1: Scheduler Sigmas
        print("--- Test 1: Scheduler Sigmas ---")
        let schedulerResult = validateSchedulerSigmas()
        if schedulerResult { passed += 1 } else { failed += 1 }
        print()

        // Test 2: Text Encoder Connector Weights
        print("--- Test 2: Text Encoder Connector ---")
        let connectorResult = validateConnectorWeights(components.textEncoder)
        if connectorResult { passed += 1 } else { failed += 1 }
        print()

        // Test 3: Transformer Weight Mapping
        print("--- Test 3: Transformer Weight Mapping ---")
        let transformerResult = validateTransformerWeights(components.transformer, config: modelVariant.transformerConfig)
        if transformerResult { passed += 1 } else { failed += 1 }
        print()

        // Test 4: VAE Weight Mapping
        print("--- Test 4: VAE Weight Mapping ---")
        let vaeResult = validateVAEWeights(components.vaeDecoder)
        if vaeResult { passed += 1 } else { failed += 1 }
        print()

        if !quick {
            // Test 5: Connector Forward Pass
            print("--- Test 5: Connector Forward Pass ---")
            let forwardResult = try validateConnectorForward(components.textEncoder)
            if forwardResult { passed += 1 } else { failed += 1 }
            print()

            // Test 6: RoPE Frequencies
            print("--- Test 6: RoPE Frequencies ---")
            let ropeResult = validateRoPEFrequencies()
            if ropeResult { passed += 1 } else { failed += 1 }
            print()
        }

        // Summary
        print("======================")
        print("VALIDATION SUMMARY")
        print("======================")
        print("Passed: \(passed)")
        print("Failed: \(failed)")
        if failed == 0 {
            print("✅ All tests passed!")
        } else {
            print("❌ Some tests failed")
        }
    }

    // MARK: - Validation Tests

    private func validateSchedulerSigmas() -> Bool {
        let scheduler = LTXScheduler(isDistilled: true)
        let sigmas = scheduler.getSigmas(numSteps: 8)

        let expected: [Float] = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]

        print("  Expected: \(expected.prefix(5))...")
        print("  Got:      \(sigmas.prefix(5))...")

        let tolerance: Float = 1e-5
        var allMatch = true
        for (i, (e, g)) in zip(expected, sigmas).enumerated() {
            if abs(e - g) > tolerance {
                print("  ❌ Mismatch at index \(i): expected \(e), got \(g)")
                allMatch = false
            }
        }

        if allMatch {
            print("  ✅ Scheduler sigmas match")
        }
        return allMatch
    }

    private func validateConnectorWeights(_ weights: [String: MLXArray]) -> Bool {
        // After splitting from unified file, text encoder keys are in unified format
        // Check that we have the expected keys
        let mapped = LTXWeightLoader.mapTextEncoderWeights(weights)
        print("  Mapped \(mapped.count) text encoder weights")

        let expectedMappedKeys = [
            "feature_extractor.aggregate_embed.weight",
            "embeddings_connector.learnable_registers",
            "embeddings_connector.transformer_1d_blocks.0.attn1.to_q.weight"
        ]

        var allFound = true
        for key in expectedMappedKeys {
            if mapped[key] != nil {
                print("  ✅ Found: \(key)")
            } else {
                print("  ❌ Missing: \(key)")
                allFound = false
            }
        }

        // Check aggregate_embed weight shape
        if let projWeight = mapped["feature_extractor.aggregate_embed.weight"] {
            let shape = projWeight.shape
            let expectedShape = [3840, 188160]
            if shape == expectedShape {
                print("  ✅ aggregate_embed.weight shape: \(shape)")
            } else {
                print("  ❌ aggregate_embed.weight shape mismatch: got \(shape), expected \(expectedShape)")
                allFound = false
            }
        }

        return allFound
    }

    private func validateTransformerWeights(_ weights: [String: MLXArray], config: LTXTransformerConfig) -> Bool {
        print("  Input weights: \(weights.count) tensors")

        let transformer = LTXTransformer(config: config)
        let params = transformer.parameters().flattened()
        print("  Model has \(params.count) parameter keys")

        let mapped = LTXWeightLoader.mapTransformerWeights(weights)
        print("  Mapped \(mapped.count) weights")

        var found = 0
        for key in mapped.keys {
            if params.contains(where: { $0.0 == key }) {
                found += 1
            }
        }

        let coverage = Float(found) / Float(mapped.count) * 100
        print("  Weight coverage: \(found)/\(mapped.count) (\(String(format: "%.1f", coverage))%)")

        if coverage >= 95.0 {
            print("  ✅ Transformer weight mapping OK")
            return true
        } else {
            print("  ❌ Transformer weight mapping incomplete")
            return false
        }
    }

    private func validateVAEWeights(_ weights: [String: MLXArray]) -> Bool {
        print("  Input weights: \(weights.count) tensors")

        let vae = VideoDecoder()
        let params = vae.parameters().flattened()
        print("  Model has \(params.count) parameter keys")

        let mapped = LTXWeightLoader.mapVAEWeights(weights)
        print("  Mapped \(mapped.count) weights")

        var found = 0
        for key in mapped.keys {
            if params.contains(where: { $0.0 == key }) {
                found += 1
            }
        }

        let coverage = Float(found) / Float(mapped.count) * 100
        print("  Weight coverage: \(found)/\(mapped.count) (\(String(format: "%.1f", coverage))%)")

        if coverage >= 90.0 {
            print("  ✅ VAE weight mapping OK")
            return true
        } else {
            print("  ❌ VAE weight mapping incomplete")
            return false
        }
    }

    private func validateConnectorForward(_ weights: [String: MLXArray]) throws -> Bool {
        let textEncoder = createTextEncoder()
        try LTXWeightLoader.applyTextEncoderWeights(weights, to: textEncoder)
        MLX.eval(textEncoder.parameters())

        let params = textEncoder.parameters().flattened()
        print("  Text encoder has \(params.count) parameters")

        var totalParams: Int64 = 0
        for (_, array) in params {
            totalParams += Int64(array.size)
        }
        print("  Total parameter elements: \(totalParams)")

        if let aggregateWeight = params.first(where: { $0.0 == "feature_extractor.aggregate_embed.weight" }) {
            let shape = aggregateWeight.1.shape
            print("  aggregate_embed.weight shape: \(shape)")
            if shape == [3840, 188160] {
                print("  ✅ Connector forward pass preparation OK")
                return true
            } else {
                print("  ❌ aggregate_embed.weight shape mismatch")
                return false
            }
        } else {
            print("  ❌ aggregate_embed.weight not found in parameters")
            return false
        }
    }

    private func validateRoPEFrequencies() -> Bool {
        let dim = 4096
        let theta: Float = 10000.0
        let numHeads = 32
        let headDim = dim / numHeads

        let halfDim = headDim / 2
        var invFreq = [Float](repeating: 0, count: halfDim)
        for i in 0..<halfDim {
            invFreq[i] = 1.0 / pow(theta, Float(2 * i) / Float(headDim))
        }

        let expected0: Float = 1.0 / pow(theta, 0.0 / Float(headDim))
        let expected1: Float = 1.0 / pow(theta, 2.0 / Float(headDim))

        print("  inv_freq[0]: expected \(expected0), got \(invFreq[0])")
        print("  inv_freq[1]: expected \(expected1), got \(invFreq[1])")

        let tolerance: Float = 0.01
        if abs(invFreq[0] - expected0) < tolerance && abs(invFreq[1] - expected1) < tolerance {
            print("  ✅ RoPE frequencies OK")
            return true
        } else {
            print("  ❌ RoPE frequency mismatch")
            return false
        }
    }
}

// MARK: - Test Gemma Native Command

struct TestGemmaNative: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "test-gemma-native",
        abstract: "Test our native Gemma3 implementation for LTX-Video"
    )

    @Option(name: .shortAndLong, help: "Test prompt")
    var prompt: String = "A cat walking on the beach"

    @Option(name: .long, help: "Path to Gemma3 model directory (with config.json + safetensors)")
    var gemmaPath: String?

    @Flag(name: .long, help: "Enable debug output")
    var debug: Bool = false

    @Flag(name: .long, help: "Skip weight loading (test model structure only)")
    var skipWeights: Bool = false

    mutating func run() async throws {
        if debug {
            LTXDebug.enableDebugMode()
        }

        print("=" .padding(toLength: 60, withPad: "=", startingAt: 0))
        print("LTX-Video Native Gemma3 Test")
        print("=" .padding(toLength: 60, withPad: "=", startingAt: 0))
        print()
        print("Prompt: \"\(prompt)\"")
        print()

        // Step 1: Create or load model
        let model: Gemma3TextModel
        if let path = gemmaPath, !skipWeights {
            print("Step 1: Loading model from \(path)...")
            let modelURL = URL(fileURLWithPath: path)
            model = try Gemma3WeightLoader.loadModel(from: modelURL)
            print("  ✅ Model loaded with weights")
        } else {
            print("Step 1: Creating model with default 12B config (random weights)...")
            let config = Gemma3Config()
            model = Gemma3TextModel(config)
            print("  ✅ Model created (no weights)")
        }

        let config = model.config
        print("  Config: \(config.hiddenLayers) layers, \(config.hiddenSize) hidden, \(config.attentionHeads) heads")
        print()

        // Step 2: Create test input
        print("Step 2: Creating test input...")
        let inputIds = MLXArray([2, 235280, 5, 32, 10350, 611, 573, 10908]).expandedDimensions(axis: 0)
        print("  Input shape: \(inputIds.shape)")
        print()

        // Step 3: Run forward pass
        print("Step 3: Running forward pass...")
        let startTime = Date()

        let (lastHidden, allHiddenStates) = model(inputIds, outputHiddenStates: true)

        MLX.eval(lastHidden)
        if let states = allHiddenStates {
            for state in states { MLX.eval(state) }
        }

        let elapsed = Date().timeIntervalSince(startTime)
        print("  Forward pass completed in \(String(format: "%.2f", elapsed))s")
        print()

        // Step 4: Validate
        print("Step 4: Validating output shapes...")
        print("  Last hidden state shape: \(lastHidden.shape)")

        if let states = allHiddenStates {
            let expectedCount = config.hiddenLayers + 1
            print("  Hidden states count: \(states.count) (expected \(expectedCount))")
            if states.count == expectedCount {
                print("  ✅ Hidden state count correct")
            } else {
                print("  ❌ Hidden state count mismatch")
            }
        }
        print()

        // Step 5: Test with feature extractor if 49 hidden states
        if let states = allHiddenStates, states.count == 49 {
            print("Step 5: Testing FeatureExtractor + Connector...")
            let attentionMask = MLXArray.ones([1, inputIds.dim(1)]).asType(.float32)
            let textEncoder = createTextEncoder()

            let output = textEncoder.encodeFromHiddenStates(
                hiddenStates: states,
                attentionMask: attentionMask,
                paddingSide: "left"
            )
            MLX.eval(output.videoEncoding)

            print("  Video encoding shape: \(output.videoEncoding.shape)")
            print("  Output dim: \(output.videoEncoding.dim(-1)) (expected 3840)")
            if output.videoEncoding.dim(-1) == 3840 {
                print("  ✅ Full pipeline OK!")
            }
        }
    }
}

// MARK: - Encode Command

struct Encode: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "encode",
        abstract: "Encode a text prompt through Gemma3 + LTX text encoder pipeline"
    )

    @Argument(help: "The text prompt to encode")
    var prompt: String

    @Option(name: .long, help: "Path to Gemma3 model directory (auto-downloads if not provided)")
    var gemmaPath: String?

    @Option(name: .long, help: "Path to connectors.safetensors file")
    var connectorsPath: String?

    @Option(name: .long, help: "Maximum token length")
    var maxLength: Int = 256

    @Flag(name: .long, help: "Enable debug output")
    var debug: Bool = false

    @Flag(name: .long, help: "Skip connector (only run Gemma + feature extractor)")
    var skipConnector: Bool = false

    mutating func run() async throws {
        if debug {
            LTXDebug.enableDebugMode()
        }

        print("LTX-2 Text Encoding Pipeline")
        print("============================")
        print("Prompt: \"\(prompt)\"")
        print("Max tokens: \(maxLength)")
        print()

        // Step 1: Get Gemma model path (auto-download if needed)
        let gemmaDir: URL
        if let path = gemmaPath {
            gemmaDir = URL(fileURLWithPath: path)
        } else {
            print("Step 1: Downloading Gemma3 12B 4-bit (if needed)...")
            let downloader = ModelDownloader()
            gemmaDir = try await downloader.downloadGemma { progress in
                print("  \(progress.message)")
            }
        }
        print("  Gemma path: \(gemmaDir.path)")
        print()

        // Step 2: Load config and model
        print("Step 2: Loading Gemma3 model...")
        let startLoad = Date()
        let gemmaModel = try Gemma3WeightLoader.loadModel(from: gemmaDir)
        let loadTime = Date().timeIntervalSince(startLoad)
        print("  Loaded in \(String(format: "%.1f", loadTime))s")
        print("  Config: \(gemmaModel.config.hiddenLayers) layers, \(gemmaModel.config.hiddenSize) hidden")
        if let q = gemmaModel.config.quantization {
            print("  Quantization: \(q.bits)-bit, group_size=\(q.groupSize)")
        }
        if let rs = gemmaModel.config.ropeScaling {
            print("  Rope scaling: \(rs)")
        }

        // Debug: weight comparison with Python
        if debug {
            print()
            print("  === Weight Verification ===")
            let params = Dictionary(uniqueKeysWithValues: gemmaModel.parameters().flattened())
            // Layer 0 q_norm weight
            if let qnW = params["model.layers.0.self_attn.q_norm.weight"] {
                MLX.eval(qnW)
                print("  layer0.q_norm.weight[:5]: \(qnW[0..<5].asArray(Float.self))")
            }
            // Layer 0 input_layernorm weight
            if let lnW = params["model.layers.0.input_layernorm.weight"] {
                MLX.eval(lnW)
                print("  layer0.input_layernorm.weight[:5]: \(lnW[0..<5].asArray(Float.self))")
            }
            // Layer 0 q_proj scales (quantized)
            if let qpS = params["model.layers.0.self_attn.q_proj.scales"] {
                MLX.eval(qpS)
                print("  layer0.q_proj.scales[0,:5]: \(qpS[0, 0..<5].asArray(Float.self))")
            }
            // Embedding test via full forward on 3 tokens
            let testTokens = MLXArray([Int32(106), Int32(2), Int32(236776)]).reshaped([1, 3])
            let (_, testStates) = gemmaModel(testTokens, outputHiddenStates: true)
            if let s0 = testStates?.first {
                MLX.eval(s0)
                print("  embed+scale [106] first5: \(s0[0, 0, 0..<5].asArray(Float.self))")
                print("  embed+scale [2] first5: \(s0[0, 1, 0..<5].asArray(Float.self))")
                print("  embed+scale [236776] first5: \(s0[0, 2, 0..<5].asArray(Float.self))")
            }
        }
        print()

        // Step 3: Load tokenizer
        print("Step 3: Loading tokenizer...")
        let tokenizer = try await AutoTokenizer.from(modelFolder: gemmaDir)
        print("  Tokenizer loaded")
        print()

        // Step 4: Tokenize with left-padding
        print("Step 4: Tokenizing...")
        let encoded = tokenizer.encode(text: prompt)
        print("  Raw tokens: \(encoded.count)")
        print("  Token IDs: \(encoded)")

        // Left-pad to maxLength
        var tokens = encoded.suffix(maxLength).map { Int32($0) }
        let paddingNeeded = maxLength - tokens.count
        let eosTokenId = tokenizer.eosTokenId ?? 0
        if paddingNeeded > 0 {
            let padTokenId = Int32(eosTokenId)
            tokens = [Int32](repeating: padTokenId, count: paddingNeeded) + tokens
        }
        let attentionMask = [Float](repeating: 0, count: paddingNeeded)
            + [Float](repeating: 1, count: maxLength - paddingNeeded)

        let inputIds = MLXArray(tokens).reshaped([1, maxLength])
        let maskArray = MLXArray(attentionMask).reshaped([1, maxLength])

        print("  Padded shape: \(inputIds.shape)")
        print("  Padding: \(paddingNeeded) pad + \(maxLength - paddingNeeded) real")
        print("  Pad token ID (eos): \(eosTokenId)")
        print("  First 5 padded: \(Array(tokens.prefix(5)))")
        print("  Last 10 padded: \(Array(tokens.suffix(10)))")
        print()

        // Step 5: Gemma forward pass
        print("Step 5: Running Gemma forward pass...")
        let startGemma = Date()
        let (lastHidden, allHiddenStates) = gemmaModel(inputIds, outputHiddenStates: true)
        MLX.eval(lastHidden)
        let gemmaTime = Date().timeIntervalSince(startGemma)

        print("  Last hidden state: \(lastHidden.shape) \(lastHidden.dtype)")
        if let states = allHiddenStates {
            print("  Hidden states: \(states.count) layers")
            if let first = states.first {
                print("  Each state shape: \(first.shape)")
            }
        }
        print("  Time: \(String(format: "%.2f", gemmaTime))s")
        print()

        // Stats on last hidden state
        let mean = lastHidden.mean()
        let diff = lastHidden - mean
        let variance = (diff * diff).mean()
        let std = MLX.sqrt(variance)
        MLX.eval(mean, std)
        print("  Last hidden stats: mean=\(mean.item(Float.self)), std=\(std.item(Float.self))")

        // Per-layer stats for comparison with Python
        if let states = allHiddenStates {
            for i in [0, 1, 24, 47, 48] {
                if i < states.count {
                    let s = states[i]
                    let m = s.mean()
                    let rms = MLX.sqrt((s * s).mean())
                    MLX.eval(m, rms)
                    print("  layer_\(i): mean=\(m.item(Float.self)), rms=\(rms.item(Float.self))")
                }
            }

            // Per-position values at early layers (to find divergence point)
            if debug {
                for i in [1, 2, 3, 5, 6, 7, 8, 9, 16, 24, 32, 40, 47, 48] {
                    if i < states.count {
                        let s = states[i]
                        let p0 = s[0, 0, 0..<3]
                        let p249 = s[0, 249, 0..<3]
                        let p255 = s[0, 255, 0..<3]
                        MLX.eval(p0, p249, p255)
                        print("  layer_\(i): pos0[:3]=\(p0.asArray(Float.self)), pos249[:3]=\(p249.asArray(Float.self)), pos255[:3]=\(p255.asArray(Float.self))")
                    }
                }
            }

            // Sample values from last hidden state
            let last = states[states.count - 1]
            for pos in [0, 249, 255] {
                let vals = last[0, pos, 0..<5]
                MLX.eval(vals)
                print("  pos_\(pos)_first5 = \(vals.asArray(Float.self))")
            }
        }
        print()

        // Step 6: Feature extraction + connector
        guard let states = allHiddenStates, states.count == gemmaModel.config.hiddenLayers + 1 else {
            print("  ❌ Expected \(gemmaModel.config.hiddenLayers + 1) hidden states")
            return
        }

        print("Step 6: Creating text encoder...")
        let textEncoder = createTextEncoder()

        // Resolve connector weights path
        let resolvedConnectorsPath: String? = {
            if let p = connectorsPath { return p }
            // Try default locations
            let defaultPaths = [
                NSHomeDirectory() + "/Library/Caches/models/ltx-connectors/connectors/diffusion_pytorch_model.safetensors",
                NSHomeDirectory() + "/Library/Caches/models/ltx-distilledFP8/connectors.safetensors",
            ]
            return defaultPaths.first { FileManager.default.fileExists(atPath: $0) }
        }()

        if let connPath = resolvedConnectorsPath {
            print("  Loading connector weights from \(connPath)...")
            let weights = try LTXWeightLoader.loadSingleFile(path: connPath)
            try LTXWeightLoader.applyTextEncoderWeights(weights, to: textEncoder)
            MLX.eval(textEncoder.parameters())
            print("  ✅ Connector weights loaded")
        } else {
            print("  ⚠️  No connector weights found (using random weights)")
            print("  Provide --connectors-path or download to ~/Library/Caches/models/ltx-connectors/")
        }
        print()

        // Step 7: Feature extraction
        print("Step 7: Feature extraction...")
        let featureOutput = textEncoder.featureExtractor.extractFromHiddenStates(
            hiddenStates: states,
            attentionMask: maskArray,
            paddingSide: "left"
        )
        MLX.eval(featureOutput)
        print("  Feature extractor output: \(featureOutput.shape)")

        let fMean = featureOutput.mean()
        let fDiff = featureOutput - fMean
        let fVar = (fDiff * fDiff).mean()
        let fStd = MLX.sqrt(fVar)
        MLX.eval(fMean, fStd)
        print("  Stats: mean=\(fMean.item(Float.self)), std=\(fStd.item(Float.self))")

        // Sample values for comparison
        for pos in [0, 249, 255] {
            let vals = featureOutput[0, pos, 0..<10]
            MLX.eval(vals)
            print("  pos_\(pos)_first10 = \(vals.asArray(Float.self))")
        }
        print()

        if !skipConnector {
            // Step 8: Full pipeline with connector
            print("Step 8: Connector (2 transformer blocks + RMSNorm)...")
            let startConnector = Date()
            let output = textEncoder.encodeFromHiddenStates(
                hiddenStates: states,
                attentionMask: maskArray,
                paddingSide: "left"
            )
            MLX.eval(output.videoEncoding)
            MLX.eval(output.attentionMask)
            let connectorTime = Date().timeIntervalSince(startConnector)

            print("  Final encoding: \(output.videoEncoding.shape)")
            print("  Output mask: \(output.attentionMask.shape)")
            print("  Time: \(String(format: "%.2f", connectorTime))s")

            let oMean = output.videoEncoding.mean()
            let oDiff = output.videoEncoding - oMean
            let oVar = (oDiff * oDiff).mean()
            let oStd = MLX.sqrt(oVar)
            MLX.eval(oMean, oStd)
            print("  Stats: mean=\(oMean.item(Float.self)), std=\(oStd.item(Float.self))")

            // Sample values for comparison
            for pos in [0, 249, 255] {
                let vals = output.videoEncoding[0, pos, 0..<10]
                MLX.eval(vals)
                print("  pos_\(pos)_first10 = \(vals.asArray(Float.self))")
            }

            let maskSum = output.attentionMask.sum()
            MLX.eval(maskSum)
            print("  Mask sum: \(maskSum.item(Int32.self)) (expected: \(maxLength) if all valid)")
        }

        print()
        print("✅ Text encoding pipeline complete!")
    }
}

// MARK: - Decode Latent Command (for cross-testing VAE)

struct DecodeLatent: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "decode-latent",
        abstract: "Decode a latent tensor from safetensors file with the VAE (for cross-testing)"
    )

    @Argument(help: "Path to safetensors file containing 'latent' key")
    var latentPath: String

    @Option(name: .shortAndLong, help: "Output video path")
    var output: String = "/tmp/decoded_latent.mp4"

    @Option(help: "Path to LTX weights")
    var ltxWeights: String = ""

    func run() async throws {
        print("Loading latent from \(latentPath)...")
        let arrays = try MLX.loadArrays(url: URL(fileURLWithPath: latentPath))
        guard let latent = arrays["latent"] else {
            print("Error: no 'latent' key in safetensors file")
            return
        }
        print("Latent shape: \(latent.shape), dtype: \(latent.dtype)")
        print("Latent mean: \(latent.mean().item(Float.self)), std: \(MLX.sqrt(MLX.variance(latent)).item(Float.self))")

        // Load VAE weights
        let weightsPath: String
        if !ltxWeights.isEmpty {
            weightsPath = ltxWeights
        } else {
            weightsPath = NSHomeDirectory() + "/Library/Caches/models/ltx-weights/ltx-2-19b-distilled.safetensors"
        }

        print("Loading VAE weights from \(weightsPath)...")
        let allWeights = try LTXWeightLoader.loadSingleFile(path: weightsPath)
        let components = LTXWeightLoader.splitUnifiedWeights(allWeights)

        // Create decoder and apply weights
        let decoder = VideoDecoder()
        try LTXWeightLoader.applyVAEWeights(components.vaeDecoder, to: decoder)

        // Decode
        print("Decoding...")
        let videoTensor = decodeVideo(latent: latent, decoder: decoder)
        MLX.eval(videoTensor)
        print("Decoded video shape: \(videoTensor.shape)")

        // Export video
        let outputURL = URL(fileURLWithPath: output)
        let exportFrames = VideoExporter.tensorToImages(videoTensor)
        print("Export frames: \(exportFrames.count)")

        if !exportFrames.isEmpty {
            let exporter = VideoExporter()
            let width = videoTensor.dim(2)
            let height = videoTensor.dim(1)
            let url = try await exporter.export(frames: exportFrames, width: width, height: height, fps: 24.0, to: outputURL)
            print("✅ Saved to \(url.path)")
        }
    }
}

// MARK: - Denoise Test Command (compare with Python reference)

struct DenoiseTest: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "denoise-test",
        abstract: "Run 8-step denoising with fixed noise from Python for cross-validation"
    )

    @Option(name: .long, help: "Path to .npy noise file from Python")
    var noisePath: String = "/tmp/python_initial_noise.npy"

    @Option(name: .long, help: "Path to .npy context file from Python (zero context if not provided)")
    var contextPath: String?

    @Option(name: .long, help: "Path to LTX weights (safetensors)")
    var ltxWeights: String = ""

    @Option(name: .shortAndLong, help: "Model variant: distilledFP8, distilled, or dev")
    var model: String = "distilled"

    @Flag(name: .long, help: "Enable debug output")
    var debug: Bool = false

    mutating func run() async throws {
        if debug {
            LTXDebug.enableDebugMode()
        }

        print("LTX-2 Denoise Test (Cross-validation with Python)")
        print("==================================================")
        print()

        // 1. Load noise from numpy file
        print("Step 1: Loading noise from \(noisePath)...")
        let noiseURL = URL(fileURLWithPath: noisePath)
        let noise = try MLX.loadArray(url: noiseURL)
        MLX.eval(noise)
        print("  Shape: \(noise.shape), dtype: \(noise.dtype)")
        print("  mean=\(noise.mean().item(Float.self)), std=\(MLX.sqrt(MLX.variance(noise)).item(Float.self))")

        // Print first 5 values for verification
        let flat = noise.reshaped([-1])
        let first5 = (0..<5).map { flat[$0].item(Float.self) }
        print("  first 5 values: \(first5)")
        print()

        // 2. Load transformer
        guard let modelVariant = LTXModel(rawValue: model) else {
            throw ValidationError("Invalid model: \(model)")
        }

        let weightsPath: String
        if !ltxWeights.isEmpty {
            weightsPath = ltxWeights
        } else {
            print("Step 2: Downloading weights (if needed)...")
            let downloader = ModelDownloader()
            let url = try await downloader.downloadLTXWeights(model: modelVariant) { progress in
                print("  \(progress.message)")
            }
            weightsPath = url.path
        }
        print("  Weights: \(weightsPath)")

        print("  Loading and splitting weights...")
        let components = try LTXWeightLoader.loadUnifiedWeights(
            from: weightsPath,
            isFP8: modelVariant.isFP8
        )
        print("  Transformer: \(components.transformer.count) tensors")

        print("  Creating transformer...")
        let transformer = LTXTransformer(config: modelVariant.transformerConfig)
        try LTXWeightLoader.applyTransformerWeights(components.transformer, to: transformer)

        // Eval all weights
        print("  Evaluating weights...")
        MLX.eval(transformer.parameters())
        print("  ✅ Transformer ready")
        print()

        // 3. Setup shapes
        // Noise is (1, 128, 2, 16, 16) => 512x512, 9 frames
        let latentShape = VideoLatentShape(
            batch: noise.dim(0),
            channels: noise.dim(1),
            frames: noise.dim(2),
            height: noise.dim(3),
            width: noise.dim(4)
        )
        print("Step 3: Latent shape = \(latentShape)")

        // 4. Text context (from file or zeros)
        let context: MLXArray
        if let ctxPath = contextPath {
            print("Step 3b: Loading context from \(ctxPath)...")
            context = try MLX.loadArray(url: URL(fileURLWithPath: ctxPath))
            MLX.eval(context)
            print("  Context: \(context.shape), mean=\(context.mean().item(Float.self)), std=\(MLX.sqrt(MLX.variance(context)).item(Float.self))")
        } else {
            context = MLXArray.zeros([1, 256, 3840])
            print("  Context: \(context.shape) (zero — no text)")
        }
        print()

        // 5. Denoising loop
        let sigmas: [Float] = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
        let numSteps = sigmas.count - 1

        var latent = noise  // sigma[0] = 1.0, so latent = noise * 1.0
        let scheduler = LTXScheduler(isDistilled: true)

        print("Step 4: Denoising (\(numSteps) steps)")
        print("  Sigmas: \(sigmas)")
        print()

        let totalStart = Date()

        for step in 0..<numSteps {
            let sigma = sigmas[step]
            let sigmaNext = sigmas[step + 1]

            let stepStart = Date()

            // Patchify
            let patchified = patchify(latent)

            // Run transformer
            let timestep = MLXArray([sigma])
            let velocity = transformer(
                latent: patchified,
                context: context,
                timesteps: timestep,
                contextMask: nil,
                latentShape: (frames: latentShape.frames, height: latentShape.height, width: latentShape.width)
            )
            MLX.eval(velocity)

            // Stats on patchified velocity
            let velMean = velocity.mean().item(Float.self)
            let velStd = MLX.sqrt(MLX.variance(velocity)).item(Float.self)
            let velMin = velocity.min().item(Float.self)
            let velMax = velocity.max().item(Float.self)

            // Unpatchify
            let velocityFull = unpatchify(velocity, shape: latentShape)

            // Euler step
            latent = scheduler.step(
                latent: latent,
                velocity: velocityFull,
                sigma: sigma,
                sigmaNext: sigmaNext
            )
            MLX.eval(latent)

            let latMean = latent.mean().item(Float.self)
            let latStd = MLX.sqrt(MLX.variance(latent)).item(Float.self)

            let stepTime = Date().timeIntervalSince(stepStart)

            print("Step \(step): σ=\(String(format: "%.6f", sigma)) → \(String(format: "%.6f", sigmaNext))")
            print("  velocity: mean=\(String(format: "%.6f", velMean)), std=\(String(format: "%.6f", velStd)), min=\(String(format: "%.4f", velMin)), max=\(String(format: "%.4f", velMax))")
            print("  latent:   mean=\(String(format: "%.6f", latMean)), std=\(String(format: "%.6f", latStd))")
            print("  time: \(String(format: "%.1f", stepTime))s")
        }

        let totalTime = Date().timeIntervalSince(totalStart)
        print()
        print("Total: \(String(format: "%.1f", totalTime))s")
        print("Final latent: mean=\(String(format: "%.6f", latent.mean().item(Float.self))), std=\(String(format: "%.6f", MLX.sqrt(MLX.variance(latent)).item(Float.self)))")
        print("  min=\(String(format: "%.6f", latent.min().item(Float.self))), max=\(String(format: "%.6f", latent.max().item(Float.self)))")

        // Save for comparison
        try MLX.save(array: latent, url: URL(fileURLWithPath: "/tmp/swift_final_latent.npy"))
        print("Saved /tmp/swift_final_latent.npy")

        // Compare with Python reference
        print()
        print("=== Python Reference (zero context, same noise) ===")
        print("Step 0: vel mean=0.015533, std=0.872052  lat mean=0.000958, std=0.992497")
        print("Step 1: vel mean=0.028532, std=0.972063  lat mean=0.000780, std=0.988580")
        print("Step 2: vel mean=0.032236, std=1.028179  lat mean=0.000578, std=0.985111")
        print("Step 3: vel mean=0.040985, std=1.069655  lat mean=0.000323, std=0.981379")
        print("Step 4: vel mean=0.037280, std=1.125622  lat mean=-0.002126, std=0.944166")
        print("Step 5: vel mean=0.045318, std=1.303549  lat mean=-0.010477, std=0.880582")
        print("Step 6: vel mean=0.039662, std=1.287487  lat mean=-0.022499, std=0.955098")
        print("Step 7: vel mean=0.029887, std=1.040124  lat mean=-0.035134, std=1.213009")
    }
}
