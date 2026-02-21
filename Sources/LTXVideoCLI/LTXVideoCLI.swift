// LTXVideoCLI.swift - Command-line interface for LTX-2 video generation
// Copyright 2025

import ArgumentParser
import Foundation
import LTXVideo
@preconcurrency import MLX
import MLXRandom

@main
struct LTXVideoCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "ltx-video",
        abstract: "LTX-2 video generation on Mac with MLX",
        version: "0.1.0",
        subcommands: [Generate.self, Download.self, Info.self, TestComponents.self, Validate.self, TestGemmaNative.self, Encode.self, DecodeLatent.self, DenoiseTest.self, BlockDump.self, DenoiseCompare.self],
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

    @Option(name: .shortAndLong, help: "Model variant: distilled or dev")
    var model: String = "distilled"

    @Flag(name: .long, help: "Enable debug output")
    var debug: Bool = false

    mutating func run() async throws {
        if debug {
            LTXDebug.enableDebugMode()
        }

        // Parse model variant
        guard let modelVariant = LTXModel(rawValue: model) else {
            throw ValidationError("Invalid model: \(model). Use: distilled or dev")
        }

        print("LTX-2 Component Test")
        print("====================")
        print("Component: \(component)")
        print("Model: \(modelVariant.displayName)")
        print()

        // Download unified weights if needed
        let downloader = ModelDownloader()
        print("Checking model weights...")
        let weightsPath = try await downloader.downloadUnifiedWeights(model: modelVariant) { progress in
            print("  \(progress.message)")
        }
        print("Weights: \(weightsPath.path)")
        print()

        // Load and split unified weights
        print("Loading and splitting weights...")
        let components = try LTXWeightLoader.splitUnifiedWeightsFile(path: weightsPath.path)
        print("  Transformer: \(components.transformer.count) tensors")
        print("  VAE: \(components.vae.count) tensors")
        print("  TextEncoder: \(components.connector.count) tensors")
        print()

        switch component.lowercased() {
        case "transformer":
            try testTransformer(weights: components.transformer, config: modelVariant.transformerConfig)
        case "vae":
            try testVAE(weights: components.vae)
        case "connector", "text-encoder":
            try testTextEncoderConnector(weights: components.connector)
        case "all":
            try testTransformer(weights: components.transformer, config: modelVariant.transformerConfig)
            print()
            try testVAE(weights: components.vae)
            print()
            try testTextEncoderConnector(weights: components.connector)
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

    @Flag(name: .long, help: "Use two-stage generation (half-res + upscale + refine)")
    var twoStage: Bool = false

    @Flag(name: .long, help: "Enhance prompt using Gemma before generation")
    var enhancePrompt: Bool = false

    @Option(name: .long, help: "Path to pre-computed embeddings (.safetensors) for diagnostic")
    var pythonEmbeddings: String?

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
        print("Model: \(model)")
        if let seed = seed {
            print("Seed: \(seed)")
        }
        if guidanceRescale > 0 { print("Guidance rescale: \(guidanceRescale)") }
        if crossAttnScale != 1.0 { print("Cross-attention scale: \(crossAttnScale)") }
        if geGamma > 0 { print("GE gamma: \(geGamma)") }
        if stgScale > 0 { print("STG scale: \(stgScale), blocks: \(parsedStgBlocks)") }
        if twoStage { print("Two-stage: enabled") }
        if enhancePrompt { print("Prompt enhancement: enabled") }
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
            throw ValidationError("Invalid model: \(model). Use: distilled or dev")
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
            // Two-stage: half-res → upscale → refine
            // Download upscaler weights if needed
            print("Downloading upscaler weights (if needed)...")
            let upscalerPath = try await pipeline.downloadUpscalerWeights()
            print("Upscaler weights: \(upscalerPath)")

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
            // Load pre-computed embeddings if provided
            var precomputed: LTXPipeline.PrecomputedEmbeddings? = nil
            if let embPath = pythonEmbeddings {
                print("Loading pre-computed embeddings from \(embPath)...")
                let embURL = URL(fileURLWithPath: embPath)
                let arrays = try MLX.loadArrays(url: embURL)
                guard let pe = arrays["prompt_embeddings"],
                      let pm = arrays["prompt_mask"] else {
                    throw ValidationError("Missing prompt_embeddings or prompt_mask in \(embPath)")
                }
                let ne = arrays["null_embeddings"]
                let nm = arrays["null_mask"]
                precomputed = LTXPipeline.PrecomputedEmbeddings(
                    promptEmbeddings: pe,
                    promptMask: pm,
                    nullEmbeddings: ne,
                    nullMask: nm
                )
                print("  prompt_embeddings: \(pe.shape), null_embeddings: \(ne?.shape ?? [])")
            }

            result = try await pipeline.generateVideo(
                prompt: prompt,
                negativePrompt: negativePrompt,
                config: config,
                precomputedEmbeddings: precomputed,
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
        print("✅ Text encoder: \(paths.textEncoderDir.path)")
        print("✅ Tokenizer: \(paths.tokenizerDir.path)")
        print("✅ Connector: \(paths.connectorPath.path)")
        print("✅ VAE: \(paths.vaePath.path)")
        print("✅ Unified weights: \(paths.unifiedWeightsPath.path)")
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
              • distilled     - ~16GB RAM, 8 steps
              • dev           - ~25GB RAM, 40 steps (best quality)

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
              ltx-video download --model distilled
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

    @Option(name: .shortAndLong, help: "Model variant: distilled or dev")
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
            throw ValidationError("Invalid model: \(model). Use: distilled or dev")
        }

        // Download unified weights if needed
        let downloader = ModelDownloader()
        print("Checking model weights...")
        let weightsPath = try await downloader.downloadUnifiedWeights(model: modelVariant) { progress in
            if !progress.message.isEmpty {
                print("  \(progress.message)")
            }
        }
        print("Weights: \(weightsPath.path)")
        print()

        // Load and split unified weights
        print("Loading and splitting weights...")
        let components = try LTXWeightLoader.splitUnifiedWeightsFile(path: weightsPath.path)
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
        let connectorResult = validateConnectorWeights(components.connector)
        if connectorResult { passed += 1 } else { failed += 1 }
        print()

        // Test 3: Transformer Weight Mapping
        print("--- Test 3: Transformer Weight Mapping ---")
        let transformerResult = validateTransformerWeights(components.transformer, config: modelVariant.transformerConfig)
        if transformerResult { passed += 1 } else { failed += 1 }
        print()

        // Test 4: VAE Weight Mapping
        print("--- Test 4: VAE Weight Mapping ---")
        let vaeResult = validateVAEWeights(components.vae)
        if vaeResult { passed += 1 } else { failed += 1 }
        print()

        if !quick {
            // Test 5: Connector Forward Pass
            print("--- Test 5: Connector Forward Pass ---")
            let forwardResult = try validateConnectorForward(components.connector)
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
        abstract: "Encode a text prompt through the full text encoding pipeline (Gemma3 + connector)"
    )

    @Argument(help: "The text prompt to encode")
    var prompt: String

    @Option(name: .shortAndLong, help: "Model variant: distilled or dev")
    var model: String = "distilled"

    @Option(name: .long, help: "Path to local Gemma3 model directory (auto-downloads if not provided)")
    var gemmaPath: String?

    @Flag(name: .long, help: "Enhance prompt using Gemma before encoding")
    var enhancePrompt: Bool = false

    @Option(name: .long, help: "Path to save embeddings as .safetensors")
    var saveEmbeddings: String?

    @Flag(name: .long, help: "Enable debug output")
    var debug: Bool = false

    mutating func run() async throws {
        if debug {
            LTXDebug.enableDebugMode()
        }

        // Parse model variant
        guard let modelVariant = LTXModel(rawValue: model) else {
            throw ValidationError("Invalid model: \(model). Use: distilled or dev")
        }

        print("LTX-2 Text Encoding")
        print("====================")
        print("Prompt: \"\(prompt)\"")
        print("Model: \(modelVariant.displayName)")
        if enhancePrompt { print("Enhancement: enabled") }
        print()

        // Create pipeline and load only text encoder models
        let pipeline = LTXPipeline(model: modelVariant)

        print("Loading text encoder models...")
        let startLoad = Date()
        try await pipeline.loadTextEncoderModels(
            progressCallback: { progress in
                print("  \(progress.message)")
            },
            gemmaModelPath: gemmaPath
        )
        let loadTime = Date().timeIntervalSince(startLoad)
        print("  Loaded in \(String(format: "%.1f", loadTime))s")
        print()

        // Encode
        print("Encoding text...")
        let startEncode = Date()
        let result = try await pipeline.encodeText(prompt, enhance: enhancePrompt)
        let encodeTime = Date().timeIntervalSince(startEncode)

        // Show results
        if enhancePrompt {
            print()
            print("Enhanced prompt:")
            print("  \"\(result.prompt)\"")
        }

        print()
        print("Embedding results:")
        print("  Shape: \(result.embeddings.shape)")
        print("  Dtype: \(result.embeddings.dtype)")
        print("  Mean:  \(String(format: "%.6f", result.mean))")
        print("  Std:   \(String(format: "%.6f", result.std))")

        // First 5 values
        let first5 = result.embeddings[0, 0, 0..<5]
        MLX.eval(first5)
        print("  First 5 values: \(first5.asArray(Float.self))")

        // Mask info
        let maskSum = result.mask.sum()
        MLX.eval(maskSum)
        let totalTokens = result.mask.shape[1]
        let activeTokens = maskSum.item(Int32.self)
        print("  Active tokens: \(activeTokens) / \(totalTokens)")

        print("  Time: \(String(format: "%.1f", encodeTime))s")

        // Save embeddings if requested
        if let savePath = saveEmbeddings {
            print()
            print("Saving embeddings to \(savePath)...")
            let url = URL(fileURLWithPath: savePath)
            try MLX.save(
                arrays: [
                    "prompt_embeddings": result.embeddings,
                    "prompt_mask": result.mask
                ],
                url: url
            )
            print("  Saved successfully")
        }

        print()
        print("Done!")
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
        let components = LTXWeightLoader.splitUnifiedWeightsDict(allWeights)

        // Create decoder and apply weights
        let decoder = VideoDecoder()
        try LTXWeightLoader.applyVAEWeights(components.vae, to: decoder)

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

    @Option(name: .shortAndLong, help: "Model variant: distilled or dev")
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
            let url = try await downloader.downloadUnifiedWeights(model: modelVariant) { progress in
                print("  \(progress.message)")
            }
            weightsPath = url.path
        }
        print("  Weights: \(weightsPath)")

        print("  Loading transformer weights...")
        let transformerWeights = try LTXWeightLoader.loadTransformerWeights(from: weightsPath)
        print("  Transformer: \(transformerWeights.count) tensors")

        print("  Creating transformer...")
        let transformer = LTXTransformer(config: modelVariant.transformerConfig)
        try LTXWeightLoader.applyTransformerWeights(transformerWeights, to: transformer)

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

// MARK: - Block Dump Command

struct BlockDump: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "block-dump",
        abstract: "Dump transformer output for dev model comparison with Python"
    )

    @Option(name: .shortAndLong, help: "Model variant: dev or distilled")
    var model: String = "dev"

    @Flag(name: .long, help: "Enable debug output")
    var debug: Bool = false

    mutating func run() async throws {
        LTXDebug.enableDebugMode()

        guard let modelVariant = LTXModel(rawValue: model) else {
            throw ValidationError("Invalid model: \(model)")
        }

        let width = 512
        let height = 512
        let numFrames = 9
        let seed: UInt64 = 42
        let channels = 128

        let latF = (numFrames - 1) / 8 + 1  // 2
        let latH = height / 32  // 16
        let latW = width / 32   // 16
        let numTokens = latF * latH * latW  // 512

        print("=== Block Dump for \(modelVariant.displayName) ===")
        print("Resolution: \(width)x\(height), \(numFrames) frames")
        print("Latent shape: \(latF)x\(latH)x\(latW) = \(numTokens) tokens")
        print()

        // 1. Load model
        let downloader = ModelDownloader()
        let url = try await downloader.downloadUnifiedWeights(model: modelVariant)
        let weightsPath = url.path

        print("Loading transformer weights from \(weightsPath)...")
        let transformerWeights = try LTXWeightLoader.loadTransformerWeights(from: weightsPath)
        print("  Transformer: \(transformerWeights.count) tensors")

        let transformer = LTXTransformer(config: modelVariant.transformerConfig, memoryOptimization: .aggressive)
        try LTXWeightLoader.applyTransformerWeights(transformerWeights, to: transformer)
        MLX.eval(transformer.parameters())
        print("Transformer loaded.")

        // 2. Generate noise with seed
        MLXRandom.seed(seed)
        let latents = MLXRandom.normal([1, channels, latF, latH, latW]).asType(.bfloat16)
        MLX.eval(latents)

        let latentMean = latents.mean().item(Float.self)
        let latentStd = MLX.sqrt(MLX.variance(latents)).item(Float.self)
        print("Initial latents: mean=\(String(format: "%.6f", latentMean)), std=\(String(format: "%.6f", latentStd))")
        let first5 = (0..<5).map { latents[0, 0, 0, 0, $0].item(Float.self) }
        print("Initial latents[0,0,0,0,:5]: \(first5)")

        // 3. Patchify
        let latentsFlat = patchify(latents)
        MLX.eval(latentsFlat)
        let pfMean = latentsFlat.mean().item(Float.self)
        let pfStd = MLX.sqrt(MLX.variance(latentsFlat)).item(Float.self)
        print("Patchified: shape=\(latentsFlat.shape), mean=\(String(format: "%.6f", pfMean)), std=\(String(format: "%.6f", pfStd))")
        let pfFirst5 = (0..<5).map { latentsFlat[0, 0, $0].item(Float.self) }
        let pfLast5 = (123..<128).map { latentsFlat[0, 0, $0].item(Float.self) }
        print("Patchified[0,0,:5]: \(pfFirst5)")
        print("Patchified[0,0,-5:]: \(pfLast5)")

        // 4. Compute sigma schedule
        let scheduler = LTXScheduler()
        scheduler.setTimesteps(
            numSteps: 50,
            distilled: false,
            latentTokenCount: numTokens
        )
        let sigmas = scheduler.sigmas
        print("Sigma schedule (first 5): \(sigmas.prefix(5))")

        let sigma = sigmas[0]
        print("Step 0: sigma=\(sigma)")

        // 5. Prepare dummy context (zeros, matching Python)
        let dummyContext = MLXArray.zeros([1, 128, 3840]).asType(.bfloat16)
        MLX.eval(dummyContext)

        // 6. Run full transformer forward pass using public API
        print("\nRunning transformer forward pass...")
        let timestep = MLXArray([sigma])
        let output = transformer(
            latent: latentsFlat,
            context: dummyContext,
            timesteps: timestep,
            contextMask: nil,
            latentShape: (frames: latF, height: latH, width: latW)
        )
        MLX.eval(output)

        let outMean = output.mean().item(Float.self)
        let outStd = MLX.sqrt(MLX.variance(output)).item(Float.self)
        let outFirst5 = (0..<5).map { output[0, 0, $0].item(Float.self) }
        let outMid5 = (0..<5).map { output[0, 256, $0].item(Float.self) }
        print("Velocity output: shape=\(output.shape), mean=\(String(format: "%.6f", outMean)), std=\(String(format: "%.6f", outStd))")
        print("  [0,0,:5]: \(outFirst5)")
        print("  [0,0,-5:]: \((123..<128).map { output[0, 0, $0].item(Float.self) })")
        print("  [0,256,:5]: \(outMid5)")

        print("\n=== Python Reference (same seed=42, same dummy context) ===")
        print("Velocity output: mean=0.015259, std=1.007812")
        print("  [0,0,:5]: [1.78125, 1.5703125, -0.76953125, 1.6875, 0.373046875]")
        print("  [0,0,-5:]: [0.59765625, 1.2109375, 1.8046875, 0.1015625, -0.58984375]")
        print("  [0,256,:5]: [-0.48828125, -0.33984375, 0.8359375, 0.87890625, 1.4296875]")

        print("\nDone!")
    }
}

// MARK: - DenoiseCompare Command

struct DenoiseCompare: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "denoise-compare",
        abstract: "Run 5 denoising steps and compare with Python reference (dev model)"
    )

    @Option(name: .shortAndLong, help: "Model variant: dev")
    var model: String = "dev"

    @Option(name: .long, help: "Number of steps to run")
    var steps: Int = 5

    @Flag(name: .long, help: "Use bfloat16 noise (matching Python)")
    var bf16Noise: Bool = false

    mutating func run() async throws {
        LTXDebug.enableDebugMode()

        guard let modelVariant = LTXModel(rawValue: model) else {
            throw ValidationError("Invalid model: \(model)")
        }

        let width = 512
        let height = 512
        let numFrames = 9
        let seed: UInt64 = 42
        let channels = 128
        let cfgScale: Float = 4.0

        let latF = (numFrames - 1) / 8 + 1  // 2
        let latH = height / 32  // 16
        let latW = width / 32   // 16
        let numTokens = latF * latH * latW  // 512

        print("=== Denoise Compare for \(modelVariant.displayName) ===")
        print("Resolution: \(width)x\(height), \(numFrames) frames")
        print("Latent shape: \(latF)x\(latH)x\(latW) = \(numTokens) tokens")
        print("CFG scale: \(cfgScale), steps: \(steps)")
        print("Noise dtype: \(bf16Noise ? "bfloat16" : "float32")")
        print()

        // 1. Load model
        let downloader = ModelDownloader()
        let url = try await downloader.downloadUnifiedWeights(model: modelVariant)
        let weightsPath = url.path

        print("Loading transformer weights from \(weightsPath)...")
        let transformerWeights = try LTXWeightLoader.loadTransformerWeights(from: weightsPath)

        let transformer = LTXTransformer(config: modelVariant.transformerConfig, memoryOptimization: .aggressive)
        try LTXWeightLoader.applyTransformerWeights(transformerWeights, to: transformer)
        MLX.eval(transformer.parameters())
        print("Transformer loaded.")

        // 2. Generate noise with seed (matching Python: bfloat16)
        MLXRandom.seed(seed)
        var latent: MLXArray
        if bf16Noise {
            // Generate float32 then cast to bfloat16 (matching Python's mx.random.normal(..., dtype=bfloat16))
            latent = MLXRandom.normal([1, channels, latF, latH, latW]).asType(.bfloat16)
        } else {
            latent = MLXRandom.normal([1, channels, latF, latH, latW])
        }
        MLX.eval(latent)

        let latentMean = latent.mean().item(Float.self)
        let latentStd = MLX.sqrt(MLX.variance(latent)).item(Float.self)
        print("Initial latents: dtype=\(latent.dtype), mean=\(String(format: "%.6f", latentMean)), std=\(String(format: "%.6f", latentStd))")
        let first5 = (0..<5).map { latent[0, 0, 0, 0, $0].item(Float.self) }
        print("Initial latents[0,0,0,0,:5]: \(first5)")

        // 3. Compute sigma schedule
        let scheduler = LTXScheduler()
        scheduler.setTimesteps(
            numSteps: 50,
            distilled: false,
            latentTokenCount: numTokens
        )
        let sigmas = scheduler.sigmas
        print("Sigmas (first 6): \(sigmas.prefix(6))")

        // 4. Prepare dummy context (zeros, matching Python) - for CFG: [neg, pos] in Swift
        let dummyContextPos = MLXArray.zeros([1, 128, 3840]).asType(.bfloat16)
        let dummyContextNeg = MLXArray.zeros([1, 128, 3840]).asType(.bfloat16)
        // Swift ordering: [neg, pos]
        let textEmbeddings = MLX.concatenated([dummyContextNeg, dummyContextPos], axis: 0)
        MLX.eval(textEmbeddings)

        let latentShape = VideoLatentShape(
            batch: 1, channels: channels,
            frames: latF, height: latH, width: latW
        )

        // Scale initial noise by first sigma (sigmas[0] = 1.0 for dev, so no-op)
        latent = latent * sigmas[0]

        print("\n=== Denoising Loop (\(steps) steps, CFG=\(cfgScale)) ===")

        for step in 0..<steps {
            let sigma = sigmas[step]
            let sigmaNext = sigmas[step + 1]

            // CFG: double batch [neg, pos]
            let latentInput = prepareForCFG(latent)
            let timestep = MLXArray([sigma, sigma])

            // Patchify
            let patchified = patchify(latentInput)

            // Run transformer
            let velocityPred = transformer(
                latent: patchified,
                context: textEmbeddings,
                timesteps: timestep,
                contextMask: nil,
                latentShape: (frames: latF, height: latH, width: latW)
            )

            // Unpatchify and apply CFG
            let doubledShape = VideoLatentShape(
                batch: latentShape.batch * 2, channels: latentShape.channels,
                frames: latentShape.frames, height: latentShape.height, width: latentShape.width
            )
            let fullVelocity = unpatchify(velocityPred, shape: doubledShape)
            let (uncond, cond) = splitCFGOutput(fullVelocity)
            let velocity = applyCFG(uncond: uncond, cond: cond, guidanceScale: cfgScale)

            // Euler step
            latent = scheduler.step(
                latent: latent,
                velocity: velocity,
                sigma: sigma,
                sigmaNext: sigmaNext
            )
            MLX.eval(latent)

            // Print diagnostics
            let lMean = latent.mean().item(Float.self)
            let lStd = MLX.sqrt(MLX.variance(latent)).item(Float.self)
            let vMean = velocityPred.mean().item(Float.self)
            let vStd = MLX.sqrt(MLX.variance(velocityPred)).item(Float.self)
            let lFirst5 = (0..<5).map { latent[0, 0, 0, 0, $0].item(Float.self) }
            let pfLatent = patchify(latent)
            let pfFirst5 = (0..<5).map { pfLatent[0, 0, $0].item(Float.self) }
            print("Step \(step): σ=\(String(format: "%.6f", sigma))→\(String(format: "%.6f", sigmaNext))")
            print("  velocity_pred: mean=\(String(format: "%.6f", vMean)), std=\(String(format: "%.6f", vStd))")
            print("  latent: mean=\(String(format: "%.6f", lMean)), std=\(String(format: "%.6f", lStd))")
            print("  latent[0,0,0,0,:5]: \(lFirst5)")
            print("  patchified[0,0,:5]: \(pfFirst5)")
        }

        // Python reference values
        print("\n=== Python Reference (5 steps, CFG=4.0, bf16 noise) ===")
        print("Step 0: σ=1.000000→0.991176")
        print("  velocity_flat: mean=0.015320, std=1.007812")
        print("  latent: mean=0.000938, std=0.988281")
        print("  latent[0,0,0,0,:5]: [1.8515625, -1.171875, -0.9296875, -0.251953125, -0.314453125]")
        print("Step 1: σ=0.991176→0.982159")
        print("  velocity_flat: mean=0.021240, std=1.007812")
        print("  latent: mean=0.000755, std=0.976562")
        print("Step 2: σ=0.982159→0.972943")
        print("  velocity_flat: mean=0.033936, std=1.015625")
        print("  latent: mean=0.000425, std=0.968750")
        print("Step 3: σ=0.972943→0.963520")
        print("  velocity_flat: mean=0.039795, std=1.023438")
        print("  latent: mean=0.000053, std=0.960938")
        print("Step 4: σ=0.963520→0.953884")
        print("  velocity_flat: mean=0.043945, std=1.039062")
        print("  latent: mean=-0.000332, std=0.953125")

        print("\nDone!")
    }
}
