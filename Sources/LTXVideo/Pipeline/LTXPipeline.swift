// LTXPipeline.swift - Main Video Generation Pipeline for LTX-2
// Copyright 2025

import CoreGraphics
import Foundation
@preconcurrency import MLX
import MLXRandom
import MLXNN
import Tokenizers
import Hub

// MARK: - Default Negative Prompt (matches Python DEFAULT_NEGATIVE_PROMPT)

/// Default negative prompt used for CFG unconditional conditioning in the dev pipeline.
/// This matches the Python mlx-video `DEFAULT_NEGATIVE_PROMPT` exactly.
public let DEFAULT_NEGATIVE_PROMPT = """
blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, \
grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, \
deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, \
wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of \
field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent \
lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny \
valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, \
mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, \
off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward \
pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, \
inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts.
"""

// MARK: - Pipeline Progress

/// Progress information during generation
public struct GenerationProgress: Sendable {
    /// Current step (0-indexed)
    public let currentStep: Int

    /// Total number of steps
    public let totalSteps: Int

    /// Current sigma value
    public let sigma: Float

    /// Progress fraction (0.0 to 1.0)
    public var progress: Double {
        Double(currentStep + 1) / Double(totalSteps)
    }

    /// Human-readable status
    public var status: String {
        "Step \(currentStep + 1)/\(totalSteps) (σ=\(String(format: "%.4f", sigma)))"
    }
}

/// Callback type for generation progress
public typealias GenerationProgressCallback = @Sendable (GenerationProgress) -> Void

/// Callback type for intermediate frame preview
public typealias FramePreviewCallback = @Sendable (Int, CGImage) -> Void

// MARK: - LTX Pipeline

/// Main pipeline for LTX-2 video generation
public actor LTXPipeline {
    // MARK: - Properties

    /// Model variant being used
    public let model: LTXModel

    /// Quantization configuration
    public let quantization: LTXQuantizationConfig

    /// Memory optimization configuration
    public let memoryOptimization: MemoryOptimizationConfig

    /// Model downloader
    private let downloader: ModelDownloader

    /// Flow-matching scheduler
    private let scheduler: LTXScheduler

    /// Gemma 3 language model for text encoding
    private var gemmaModel: Gemma3TextModel?

    /// Tokenizer for Gemma
    private var tokenizer: Tokenizer?

    /// Text encoder (feature extractor + connector)
    private var textEncoder: VideoGemmaTextEncoderModel?

    /// Diffusion transformer
    private var transformer: LTXTransformer?

    /// VAE decoder
    private var vaeDecoder: VideoDecoder?

    /// Cached null embeddings (encoding of empty string "")
    private var cachedNullEmbeddings: (encoding: MLXArray, mask: MLXArray)?

    /// Whether models are loaded (Gemma may be nil after unloading post-encoding)
    public var isLoaded: Bool {
        textEncoder != nil && transformer != nil && vaeDecoder != nil
    }

    /// Whether Gemma model is available for text encoding
    public var isGemmaLoaded: Bool {
        gemmaModel != nil && tokenizer != nil
    }

    // MARK: - Initialization

    public init(
        model: LTXModel = .distilled,
        quantization: LTXQuantizationConfig = .default,
        memoryOptimization: MemoryOptimizationConfig = .default,
        hfToken: String? = nil
    ) {
        self.model = model
        self.quantization = quantization
        self.memoryOptimization = memoryOptimization
        self.downloader = ModelDownloader(hfToken: hfToken)
        self.scheduler = LTXScheduler(isDistilled: model.isDistilled)
    }

    // MARK: - Model Loading

    /// Load all models required for generation
    ///
    /// Downloads and loads:
    /// 1. Gemma 3 12B (text encoder backbone) — from `mlx-community/gemma-3-12b-it-4bit`
    /// 2. LTX-2 unified weights (transformer + VAE + connector) — from `Lightricks/LTX-2`
    ///
    /// The LTX weights are loaded from a single safetensors file matching the Python
    /// LTX-2-MLX reference implementation.
    ///
    /// - Parameters:
    ///   - progressCallback: Optional callback for download/load progress
    ///   - gemmaModelPath: Optional local path to Gemma model (auto-downloads if nil)
    ///   - ltxWeightsPath: Optional local path to unified LTX weights file (auto-downloads if nil)
    public func loadModels(
        progressCallback: DownloadProgressCallback? = nil,
        gemmaModelPath: String? = nil,
        tokenizerPath: String? = nil,
        ltxWeightsPath: String? = nil
    ) async throws {
        LTXDebug.log("Loading models for \(model.displayName)...")
        var stepStart = Date()

        // Step 1: Load Gemma model and tokenizer
        // For dev model: uses the bundled text_encoder from LTX-2-dev-bf16 (float32, ~48.7GB)
        // This matches the Python baseline exactly.
        progressCallback?(DownloadProgress(progress: 0.1, message: "Loading Gemma model..."))

        let gemmaURL: URL
        let tokenizerURL: URL
        if let gemmaPath = gemmaModelPath {
            gemmaURL = URL(fileURLWithPath: gemmaPath)
            tokenizerURL = tokenizerPath.map { URL(fileURLWithPath: $0) } ?? gemmaURL
        } else {
            LTXDebug.log("Downloading Gemma text encoder for \(model.displayName) (if needed)...")
            let paths = try await downloader.downloadGemma(model: model) { progress in
                progressCallback?(progress)
            }
            gemmaURL = paths.modelDir
            tokenizerURL = paths.tokenizerDir
        }
        LTXDebug.log("[TIME] Gemma download check: \(String(format: "%.1f", Date().timeIntervalSince(stepStart)))s")

        stepStart = Date()
        LTXDebug.log("Loading Gemma3 model from \(gemmaURL.path)...")
        gemmaModel = try Gemma3WeightLoader.loadModel(from: gemmaURL)
        LTXDebug.log("[TIME] Gemma load: \(String(format: "%.1f", Date().timeIntervalSince(stepStart)))s — \(gemmaModel!.config.hiddenLayers) layers")

        stepStart = Date()
        progressCallback?(DownloadProgress(progress: 0.2, message: "Loading tokenizer..."))
        LTXDebug.log("Loading tokenizer from \(tokenizerURL.path)...")
        tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerURL)
        LTXDebug.log("[TIME] Tokenizer load: \(String(format: "%.1f", Date().timeIntervalSince(stepStart)))s")

        // Step 2: Download/load LTX component weights
        progressCallback?(DownloadProgress(progress: 0.3, message: "Loading LTX-2 weights..."))

        let transformerWeights: [String: MLXArray]
        let vaeWeights: [String: MLXArray]
        let connectorWeights: [String: MLXArray]
        var vaeConfigPath: URL? = nil  // Path to vae/config.json (if downloaded)

        if let path = ltxWeightsPath {
            // Unified file provided locally — split into components
            stepStart = Date()
            LTXDebug.log("Splitting unified weights from \(path)...")
            let split = try LTXWeightLoader.splitUnifiedWeightsFile(path: path)
            transformerWeights = split.transformer
            vaeWeights = split.vae
            connectorWeights = split.connector
            LTXDebug.log("[TIME] Split unified weights: \(String(format: "%.1f", Date().timeIntervalSince(stepStart)))s")
        } else {
            // Per-component downloads from Lightricks/LTX-2
            // Connector and VAE are shared between dev and distilled.
            // Only the transformer weights differ (via unifiedWeightsFilename).
            stepStart = Date()
            LTXDebug.log("Downloading LTX-2 components for \(model.displayName) (if needed)...")

            // Download connector (shared between dev and distilled)
            progressCallback?(DownloadProgress(progress: 0.35, message: "Downloading connector weights..."))
            let connectorPath = try await downloader.downloadConnector(model: model) { progress in
                progressCallback?(progress)
            }
            connectorWeights = try LTXWeightLoader.loadConnectorWeights(from: connectorPath.path)

            // Download transformer from unified file (dev or distilled)
            progressCallback?(DownloadProgress(progress: 0.4, message: "Downloading transformer weights..."))
            let unifiedPath = try await downloader.downloadUnifiedWeights(model: model) { progress in
                progressCallback?(progress)
            }
            transformerWeights = try LTXWeightLoader.loadTransformerWeights(from: unifiedPath.path)

            // Download VAE (shared between dev and distilled)
            progressCallback?(DownloadProgress(progress: 0.8, message: "Downloading VAE weights..."))
            let vaePath = try await downloader.downloadVAE(model: model) { progress in
                progressCallback?(progress)
            }
            vaeWeights = try LTXWeightLoader.loadVAEWeights(from: vaePath.path)
            vaeConfigPath = vaePath

            LTXDebug.log("[TIME] Component downloads: \(String(format: "%.1f", Date().timeIntervalSince(stepStart)))s")
        }

        // Step 3: Create and load transformer
        progressCallback?(DownloadProgress(progress: 0.5, message: "Loading transformer..."))

        let transformerConfig = model.transformerConfig
        transformer = LTXTransformer(config: transformerConfig, memoryOptimization: memoryOptimization)

        stepStart = Date()
        LTXDebug.log("Applying \(transformerWeights.count) transformer weights...")
        try LTXWeightLoader.applyTransformerWeights(transformerWeights, to: transformer!)
        LTXDebug.log("[TIME] Apply transformer weights: \(String(format: "%.1f", Date().timeIntervalSince(stepStart)))s")

        // Evaluate transformer weights to ensure they're fully materialized
        stepStart = Date()
        eval(transformer!.parameters())
        LTXDebug.log("[TIME] Eval transformer weights: \(String(format: "%.1f", Date().timeIntervalSince(stepStart)))s")

        // Step 4: Create and load VAE decoder
        progressCallback?(DownloadProgress(progress: 0.7, message: "Loading VAE decoder..."))

        vaeDecoder = VideoDecoder()
        // Read timestep_conditioning from VAE config.json (defaults to false if not found)
        if let vaeConfigPath = vaeConfigPath {
            vaeDecoder!.timestepConditioning = LTXWeightLoader.parseVAEConfig(from: vaeConfigPath)
        }

        stepStart = Date()
        LTXDebug.log("Applying \(vaeWeights.count) VAE weights...")
        try LTXWeightLoader.applyVAEWeights(vaeWeights, to: vaeDecoder!)
        LTXDebug.log("[TIME] VAE load: \(String(format: "%.1f", Date().timeIntervalSince(stepStart)))s")

        // Step 5: Create and load text encoder (connector)
        progressCallback?(DownloadProgress(progress: 0.9, message: "Loading text encoder..."))

        textEncoder = VideoGemmaTextEncoderModel()

        stepStart = Date()
        LTXDebug.log("Applying \(connectorWeights.count) text encoder weights...")
        try LTXWeightLoader.applyTextEncoderWeights(connectorWeights, to: textEncoder!)
        LTXDebug.log("[TIME] TextEncoder load: \(String(format: "%.1f", Date().timeIntervalSince(stepStart)))s")

        progressCallback?(DownloadProgress(progress: 1.0, message: "Models loaded successfully"))
        LTXDebug.log("All models loaded successfully")
    }

    /// Load only the text encoding models (Gemma + tokenizer + connector).
    /// Use this for standalone text encoding without loading the heavy transformer and VAE.
    public func loadTextEncoderModels(
        progressCallback: DownloadProgressCallback? = nil,
        gemmaModelPath: String? = nil,
        tokenizerPath: String? = nil
    ) async throws {
        LTXDebug.log("Loading text encoder models for \(model.displayName)...")
        var stepStart = Date()

        // Step 1: Load Gemma model and tokenizer
        progressCallback?(DownloadProgress(progress: 0.1, message: "Loading Gemma model..."))

        let gemmaURL: URL
        let tokenizerURL: URL
        if let gemmaPath = gemmaModelPath {
            gemmaURL = URL(fileURLWithPath: gemmaPath)
            tokenizerURL = tokenizerPath.map { URL(fileURLWithPath: $0) } ?? gemmaURL
        } else {
            LTXDebug.log("Downloading Gemma text encoder (if needed)...")
            let paths = try await downloader.downloadGemma(model: model) { progress in
                progressCallback?(progress)
            }
            gemmaURL = paths.modelDir
            tokenizerURL = paths.tokenizerDir
        }

        stepStart = Date()
        LTXDebug.log("Loading Gemma3 model from \(gemmaURL.path)...")
        gemmaModel = try Gemma3WeightLoader.loadModel(from: gemmaURL)
        LTXDebug.log("[TIME] Gemma load: \(String(format: "%.1f", Date().timeIntervalSince(stepStart)))s")

        stepStart = Date()
        progressCallback?(DownloadProgress(progress: 0.5, message: "Loading tokenizer..."))
        tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerURL)
        LTXDebug.log("[TIME] Tokenizer load: \(String(format: "%.1f", Date().timeIntervalSince(stepStart)))s")

        // Step 2: Download and load connector
        progressCallback?(DownloadProgress(progress: 0.7, message: "Loading connector weights..."))
        stepStart = Date()
        let connectorPath = try await downloader.downloadConnector(model: model) { progress in
            progressCallback?(progress)
        }
        let connectorWeights = try LTXWeightLoader.loadConnectorWeights(from: connectorPath.path)

        textEncoder = VideoGemmaTextEncoderModel()
        try LTXWeightLoader.applyTextEncoderWeights(connectorWeights, to: textEncoder!)
        LTXDebug.log("[TIME] Connector load: \(String(format: "%.1f", Date().timeIntervalSince(stepStart)))s")

        progressCallback?(DownloadProgress(progress: 1.0, message: "Text encoder models loaded"))
        LTXDebug.log("Text encoder models loaded successfully")
    }

    // MARK: - Video Generation

    /// Generate video from text prompt
    ///
    /// - Parameters:
    ///   - prompt: Text description of the video
    ///   - negativePrompt: Optional negative prompt for CFG
    ///   - config: Video generation configuration
    ///   - onProgress: Optional progress callback
    ///   - onFrame: Optional callback for intermediate frame preview
    /// - Returns: Video generation result with frames
    /// Pre-computed text embeddings for diagnostic/cross-validation
    public struct PrecomputedEmbeddings {
        public let promptEmbeddings: MLXArray  // [1, T, 3840]
        public let promptMask: MLXArray        // [1, T]
        public let nullEmbeddings: MLXArray?   // [1, T, 3840] for CFG
        public let nullMask: MLXArray?         // [1, T]

        public init(promptEmbeddings: MLXArray, promptMask: MLXArray,
                     nullEmbeddings: MLXArray? = nil, nullMask: MLXArray? = nil) {
            self.promptEmbeddings = promptEmbeddings
            self.promptMask = promptMask
            self.nullEmbeddings = nullEmbeddings
            self.nullMask = nullMask
        }
    }

    public func generateVideo(
        prompt: String,
        negativePrompt: String? = nil,
        config: LTXVideoGenerationConfig,
        precomputedEmbeddings: PrecomputedEmbeddings? = nil,
        onProgress: GenerationProgressCallback? = nil,
        onFrame: FramePreviewCallback? = nil,
        profile: Bool = false
    ) async throws -> VideoGenerationResult {
        // Validate configuration
        try config.validate()
        var timings = GenerationTimings()

        guard isLoaded else {
            throw LTXError.modelNotLoaded("Models not loaded. Call loadModels() first.")
        }

        guard let textEncoder = textEncoder,
            let transformer = transformer,
            let vaeDecoder = vaeDecoder
        else {
            throw LTXError.modelNotLoaded("One or more models failed to load")
        }

        let generationStart = Date()

        LTXDebug.log("Generating video: \(config.width)x\(config.height), \(config.numFrames) frames")
        LTXDebug.log("Prompt: \(prompt)")

        let textEmbeddings: MLXArray
        let contextMask: MLXArray
        let useCFG = config.cfgScale > 1.0
        let textEncStart = Date()

        if let precomputed = precomputedEmbeddings {
            // Use pre-computed embeddings (for diagnostic/cross-validation)
            // Cast to bfloat16 to match what the text encoder normally produces
            LTXDebug.log("Using pre-computed embeddings (bypassing text encoder)")
            let pe = precomputed.promptEmbeddings.asType(.bfloat16)
            let pm = precomputed.promptMask.asType(.int32)

            // Log injected embedding stats for verification
            do {
                let pMean = pe.mean().item(Float.self)
                let pStd = MLX.sqrt(MLX.variance(pe)).item(Float.self)
                LTXDebug.log("[DIAG] injected pos emb: mean=\(String(format: "%.8f", pMean)), std=\(String(format: "%.8f", pStd))")
                let first5 = (0..<5).map { i in
                    String(format: "%.6f", pe[0, 0, i].item(Float.self))
                }
                LTXDebug.log("[DIAG] injected pos emb[0,0,:5] = [\(first5.joined(separator: ", "))]")
            }

            if useCFG {
                let ne = (precomputed.nullEmbeddings ?? MLXArray.zeros(like: pe)).asType(.bfloat16)
                let nm = (precomputed.nullMask ?? MLXArray.zeros(like: pm)).asType(.int32)

                // Log injected neg embedding stats
                do {
                    let nMean = ne.mean().item(Float.self)
                    let nStd = MLX.sqrt(MLX.variance(ne)).item(Float.self)
                    LTXDebug.log("[DIAG] injected neg emb: mean=\(String(format: "%.8f", nMean)), std=\(String(format: "%.8f", nStd))")
                }

                textEmbeddings = MLX.concatenated([ne, pe], axis: 0)
                contextMask = MLX.concatenated([nm, pm], axis: 0)
            } else {
                textEmbeddings = pe
                contextMask = pm
            }
            MLX.eval(textEmbeddings, contextMask)
            LTXDebug.log("textEmbeddings shape: \(textEmbeddings.shape)")
        } else {
            // 0a. Reload Gemma if it was unloaded from a previous generation
            if !isGemmaLoaded {
                LTXDebug.log("Reloading Gemma model (was unloaded after previous generation)...")
                let gemmaDir = downloader.cacheDirectory.appendingPathComponent("ltx-\(model.rawValue)-text-encoder")
                let tokenizerDir = downloader.cacheDirectory.appendingPathComponent("ltx-\(model.rawValue)-tokenizer")
                if FileManager.default.fileExists(atPath: gemmaDir.path) {
                    gemmaModel = try Gemma3WeightLoader.loadModel(from: gemmaDir)
                    tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerDir)
                } else {
                    let paths = try await downloader.downloadGemma(model: model, progress: nil)
                    gemmaModel = try Gemma3WeightLoader.loadModel(from: paths.modelDir)
                    tokenizer = try await AutoTokenizer.from(modelFolder: paths.tokenizerDir)
                }
                LTXDebug.log("Gemma model reloaded")
            }

            // 0b. Optionally enhance prompt using Gemma generation
            let effectivePrompt: String
            if config.enhancePrompt {
                LTXDebug.log("Enhancing prompt with Gemma...")
                effectivePrompt = enhancePrompt(prompt)
                LTXDebug.log("Using enhanced prompt for generation")
            } else {
                effectivePrompt = prompt
            }

            // 1. Encode text prompt
            LTXDebug.log("Encoding text prompt... (CFG=\(config.cfgScale), enabled=\(useCFG))")
            let (promptEmbeddings, promptMask) = encodePrompt(effectivePrompt, encoder: textEncoder)
            MLX.eval(promptEmbeddings, promptMask)
            LTXDebug.log("promptEmbeddings shape: \(promptEmbeddings.shape)")

            // Log positive embedding stats + first values for element-wise comparison with Python
            do {
                let pMean = promptEmbeddings.mean().item(Float.self)
                let pStd = MLX.sqrt(MLX.variance(promptEmbeddings)).item(Float.self)
                LTXDebug.log("[DIAG] pos emb: mean=\(String(format: "%.8f", pMean)), std=\(String(format: "%.8f", pStd))")
                // Print first 5 values for element-wise comparison
                let first5 = (0..<5).map { i in
                    String(format: "%.6f", promptEmbeddings[0, 0, i].item(Float.self))
                }
                LTXDebug.log("[DIAG] pos emb[0,0,:5] = [\(first5.joined(separator: ", "))]")
                let mid5 = (0..<5).map { i in
                    String(format: "%.6f", promptEmbeddings[0, 512, i].item(Float.self))
                }
                LTXDebug.log("[DIAG] pos emb[0,512,:5] = [\(mid5.joined(separator: ", "))]")
            }

            if useCFG {
                // CFG mode: stack [negative, positive] embeddings for doubled batch
                let negPromptText = negativePrompt ?? DEFAULT_NEGATIVE_PROMPT
                LTXDebug.log("Encoding negative prompt for CFG (\(negPromptText.prefix(60))...)")
                let (negEmbeddings, negMask) = encodePrompt(negPromptText, encoder: textEncoder)
                MLX.eval(negEmbeddings, negMask)

                // Log negative embedding stats
                do {
                    let nMean = negEmbeddings.mean().item(Float.self)
                    let nStd = MLX.sqrt(MLX.variance(negEmbeddings)).item(Float.self)
                    LTXDebug.log("[DIAG] neg emb: mean=\(String(format: "%.8f", nMean)), std=\(String(format: "%.8f", nStd))")
                }

                textEmbeddings = MLX.concatenated([negEmbeddings, promptEmbeddings], axis: 0)
                contextMask = MLX.concatenated([negMask, promptMask], axis: 0)
                MLX.eval(textEmbeddings, contextMask)
                LTXDebug.log("textEmbeddings shape (CFG): \(textEmbeddings.shape)")
            } else {
                // No CFG: single batch, just the prompt embeddings
                textEmbeddings = promptEmbeddings
                contextMask = promptMask
                LTXDebug.log("textEmbeddings shape (no CFG): \(textEmbeddings.shape)")
            }

            // 1b. Unload Gemma model to free memory for transformer denoising
            LTXDebug.log("Unloading Gemma model to free memory...")
            self.gemmaModel = nil
            self.tokenizer = nil
            Memory.clearCache()
            LTXDebug.log("Gemma model unloaded")
        }
        LTXDebug.log("Text encoding: \(String(format: "%.1f", Date().timeIntervalSince(textEncStart)))s")

        // 2. Create latent shape
        let latentShape = VideoLatentShape.fromPixelDimensions(
            batch: 1,
            channels: 128,
            frames: config.numFrames,
            height: config.height,
            width: config.width
        )

        LTXDebug.log(
            "Latent shape: \(latentShape.frames)x\(latentShape.height)x\(latentShape.width)")

        // 3. Generate initial noise
        if let seed = config.seed {
            MLXRandom.seed(seed)
            LTXDebug.log("Using seed: \(seed)")
        }

        var latent = generateNoise(shape: latentShape, seed: config.seed)
        LTXDebug.log("Initial latent shape: \(latent.shape)")
        MLX.eval(latent)

        // Log initial noise stats for Python comparison
        do {
            let nMean = latent.mean().item(Float.self)
            let nStd = MLX.sqrt(MLX.variance(latent)).item(Float.self)
            LTXDebug.log("[DIAG] Initial noise: mean=\(String(format: "%.8f", nMean)), std=\(String(format: "%.8f", nStd))")
        }

        // 3b. Apply cross-attention scaling if configured
        if config.crossAttentionScale != 1.0 {
            transformer.setCrossAttentionScale(config.crossAttentionScale)
            LTXDebug.log("Cross-attention scale set to \(config.crossAttentionScale)")
        }

        // 4. Get sigma schedule
        let sigmas: [Float]
        if model.isDistilled {
            sigmas = scheduler.getSigmas(numSteps: config.numSteps)
        } else {
            // Dev model: use token-dependent shift
            scheduler.setTimesteps(
                numSteps: config.numSteps,
                distilled: false,
                latentTokenCount: latentShape.tokenCount
            )
            sigmas = scheduler.sigmas
        }
        LTXDebug.log("Sigma schedule: \(sigmas.map { String(format: "%.4f", $0) })")

        // 5. Scale initial noise by first sigma
        latent = latent * sigmas[0]

        // 6. Denoising loop
        LTXDebug.log("Starting denoising loop (\(config.numSteps) steps)...")
        var previousVelocity: MLXArray? = nil  // For GE velocity correction

        for step in 0..<config.numSteps {
            let stepStart = Date()
            let sigma = sigmas[step]
            let sigmaNext = sigmas[step + 1]

            onProgress?(
                GenerationProgress(
                    currentStep: step,
                    totalSteps: config.numSteps,
                    sigma: sigma
                ))

            let timestep = MLXArray([sigma])

            // Patchify for transformer — cast to bfloat16 matching Diffusers latent_model_input.to(dtype)
            let patchified = patchify(latent).asType(.bfloat16)
            LTXDebug.log("Step \(step): patchified \(patchified.shape), σ=\(String(format: "%.4f", sigma))")

            // Unpatchify velocity back to (B, C, F, H, W)
            var velocity: MLXArray
            if useCFG {
                // Separate forward passes matching Diffusers (cfg_batch=False)
                // Extract positive and negative embeddings + masks
                let posEmbeddings = textEmbeddings[1..<2]  // [neg, pos] → pos
                let negEmbeddings = textEmbeddings[0..<1]  // [neg, pos] → neg
                let posMask = contextMask[1..<2]
                let negMask = contextMask[0..<1]

                // Positive pass (conditional)
                let velPosPred = transformer(
                    latent: patchified,
                    context: posEmbeddings,
                    timesteps: timestep,
                    contextMask: posMask,
                    latentShape: (frames: latentShape.frames, height: latentShape.height, width: latentShape.width)
                )
                let velPos = unpatchify(velPosPred, shape: latentShape)
                MLX.eval(velPos)  // Eval to free transformer graph — prevents memory thrashing

                // Negative pass (unconditional)
                let velNegPred = transformer(
                    latent: patchified,
                    context: negEmbeddings,
                    timesteps: timestep,
                    contextMask: negMask,
                    latentShape: (frames: latentShape.frames, height: latentShape.height, width: latentShape.width)
                )
                let velNeg = unpatchify(velNegPred, shape: latentShape)
                MLX.eval(velNeg)

                // Detailed step-0 CFG diagnostics
                if step == 0 && profile {
                    let condMean = velPos.mean().item(Float.self)
                    let condStd = MLX.sqrt(MLX.variance(velPos)).item(Float.self)
                    let uncondMean = velNeg.mean().item(Float.self)
                    let uncondStd = MLX.sqrt(MLX.variance(velNeg)).item(Float.self)
                    LTXDebug.log("[DIAG] Step 0 vel_cond (pos): mean=\(String(format: "%.8f", condMean)), std=\(String(format: "%.8f", condStd))")
                    LTXDebug.log("[DIAG] Step 0 vel_uncond (neg): mean=\(String(format: "%.8f", uncondMean)), std=\(String(format: "%.8f", uncondStd))")
                }

                // Cast to float32 before CFG (matching Diffusers noise_pred_video.float())
                let velPosF32 = velPos.asType(.float32)
                let velNegF32 = velNeg.asType(.float32)

                // Apply CFG in float32: vel_pos + (cfg_scale - 1) * (vel_pos - vel_neg)
                velocity = applyCFG(uncond: velNegF32, cond: velPosF32, guidanceScale: config.cfgScale)

                // Log CFG velocity stats at step 0
                if step == 0 && profile {
                    let cfgMean = velocity.mean().item(Float.self)
                    let cfgStd = MLX.sqrt(MLX.variance(velocity)).item(Float.self)
                    LTXDebug.log("[DIAG] Step 0 CFG velocity: mean=\(String(format: "%.8f", cfgMean)), std=\(String(format: "%.8f", cfgStd))")
                }

                // Apply guidance rescale to reduce overexposure
                if config.guidanceRescale > 0 {
                    velocity = applyGuidanceRescale(
                        cfgOutput: velocity,
                        condOutput: velPosF32,
                        phi: config.guidanceRescale
                    )
                }
                // velocity stays float32 — scheduler operates in float32
            } else {
                // No CFG: single pass
                let velocityPred = transformer(
                    latent: patchified,
                    context: textEmbeddings,
                    timesteps: timestep,
                    contextMask: contextMask,
                    latentShape: (frames: latentShape.frames, height: latentShape.height, width: latentShape.width)
                )
                // Cast to float32 matching Diffusers noise_pred.float()
                velocity = unpatchify(velocityPred, shape: latentShape).asType(.float32)
            }

            // STG (Spatio-Temporal Guidance): perturbed forward pass with skipped self-attention
            if config.stgScale > 0 {
                // Enable skip flags on specified blocks
                transformer.setSTGSkipFlags(skipSelfAttention: true, blockIndices: config.stgBlocks)

                // Perturbed forward pass (single-batch, conditional only)
                let perturbedPatchified = patchify(latent).asType(.bfloat16)
                let perturbedTimestep = MLXArray([sigma])
                // Extract prompt-only embeddings (second half of CFG-doubled batch)
                let perturbedContext = useCFG ? textEmbeddings[1..<2] : textEmbeddings
                let perturbedMask = useCFG ? contextMask[1..<2] : contextMask
                let perturbedVelocityPred = transformer(
                    latent: perturbedPatchified,
                    context: perturbedContext,
                    timesteps: perturbedTimestep,
                    contextMask: perturbedMask,
                    latentShape: (frames: latentShape.frames, height: latentShape.height, width: latentShape.width)
                )
                let perturbedVelocity = unpatchify(perturbedVelocityPred, shape: latentShape).asType(.float32)

                // Clear skip flags
                transformer.clearSTGSkipFlags()

                // Apply STG: velocity += stgScale * (velocity - perturbedVelocity)
                velocity = velocity + MLXArray(config.stgScale) * (velocity - perturbedVelocity)
            }

            // GE velocity correction: apply momentum on velocity prediction
            if config.geGamma > 0, let prevVel = previousVelocity {
                velocity = MLXArray(config.geGamma) * (velocity - prevVel) + prevVel
            }
            previousVelocity = velocity

            // Euler step: x_{t-1} = x_t + (sigma_next - sigma) * velocity
            latent = scheduler.step(
                latent: latent,
                velocity: velocity,
                sigma: sigma,
                sigmaNext: sigmaNext
            )

            // Evaluate to free computation graph memory
            MLX.eval(latent)

            // Per-step diagnostics matching Python format
            if profile {
                let vMean = velocity.mean().item(Float.self)
                let vStd = MLX.sqrt(MLX.variance(velocity)).item(Float.self)
                let lMean = latent.mean().item(Float.self)
                let lStd = MLX.sqrt(MLX.variance(latent)).item(Float.self)
                LTXDebug.log("  Step \(step): σ=\(String(format: "%.4f", sigma))→\(String(format: "%.4f", sigmaNext)), vel mean=\(String(format: "%.4f", vMean)), std=\(String(format: "%.4f", vStd)), latent mean=\(String(format: "%.4f", lMean)), std=\(String(format: "%.4f", lStd))")
            }

            timings.denoiseSteps.append(Date().timeIntervalSince(stepStart))
        }

        // Diagnostic: check final latent stats
        do {
            let lMean = latent.mean().item(Float.self)
            let lStd = MLX.sqrt(MLX.variance(latent)).item(Float.self)
            let lMin = latent.min().item(Float.self)
            let lMax = latent.max().item(Float.self)
            LTXDebug.log("[DIAG] Final latent: mean=\(lMean), std=\(lStd), min=\(lMin), max=\(lMax)")
            // Check spatial variance: variance across spatial dims (H, W) for first channel, first frame
            let firstSlice = latent[0, 0, 0]  // (H, W) for batch=0, channel=0, frame=0
            let spatialVar = MLX.variance(firstSlice).item(Float.self)
            LTXDebug.log("[DIAG] Spatial variance (ch0, f0): \(spatialVar)")
        }

        // Diagnostic: check scale_shift_table
        do {
            let sst = transformer.scaleShiftTable
            let sstMean = sst.mean().item(Float.self)
            let sstStd = MLX.sqrt(MLX.variance(sst)).item(Float.self)
            LTXDebug.log("[DIAG] scale_shift_table: mean=\(sstMean), std=\(sstStd)")
            let block0sst = transformer.transformerBlocks[0].scaleShiftTable
            let b0Mean = block0sst.mean().item(Float.self)
            let b0Std = MLX.sqrt(MLX.variance(block0sst)).item(Float.self)
            LTXDebug.log("[DIAG] block0.scale_shift_table: mean=\(b0Mean), std=\(b0Std)")
            // Check q_norm weights (should be non-trivial if loaded from safetensors)
            let qw = transformer.transformerBlocks[0].attn1.qNorm.weight
            let qwMean = qw.mean().item(Float.self)
            let qwStd = MLX.sqrt(MLX.variance(qw)).item(Float.self)
            LTXDebug.log("[DIAG] block0.attn1.q_norm.weight: mean=\(qwMean), std=\(qwStd) (1.0/0.0 = NOT loaded)")
        }

        // 7. Decode latents to video
        // Use timestep_conditioning from VAE config.json (Diffusers default: false)
        let vaeTimestep: Float? = vaeDecoder.timestepConditioning ? 0.05 : nil
        LTXDebug.log("Decoding latents to video... (timestep=\(vaeTimestep.map { String($0) } ?? "nil"))")
        let vaeStart = Date()
        let videoTensor = decodeVideo(latent: latent, decoder: vaeDecoder, timestep: vaeTimestep)
        MLX.eval(videoTensor)
        timings.vaeDecode = Date().timeIntervalSince(vaeStart)
        LTXDebug.log("Decoded video shape: \(videoTensor.shape)")

        // 8. Trim to requested frame count
        // The VAE upsamples temporally by 2x at each stage (2→4→8→16),
        // but LTX-2 uses the formula: pixelFrames = (latentFrames - 1) * 8 + 1
        // So we trim to the exact requested number of frames.
        let trimmedVideo: MLXArray
        if videoTensor.dim(0) > config.numFrames {
            trimmedVideo = videoTensor[0..<config.numFrames]
            LTXDebug.log("Trimmed from \(videoTensor.dim(0)) to \(config.numFrames) frames")
        } else {
            trimmedVideo = videoTensor
        }

        let frames = VideoExporter.tensorToImages(trimmedVideo)
        LTXDebug.log("Generated \(frames.count) frames")

        let generationTime = Date().timeIntervalSince(generationStart)
        let usedSeed = config.seed ?? 0

        return VideoGenerationResult(
            frames: trimmedVideo,
            seed: usedSeed,
            generationTime: generationTime,
            timings: profile ? timings : nil
        )
    }

    /// Generate video with simpler API
    public func generate(
        prompt: String,
        width: Int = 704,
        height: Int = 480,
        numFrames: Int = 121,
        numSteps: Int? = nil,
        guidance: Float? = nil,
        seed: UInt64? = nil,
        negativePrompt: String? = nil,
        onProgress: GenerationProgressCallback? = nil
    ) async throws -> VideoGenerationResult {
        let config = LTXVideoGenerationConfig(
            width: width,
            height: height,
            numFrames: numFrames,
            numSteps: numSteps ?? model.defaultSteps,
            cfgScale: guidance ?? model.defaultGuidance,
            seed: seed,
            negativePrompt: negativePrompt
        )

        return try await generateVideo(
            prompt: prompt,
            negativePrompt: negativePrompt,
            config: config,
            onProgress: onProgress
        )
    }

    // MARK: - Denoising Loop

    /// Core denoising loop — reusable for both single-stage and two-stage generation.
    ///
    /// Takes an initial noisy latent and iteratively denoises it following the sigma schedule.
    /// Returns the raw denoised latent (NOT VAE-decoded).
    ///
    /// - Parameters:
    ///   - latent: Initial noisy latent (B, C, F, H, W) — already scaled by sigmas[0]
    ///   - sigmas: Sigma schedule (including terminal 0.0)
    ///   - textEmbeddings: Text embeddings (possibly doubled for CFG)
    ///   - promptEmbeddings: Conditional-only embeddings (for STG)
    ///   - latentShape: Shape descriptor for the latent
    ///   - config: Generation configuration
    ///   - transformer: The transformer model
    ///   - useCFG: Whether to use classifier-free guidance
    ///   - onProgress: Optional progress callback
    ///   - profile: Enable profiling output
    /// - Returns: Denoised latent (B, C, F, H, W)
    private func denoise(
        latent: MLXArray,
        sigmas: [Float],
        textEmbeddings: MLXArray,
        promptEmbeddings: MLXArray,
        contextMask: MLXArray,
        latentShape: VideoLatentShape,
        config: LTXVideoGenerationConfig,
        transformer: LTXTransformer,
        useCFG: Bool,
        onProgress: GenerationProgressCallback? = nil,
        profile: Bool = false
    ) -> MLXArray {
        let numSteps = sigmas.count - 1
        var currentLatent = latent
        var previousVelocity: MLXArray? = nil

        for step in 0..<numSteps {
            let stepStart = Date()
            let sigma = sigmas[step]
            let sigmaNext = sigmas[step + 1]

            onProgress?(GenerationProgress(
                currentStep: step,
                totalSteps: numSteps,
                sigma: sigma
            ))

            let latentInput: MLXArray
            let timestep: MLXArray

            if useCFG {
                latentInput = prepareForCFG(currentLatent)
                timestep = MLXArray([sigma, sigma])
            } else {
                latentInput = currentLatent
                timestep = MLXArray([sigma])
            }

            let patchified = patchify(latentInput).asType(.bfloat16)
            LTXDebug.log("Step \(step): patchified \(patchified.shape), σ=\(String(format: "%.4f", sigma))")

            let velocityPred = transformer(
                latent: patchified,
                context: textEmbeddings,
                timesteps: timestep,
                contextMask: contextMask,
                latentShape: (frames: latentShape.frames, height: latentShape.height, width: latentShape.width)
            )

            var velocity: MLXArray
            if useCFG {
                let fullVelocity = unpatchify(velocityPred, shape: latentShape.doubled())
                let (uncond, cond) = splitCFGOutput(fullVelocity)
                // Cast to float32 for CFG computation
                velocity = applyCFG(
                    uncond: uncond.asType(.float32),
                    cond: cond.asType(.float32),
                    guidanceScale: config.cfgScale
                )

                if config.guidanceRescale > 0 {
                    velocity = applyGuidanceRescale(
                        cfgOutput: velocity,
                        condOutput: cond.asType(.float32),
                        phi: config.guidanceRescale
                    )
                }
            } else {
                // Cast to float32 matching Diffusers noise_pred.float()
                velocity = unpatchify(velocityPred, shape: latentShape).asType(.float32)
            }

            // STG
            if config.stgScale > 0 {
                transformer.setSTGSkipFlags(skipSelfAttention: true, blockIndices: config.stgBlocks)
                let perturbedPatchified = patchify(currentLatent).asType(.bfloat16)
                let perturbedTimestep = MLXArray([sigma])
                // Extract prompt-only embeddings (second half of CFG-doubled batch)
                let perturbedContext = useCFG ? textEmbeddings[1..<2] : textEmbeddings
                let perturbedMask = useCFG ? contextMask[1..<2] : contextMask
                let perturbedVelocityPred = transformer(
                    latent: perturbedPatchified,
                    context: perturbedContext,
                    timesteps: perturbedTimestep,
                    contextMask: perturbedMask,
                    latentShape: (frames: latentShape.frames, height: latentShape.height, width: latentShape.width)
                )
                let perturbedVelocity = unpatchify(perturbedVelocityPred, shape: latentShape).asType(.float32)
                transformer.clearSTGSkipFlags()
                velocity = velocity + MLXArray(config.stgScale) * (velocity - perturbedVelocity)
            }

            // GE velocity correction
            if config.geGamma > 0, let prevVel = previousVelocity {
                velocity = MLXArray(config.geGamma) * (velocity - prevVel) + prevVel
            }
            previousVelocity = velocity

            // Euler step
            currentLatent = scheduler.step(
                latent: currentLatent,
                velocity: velocity,
                sigma: sigma,
                sigmaNext: sigmaNext
            )
            MLX.eval(currentLatent)

            if profile {
                let vMean = velocityPred.mean().item(Float.self)
                let vStd = MLX.sqrt(MLX.variance(velocityPred)).item(Float.self)
                let lMean = currentLatent.mean().item(Float.self)
                let lStd = MLX.sqrt(MLX.variance(currentLatent)).item(Float.self)
                LTXDebug.log("  Step \(step): σ=\(String(format: "%.4f", sigma))→\(String(format: "%.4f", sigmaNext)), vel mean=\(String(format: "%.4f", vMean)), std=\(String(format: "%.4f", vStd)), latent mean=\(String(format: "%.4f", lMean)), std=\(String(format: "%.4f", lStd))")
                LTXDebug.log("  Step \(step) time: \(String(format: "%.2f", Date().timeIntervalSince(stepStart)))s")
            }
        }

        return currentLatent
    }

    // MARK: - Two-Stage Generation

    /// Generate video using two-stage pipeline (half-res → upscale → refine)
    ///
    /// Matches Blaizzy/mlx-video approach:
    /// - Stage 1: Denoise at half resolution (8 steps with STAGE_1_SIGMAS)
    /// - Upscale: Denormalize → SpatialUpscaler 2x → Renormalize
    /// - Stage 2: Add noise → Denoise at full resolution (3 steps with STAGE_2_SIGMAS)
    /// - Same transformer for both stages — no LoRA needed
    ///
    /// - Parameters:
    ///   - prompt: Text description of the video
    ///   - config: Video generation configuration (width/height = FINAL resolution)
    ///   - upscalerWeightsPath: Path to spatial upscaler safetensors
    ///   - onProgress: Optional progress callback
    ///   - profile: Enable performance profiling
    /// - Returns: Video generation result at full resolution
    public func generateVideoTwoStage(
        prompt: String,
        config: LTXVideoGenerationConfig,
        upscalerWeightsPath: String,
        onProgress: GenerationProgressCallback? = nil,
        profile: Bool = false
    ) async throws -> VideoGenerationResult {
        try config.validate()

        guard isLoaded else {
            throw LTXError.modelNotLoaded("Models not loaded. Call loadModels() first.")
        }
        guard let textEncoder = textEncoder,
              let transformer = transformer,
              let vaeDecoder = vaeDecoder
        else {
            throw LTXError.modelNotLoaded("One or more models failed to load")
        }

        let generationStart = Date()

        // Two-stage requires width/height divisible by 64
        guard config.width % 64 == 0 && config.height % 64 == 0 else {
            throw LTXError.invalidConfiguration("Two-stage requires width and height divisible by 64. Got \(config.width)x\(config.height)")
        }

        let halfWidth = config.width / 2
        let halfHeight = config.height / 2

        LTXDebug.log("Two-stage generation: \(halfWidth)x\(halfHeight) → \(config.width)x\(config.height)")

        // 0. Optionally enhance prompt
        let effectivePrompt: String
        if config.enhancePrompt {
            LTXDebug.log("Enhancing prompt with Gemma...")
            effectivePrompt = enhancePrompt(prompt)
        } else {
            effectivePrompt = prompt
        }

        // 1. Encode text (shared between both stages)
        let textEncStart = Date()
        let useCFG = config.cfgScale > 1.0
        let (promptEmbeddings, promptMask) = encodePrompt(effectivePrompt, encoder: textEncoder)
        MLX.eval(promptEmbeddings, promptMask)

        let textEmbeddings: MLXArray
        let contextMask: MLXArray
        if useCFG {
            // Use DEFAULT_NEGATIVE_PROMPT for CFG (matching Python behavior)
            let (negEmbeddings, negMask) = encodePrompt(DEFAULT_NEGATIVE_PROMPT, encoder: textEncoder)
            MLX.eval(negEmbeddings, negMask)
            textEmbeddings = MLX.concatenated([negEmbeddings, promptEmbeddings], axis: 0)
            contextMask = MLX.concatenated([negMask, promptMask], axis: 0)
            MLX.eval(textEmbeddings, contextMask)
        } else {
            textEmbeddings = promptEmbeddings
            contextMask = promptMask
        }
        LTXDebug.log("Text encoding: \(String(format: "%.1f", Date().timeIntervalSince(textEncStart)))s")

        // 2. Seed
        if let seed = config.seed {
            MLXRandom.seed(seed)
            LTXDebug.log("Using seed: \(seed)")
        }

        // Apply cross-attention scaling if configured
        if config.crossAttentionScale != 1.0 {
            transformer.setCrossAttentionScale(config.crossAttentionScale)
        }

        // === STAGE 1: Denoise at half resolution ===
        LTXDebug.log("=== Stage 1: Half-resolution denoising (8 steps) ===")
        let stage1Start = Date()

        let stage1Shape = VideoLatentShape.fromPixelDimensions(
            batch: 1, channels: 128,
            frames: config.numFrames,
            height: halfHeight,
            width: halfWidth
        )
        LTXDebug.log("Stage 1 latent: \(stage1Shape.frames)x\(stage1Shape.height)x\(stage1Shape.width)")

        // Use distilled sigmas for stage 1
        let stage1Sigmas = DISTILLED_SIGMA_VALUES

        // Generate noise and scale by initial sigma
        var latent = generateNoise(shape: stage1Shape, seed: config.seed)
        latent = latent * stage1Sigmas[0]
        MLX.eval(latent)

        // Denoise
        latent = denoise(
            latent: latent,
            sigmas: stage1Sigmas,
            textEmbeddings: textEmbeddings,
            promptEmbeddings: promptEmbeddings,
            contextMask: contextMask,
            latentShape: stage1Shape,
            config: config,
            transformer: transformer,
            useCFG: useCFG,
            profile: profile
        )
        LTXDebug.log("Stage 1 complete: \(String(format: "%.1f", Date().timeIntervalSince(stage1Start)))s")
        LTXDebug.log("Stage 1 latent stats: mean=\(latent.mean().item(Float.self)), std=\(MLX.sqrt(MLX.variance(latent)).item(Float.self))")

        // Save stage 1 output for AdaIN reference (in normalized space)
        let stage1Output = latent

        // === UPSCALE 2x ===
        LTXDebug.log("=== Upscaling latent 2x ===")
        let upscaleStart = Date()

        // Load upscaler
        let upscaler = try loadSpatialUpscaler(from: upscalerWeightsPath)

        // Get VAE per-channel statistics for denormalization
        let latentMean = vaeDecoder.meanOfMeans
        let latentStd = vaeDecoder.stdOfMeans
        MLX.eval(latentMean, latentStd)
        LTXDebug.log("VAE stats: mean shape=\(latentMean.shape), std shape=\(latentStd.shape)")

        // Inline upsample with diagnostics
        let mean5d = latentMean.reshaped([1, -1, 1, 1, 1])
        let std5d = latentStd.reshaped([1, -1, 1, 1, 1])

        // Step 1: Denormalize
        let denormedLatent = latent * std5d + mean5d
        MLX.eval(denormedLatent)
        LTXDebug.log("After denormalize: mean=\(denormedLatent.mean().item(Float.self)), std=\(MLX.sqrt(MLX.variance(denormedLatent)).item(Float.self))")

        // Step 2: Upscale
        let upscaledLatent = upscaler(denormedLatent)
        MLX.eval(upscaledLatent)
        LTXDebug.log("After upscaler: shape=\(upscaledLatent.shape), mean=\(upscaledLatent.mean().item(Float.self)), std=\(MLX.sqrt(MLX.variance(upscaledLatent)).item(Float.self))")

        // Step 3: Renormalize
        latent = (upscaledLatent - mean5d) / std5d
        MLX.eval(latent)
        LTXDebug.log("After renormalize: mean=\(latent.mean().item(Float.self)), std=\(MLX.sqrt(MLX.variance(latent)).item(Float.self))")

        // Step 4: AdaIN filtering — match per-channel stats to stage 1 output
        // Prevents distribution shift from the upsampler (matches Lightricks pipeline)
        latent = adainFilterLatent(latent, reference: stage1Output)
        MLX.eval(latent)
        LTXDebug.log("After AdaIN: mean=\(latent.mean().item(Float.self)), std=\(MLX.sqrt(MLX.variance(latent)).item(Float.self))")

        LTXDebug.log("Upscaled latent shape: \(latent.shape)")
        LTXDebug.log("Upscale time: \(String(format: "%.1f", Date().timeIntervalSince(upscaleStart)))s")

        // === STAGE 2: Refine at full resolution ===
        LTXDebug.log("=== Stage 2: Full-resolution refinement (3 steps) ===")
        let stage2Start = Date()

        let stage2Shape = VideoLatentShape.fromPixelDimensions(
            batch: 1, channels: 128,
            frames: config.numFrames,
            height: config.height,
            width: config.width
        )
        LTXDebug.log("Stage 2 latent: \(stage2Shape.frames)x\(stage2Shape.height)x\(stage2Shape.width)")

        // Add noise for refinement: latent = noise * sigma + latent * (1 - sigma)
        let stage2Sigmas = STAGE_2_DISTILLED_SIGMA_VALUES
        let noiseScale = stage2Sigmas[0]  // 0.909375
        let noise = generateNoise(shape: stage2Shape)
        latent = MLXArray(noiseScale) * noise + MLXArray(1.0 - noiseScale) * latent
        MLX.eval(latent)
        LTXDebug.log("Added stage 2 noise (σ=\(noiseScale))")
        LTXDebug.log("Stage 2 input stats: mean=\(latent.mean().item(Float.self)), std=\(MLX.sqrt(MLX.variance(latent)).item(Float.self))")

        // Denoise (same transformer, no LoRA needed!)
        latent = denoise(
            latent: latent,
            sigmas: stage2Sigmas,
            textEmbeddings: textEmbeddings,
            promptEmbeddings: promptEmbeddings,
            contextMask: contextMask,
            latentShape: stage2Shape,
            config: config,
            transformer: transformer,
            useCFG: useCFG,
            profile: profile
        )
        LTXDebug.log("Stage 2 complete: \(String(format: "%.1f", Date().timeIntervalSince(stage2Start)))s")
        LTXDebug.log("Stage 2 output stats: mean=\(latent.mean().item(Float.self)), std=\(MLX.sqrt(MLX.variance(latent)).item(Float.self)), min=\(latent.min().item(Float.self)), max=\(latent.max().item(Float.self))")

        // === VAE Decode ===
        // Use timestep_conditioning from VAE config.json
        let vaeTimestep2: Float? = vaeDecoder.timestepConditioning ? 0.05 : nil
        LTXDebug.log("Decoding latents to video... (timestep=\(vaeTimestep2.map { String($0) } ?? "nil"))")
        let vaeStart = Date()
        let videoTensor = decodeVideo(latent: latent, decoder: vaeDecoder, timestep: vaeTimestep2)
        MLX.eval(videoTensor)
        LTXDebug.log("VAE decode: \(String(format: "%.1f", Date().timeIntervalSince(vaeStart)))s")

        // Trim to requested frame count
        let trimmedVideo: MLXArray
        if videoTensor.dim(0) > config.numFrames {
            trimmedVideo = videoTensor[0..<config.numFrames]
        } else {
            trimmedVideo = videoTensor
        }

        let generationTime = Date().timeIntervalSince(generationStart)
        LTXDebug.log("Total two-stage time: \(String(format: "%.1f", generationTime))s")

        return VideoGenerationResult(
            frames: trimmedVideo,
            seed: config.seed ?? 0,
            generationTime: generationTime,
            timings: nil
        )
    }

    // MARK: - LoRA Support

    /// Apply LoRA weights to the transformer
    ///
    /// - Parameters:
    ///   - loraPath: Path to LoRA .safetensors file
    ///   - scale: LoRA scale factor
    /// - Returns: Application result
    @discardableResult
    public func applyLoRA(
        from loraPath: String,
        scale: Float = 1.0
    ) throws -> LoRAApplicationResult {
        guard let transformer = transformer else {
            throw LTXError.modelNotLoaded("Transformer not loaded")
        }

        return try transformer.applyLoRA(from: loraPath, scale: scale)
    }

    // MARK: - Prompt Enhancement

    /// Official Lightricks T2V system prompt for Gemma-based prompt enhancement
    private static let promptEnhancementSystemPrompt = """
    You are a Creative Assistant. Given a user's raw input prompt describing a scene or concept, expand it into a detailed video generation prompt with specific visuals and integrated audio to guide a text-to-video model.

    #### Guidelines
    - Strictly follow all aspects of the user's raw input: include every element requested (style, visuals, motions, actions, camera movement, audio).
        - If the input is vague, invent concrete details: lighting, textures, materials, scene settings, etc.
            - For characters: describe gender, clothing, hair, expressions. DO NOT invent unrequested characters.
    - Use active language: present-progressive verbs ("is walking," "speaking"). If no action specified, describe natural movements.
    - Maintain chronological flow: use temporal connectors ("as," "then," "while").
    - Audio layer: Describe complete soundscape (background audio, ambient sounds, SFX, speech/music when requested). Integrate sounds chronologically alongside actions. Be specific (e.g., "soft footsteps on tile"), not vague (e.g., "ambient sound is present").
    - Speech (only when requested):
        - For ANY speech-related input (talking, conversation, singing, etc.), ALWAYS include exact words in quotes with voice characteristics (e.g., "The man says in an excited voice: 'You won't believe what I just saw!'").
        - Specify language if not English and accent if relevant.
    - Style: Include visual style at the beginning: "Style: <style>, <rest of prompt>." Default to cinematic-realistic if unspecified. Omit if unclear.
    - Visual and audio only: NO non-visual/auditory senses (smell, taste, touch).
    - Restrained language: Avoid dramatic/exaggerated terms. Use mild, natural phrasing.
        - Colors: Use plain terms ("red dress"), not intensified ("vibrant blue," "bright red").
        - Lighting: Use neutral descriptions ("soft overhead light"), not harsh ("blinding light").
        - Facial features: Use delicate modifiers for subtle features (i.e., "subtle freckles").

    #### Important notes:
    - Analyze the user's raw input carefully. In cases of FPV or POV, exclude the description of the subject whose POV is requested.
    - Camera motion: DO NOT invent camera motion unless requested by the user.
    - Speech: DO NOT modify user-provided character dialogue unless it's a typo.
    - No timestamps or cuts: DO NOT use timestamps or describe scene cuts unless explicitly requested.
    - Format: DO NOT use phrases like "The scene opens with...". Start directly with Style (optional) and chronological scene description.
    - Format: DO NOT start your response with special characters.
    - DO NOT invent dialogue unless the user mentions speech/talking/singing/conversation.
    - If the user's raw input prompt is highly detailed, chronological and in the requested format: DO NOT make major edits or introduce new elements. Add/enhance audio descriptions if missing.

    #### Output Format (Strict):
    - Single continuous paragraph in natural language (English).
    - NO titles, headings, prefaces, code fences, or Markdown.
    - If unsafe/invalid, return original user prompt. Never ask questions or clarifications.

    Your output quality is CRITICAL. Generate visually rich, dynamic prompts with integrated audio for high-quality video generation.

    #### Example
    Input: "A woman at a coffee shop talking on the phone"
    Output:
    Style: realistic with cinematic lighting. In a medium close-up, a woman in her early 30s with shoulder-length brown hair sits at a small wooden table by the window. She wears a cream-colored turtleneck sweater, holding a white ceramic coffee cup in one hand and a smartphone to her ear with the other. Ambient cafe sounds fill the space—espresso machine hiss, quiet conversations, gentle clinking of cups. The woman listens intently, nodding slightly, then takes a sip of her coffee and sets it down with a soft clink. Her face brightens into a warm smile as she speaks in a clear, friendly voice, 'That sounds perfect! I'd love to meet up this weekend. How about Saturday afternoon?' She laughs softly—a genuine chuckle—and shifts in her chair. Behind her, other patrons move subtly in and out of focus. 'Great, I'll see you then,' she concludes cheerfully, lowering the phone.
    """

    /// Enhance a short prompt into a detailed video description using Gemma generation.
    ///
    /// Uses Gemma in autoregressive mode (with KV cache for efficiency) to expand
    /// a brief prompt into a rich, detailed description better suited for video generation.
    /// Formats the input using Gemma 3's chat template for proper generation.
    ///
    /// - Parameters:
    ///   - prompt: Short text prompt to enhance
    ///   - maxTokens: Maximum tokens to generate (default: 512)
    ///   - temperature: Sampling temperature (default: 0.7)
    /// - Returns: Enhanced prompt string
    public func enhancePrompt(_ prompt: String, maxTokens: Int = 512, temperature: Float = 0.7) -> String {
        guard let gemma = gemmaModel, let tokenizer = tokenizer else {
            LTXDebug.log("Warning: Gemma/tokenizer not loaded for prompt enhancement, using original prompt")
            return prompt
        }

        LTXDebug.log("Enhancing prompt: \"\(prompt)\"")
        let startTime = Date()

        // Determine stop tokens: <eos> (1) and <end_of_turn> (236764 for Gemma 3)
        let eosId = Int32(tokenizer.eosTokenId ?? 1)
        let endOfTurnTokens = tokenizer.encode(text: "<end_of_turn>")
        var stopIds: Set<Int32> = [eosId]
        if let eotId = endOfTurnTokens.last {
            stopIds.insert(Int32(eotId))
            LTXDebug.log("Stop tokens: eos=\(eosId), end_of_turn=\(eotId)")
        }

        // Build chat messages matching Lightricks official format
        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": Self.promptEnhancementSystemPrompt],
            ["role": "user", "content": "user prompt: \(prompt)"],
        ]

        // Tokenize using Gemma 3 chat template
        let encoded: [Int]
        do {
            encoded = try tokenizer.applyChatTemplate(messages: messages)
            LTXDebug.log("Enhancement input (chat template): \(encoded.count) tokens")
        } catch {
            LTXDebug.log("Chat template failed (\(error)), falling back to raw tokenization")
            // Fallback: manual Gemma 3 chat format
            let rawPrompt = "<start_of_turn>user\n\(Self.promptEnhancementSystemPrompt)\n\nuser prompt: \(prompt)<end_of_turn>\n<start_of_turn>model\n"
            let fallbackEncoded = tokenizer.encode(text: rawPrompt)
            LTXDebug.log("Enhancement input (fallback): \(fallbackEncoded.count) tokens")
            let inputIds = MLXArray(fallbackEncoded.map { Int32($0) }).reshaped(1, fallbackEncoded.count)
            let generatedTokenIds = gemma.generateTokens(
                inputIds: inputIds,
                maxNewTokens: maxTokens,
                temperature: temperature,
                topP: 0.95,
                repetitionPenalty: 1.1,
                repetitionContextSize: 64,
                eosTokenIds: stopIds
            )
            let elapsed = Date().timeIntervalSince(startTime)
            LTXDebug.log("Generated \(generatedTokenIds.count) tokens in \(String(format: "%.1f", elapsed))s")
            let enhanced = tokenizer.decode(tokens: generatedTokenIds.map { Int($0) })
            let cleaned = cleanEnhancedPrompt(enhanced)
            if cleaned.isEmpty { return prompt }
            LTXDebug.log("Enhanced prompt: \"\(cleaned)\"")
            return cleaned
        }

        let inputIds = MLXArray(encoded.map { Int32($0) }).reshaped(1, encoded.count)

        // Generate with official parameters: temperature=0.7, top-p sampling, repetition penalty
        let generatedTokenIds = gemma.generateTokens(
            inputIds: inputIds,
            maxNewTokens: maxTokens,
            temperature: temperature,
            topP: 0.95,
            repetitionPenalty: 1.1,
            repetitionContextSize: 64,
            eosTokenIds: stopIds
        )

        let elapsed = Date().timeIntervalSince(startTime)
        LTXDebug.log("Generated \(generatedTokenIds.count) tokens in \(String(format: "%.1f", elapsed))s (\(String(format: "%.1f", Double(generatedTokenIds.count) / elapsed)) tok/s)")

        // Decode tokens to text
        let enhanced = tokenizer.decode(tokens: generatedTokenIds.map { Int($0) })
        let cleaned = cleanEnhancedPrompt(enhanced)

        if cleaned.isEmpty {
            LTXDebug.log("Enhancement produced empty result, using original prompt")
            return prompt
        }

        LTXDebug.log("Enhanced prompt: \"\(cleaned)\"")
        return cleaned
    }

    /// Clean up a Gemma-enhanced prompt: strip control tokens and trailing noise
    private func cleanEnhancedPrompt(_ raw: String) -> String {
        var text = raw
        // Remove <end_of_turn>, <start_of_turn>, and any role markers
        text = text.replacingOccurrences(of: "<end_of_turn>", with: "")
        text = text.replacingOccurrences(of: "<start_of_turn>", with: "")
        text = text.replacingOccurrences(of: "<eos>", with: "")
        text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return text
    }

    // MARK: - Standalone Text Encoding

    /// Encode text prompt result
    public struct TextEncodingResult {
        /// The final prompt that was encoded (enhanced if requested)
        public let prompt: String
        /// Encoded embeddings [1, 1024, 3840]
        public let embeddings: MLXArray
        /// Attention mask [1, 1024]
        public let mask: MLXArray
        /// Encoding statistics
        public let mean: Float
        public let std: Float
    }

    /// Encode a text prompt without generating video
    ///
    /// Runs the full text encoding pipeline: tokenize → Gemma → feature extractor → connector.
    /// Optionally enhances the prompt first using Gemma generation.
    ///
    /// - Parameters:
    ///   - prompt: Text prompt to encode
    ///   - enhance: Whether to enhance the prompt first (default: false)
    /// - Returns: TextEncodingResult with embeddings, mask, and statistics
    public func encodeText(
        _ prompt: String,
        enhance: Bool = false
    ) throws -> TextEncodingResult {
        guard let textEncoder = textEncoder else {
            throw LTXError.modelNotLoaded("Text encoder not loaded. Call loadModels() first.")
        }
        guard isGemmaLoaded else {
            throw LTXError.modelNotLoaded("Gemma model not loaded. Call loadModels() first.")
        }

        // Optionally enhance
        let effectivePrompt: String
        if enhance {
            effectivePrompt = enhancePrompt(prompt)
        } else {
            effectivePrompt = prompt
        }

        // Encode
        let (embeddings, mask) = encodePrompt(effectivePrompt, encoder: textEncoder)
        MLX.eval(embeddings, mask)

        // Stats
        let mean = embeddings.mean().item(Float.self)
        let std = MLX.sqrt(MLX.variance(embeddings)).item(Float.self)

        return TextEncodingResult(
            prompt: effectivePrompt,
            embeddings: embeddings,
            mask: mask,
            mean: mean,
            std: std
        )
    }

    // MARK: - Download Helpers

    /// Download spatial upscaler weights (if not already cached)
    /// - Returns: Path to the upscaler safetensors file
    public func downloadUpscalerWeights() async throws -> String {
        let url = try await downloader.downloadUpscalerWeights()
        return url.path
    }

    /// Download distilled LoRA weights (if not already cached)
    /// - Returns: Path to the distilled LoRA safetensors file
    public func downloadDistilledLoRA() async throws -> String {
        let url = try await downloader.downloadDistilledLoRA()
        return url.path
    }

    /// Fuse LoRA weights into the transformer (permanent merge)
    ///
    /// Uses batched processing per transformer block to minimize peak memory.
    /// LoRA tensors are freed after fusion via scope exit + cache clearing.
    ///
    /// - Parameters:
    ///   - loraPath: Path to LoRA .safetensors file
    ///   - scale: LoRA scale factor
    /// - Returns: Number of layers modified
    @discardableResult
    public func fuseLoRA(
        from loraPath: String,
        scale: Float = 1.0
    ) throws -> Int {
        guard let transformer = transformer else {
            throw LTXError.modelNotLoaded("Transformer not loaded")
        }
        let (originals, _) = try transformer.fuseLoRA(from: loraPath, scale: scale)
        // LoRA weights (LoRAWeights struct) go out of scope here — freed by ARC.
        // GPU cache cleared inside fuseWeights() after all batches.
        // Final cleanup to release any remaining intermediate tensors.
        Memory.clearCache()
        return originals.count
    }

    // MARK: - Memory Management

    /// Clear all loaded models
    public func clearAll() {
        gemmaModel = nil
        tokenizer = nil
        textEncoder = nil
        transformer = nil
        vaeDecoder = nil
        cachedNullEmbeddings = nil
        LTXDebug.log("All models cleared")
    }

    /// Clear only Gemma model (to save memory after encoding)
    public func clearGemma() {
        gemmaModel = nil
        LTXDebug.log("Gemma model cleared")
    }

    /// Get estimated memory usage for a generation config
    public func estimateMemory(for config: LTXVideoGenerationConfig) -> Int64 {
        let shape = VideoLatentShape.fromPixelDimensions(
            frames: config.numFrames,
            height: config.height,
            width: config.width
        )

        return estimateMemoryUsage(
            shape: shape,
            numSteps: config.numSteps,
            cfg: config.cfgScale > 1.0
        )
    }

    // MARK: - Private Helpers

    /// Encode text prompt to embeddings using Gemma + text encoder pipeline
    ///
    /// Pipeline:
    /// 1. Tokenize prompt with left-padding
    /// 2. Run through Gemma3 model to get all 49 hidden states
    /// 3. Pass hidden states through feature extractor + connector
    /// Text encoding max sequence length (must match Python mlx-video default)
    private let textMaxLength = 1024

    /// 4. Return video encoding [1, textMaxLength, 3840] and attention mask [1, textMaxLength]
    private func encodePrompt(_ prompt: String, encoder: VideoGemmaTextEncoderModel) -> (encoding: MLXArray, mask: MLXArray) {
        guard let gemma = gemmaModel else {
            LTXDebug.log("Warning: Gemma model not loaded, using placeholder embeddings")
            let placeholder = createPlaceholderEmbeddings(prompt: prompt)
            let mask = MLXArray.ones([1, textMaxLength]).asType(.int32)
            return (placeholder, mask)
        }

        // Step 1: Tokenize with left-padding
        let (inputIds, attentionMask) = tokenizePrompt(prompt, maxLength: textMaxLength)
        let activeTokens = Int(attentionMask.sum().item(Int32.self))
        LTXDebug.log("Tokenized: \(inputIds.shape), padding=\(textMaxLength - activeTokens), active=\(activeTokens)")
        // Debug: show first and last tokens for comparison with Python
        MLX.eval(inputIds)
        let idsFlat = inputIds.reshaped([-1])
        var firstTokens: [Int32] = []
        var lastTokens: [Int32] = []
        for i in 0..<min(5, textMaxLength) { firstTokens.append(idsFlat[i].item(Int32.self)) }
        for i in max(0, textMaxLength-10)..<textMaxLength { lastTokens.append(idsFlat[i].item(Int32.self)) }
        LTXDebug.log("  First 5 tokens: \(firstTokens)")
        LTXDebug.log("  Last 10 tokens: \(lastTokens)")

        // Step 2: Run Gemma forward pass to extract all 49 hidden states
        LTXDebug.log("Running Gemma forward pass...")
        let (_, allHiddenStates) = gemma(inputIds, attentionMask: attentionMask, outputHiddenStates: true)

        guard let states = allHiddenStates, states.count == gemma.config.hiddenLayers + 1 else {
            LTXDebug.log("Warning: Expected \(gemma.config.hiddenLayers + 1) hidden states, using placeholder")
            let placeholder = createPlaceholderEmbeddings(prompt: prompt)
            let mask = MLXArray.ones([1, textMaxLength]).asType(.int32)
            return (placeholder, mask)
        }
        LTXDebug.log("Got \(states.count) hidden states from Gemma")

        // Debug: dump hidden state stats for comparison with Python (disabled for production)
        // (HS comparison confirmed identical active tokens across all 49 layers)

        // Step 3: Pass through text encoder (feature extractor + connector)
        let encoderOutput = encoder.encodeFromHiddenStates(
            hiddenStates: states,
            attentionMask: attentionMask,
            paddingSide: "left"
        )

        MLX.eval(encoderOutput.videoEncoding, encoderOutput.attentionMask)
        let maskSum = encoderOutput.attentionMask.sum().item(Int32.self)
        LTXDebug.log("Text encoding: \(encoderOutput.videoEncoding.shape), mean=\(encoderOutput.videoEncoding.mean().item(Float.self))")
        LTXDebug.log("Text mask: \(encoderOutput.attentionMask.shape), active=\(maskSum)/\(encoderOutput.attentionMask.dim(-1))")

        return (encoderOutput.videoEncoding, encoderOutput.attentionMask)
    }

    /// Tokenize prompt with left-padding (matching Python mlx-video max_length=1024)
    private func tokenizePrompt(_ prompt: String, maxLength: Int = 1024) -> (inputIds: MLXArray, attentionMask: MLXArray) {
        guard let tokenizer = tokenizer else {
            fatalError("Tokenizer not loaded. Call loadModels() first.")
        }

        // Tokenize (Gemma tokenizer adds BOS=2 automatically)
        let encoded = tokenizer.encode(text: prompt)
        var tokens = Array(encoded.suffix(maxLength)).map { Int32($0) }

        // Left-pad with pad_token_id=0 (matching Python tokenizer)
        let paddingNeeded = maxLength - tokens.count
        let padTokenId: Int32 = 0  // Gemma pad_token_id=0 (NOT eos=1)
        if paddingNeeded > 0 {
            tokens = [Int32](repeating: padTokenId, count: paddingNeeded) + tokens
        }

        // Attention mask: 0 for padding, 1 for real tokens
        let mask = [Float](repeating: 0, count: paddingNeeded)
            + [Float](repeating: 1, count: maxLength - paddingNeeded)

        let inputIds = MLXArray(tokens).reshaped([1, maxLength])
        let attentionMask = MLXArray(mask).reshaped([1, maxLength])

        return (inputIds, attentionMask)
    }

    /// Create placeholder embeddings when Gemma is not available
    private func createPlaceholderEmbeddings(prompt: String) -> MLXArray {
        let hiddenDim = 3840
        return MLXArray.zeros([1, textMaxLength, hiddenDim]).asType(.float32)
    }

    /// Get null embeddings for unconditional generation (cached encoding of empty string "")
    private func getNullEmbeddings() -> (encoding: MLXArray, mask: MLXArray) {
        if let cached = cachedNullEmbeddings { return cached }
        guard let encoder = textEncoder else {
            LTXDebug.log("Warning: textEncoder not loaded for null embeddings, using zeros")
            let zeros = MLXArray.zeros([1, textMaxLength, 3840]).asType(.float32)
            let mask = MLXArray.ones([1, textMaxLength]).asType(.int32)
            return (zeros, mask)
        }
        LTXDebug.log("Encoding empty string for null embeddings...")
        let result = encodePrompt("", encoder: encoder)
        MLX.eval(result.encoding, result.mask)
        cachedNullEmbeddings = result
        LTXDebug.log("Null embeddings cached: \(result.encoding.shape)")
        return result
    }

    /// Create position indices for RoPE
    private func createPositionIndices(shape: VideoLatentShape) -> MLXArray {
        // Create 3D position grid (time, height, width)
        var indices: [MLXArray] = []

        for t in 0..<shape.frames {
            for h in 0..<shape.height {
                for w in 0..<shape.width {
                    indices.append(MLXArray([Int32(t), Int32(h), Int32(w)]))
                }
            }
        }

        return MLX.stacked(indices, axis: 0).reshaped([1, shape.tokenCount, 3])
    }

    // Preview frame generation removed — calling VAE at each step wastes time
    // and the raw decoder output (B,C,F,H,W) needs transposition for tensorToImages.
}

// MARK: - VideoLatentShape Extension

extension VideoLatentShape {
    /// Create doubled shape for CFG
    func doubled() -> VideoLatentShape {
        VideoLatentShape(
            batch: batch * 2,
            channels: channels,
            frames: frames,
            height: height,
            width: width
        )
    }
}

// MARK: - Convenience Functions

/// Create and configure an LTX pipeline
public func createPipeline(
    model: LTXModel = .distilled,
    hfToken: String? = nil
) -> LTXPipeline {
    return LTXPipeline(model: model, hfToken: hfToken)
}
