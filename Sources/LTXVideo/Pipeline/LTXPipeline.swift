// LTXPipeline.swift - Main Video Generation Pipeline for LTX-2
// Copyright 2025

import CoreGraphics
import Foundation
@preconcurrency import MLX
import MLXRandom
import MLXNN
import Tokenizers
import Hub

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

    /// Whether models are loaded
    public var isLoaded: Bool {
        gemmaModel != nil && textEncoder != nil && transformer != nil && vaeDecoder != nil
    }

    // MARK: - Initialization

    public init(
        model: LTXModel = .distilled,
        quantization: LTXQuantizationConfig = .default,
        hfToken: String? = nil
    ) {
        self.model = model
        self.quantization = quantization
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
        ltxWeightsPath: String? = nil
    ) async throws {
        LTXDebug.log("Loading models for \(model.displayName)...")
        var stepStart = Date()

        // Step 1: Load Gemma model and tokenizer
        progressCallback?(DownloadProgress(progress: 0.1, message: "Loading Gemma model..."))

        let gemmaURL: URL
        if let gemmaPath = gemmaModelPath {
            gemmaURL = URL(fileURLWithPath: gemmaPath)
        } else {
            LTXDebug.log("Downloading Gemma3 12B 4-bit (if needed)...")
            gemmaURL = try await downloader.downloadGemma { progress in
                progressCallback?(progress)
            }
        }
        LTXDebug.log("[TIME] Gemma download check: \(String(format: "%.1f", Date().timeIntervalSince(stepStart)))s")

        stepStart = Date()
        LTXDebug.log("Loading Gemma3 model from \(gemmaURL.path)...")
        gemmaModel = try Gemma3WeightLoader.loadModel(from: gemmaURL)
        LTXDebug.log("[TIME] Gemma load: \(String(format: "%.1f", Date().timeIntervalSince(stepStart)))s — \(gemmaModel!.config.hiddenLayers) layers")

        stepStart = Date()
        progressCallback?(DownloadProgress(progress: 0.2, message: "Loading tokenizer..."))
        tokenizer = try await AutoTokenizer.from(modelFolder: gemmaURL)
        LTXDebug.log("[TIME] Tokenizer load: \(String(format: "%.1f", Date().timeIntervalSince(stepStart)))s")

        // Step 2: Download/load unified LTX weights
        progressCallback?(DownloadProgress(progress: 0.3, message: "Loading LTX-2 weights..."))

        let weightsURL: URL
        if let path = ltxWeightsPath {
            weightsURL = URL(fileURLWithPath: path)
        } else {
            stepStart = Date()
            LTXDebug.log("Downloading LTX-2 unified weights (if needed)...")
            weightsURL = try await downloader.downloadLTXWeights(model: model) { progress in
                progressCallback?(progress)
            }
            LTXDebug.log("[TIME] LTX weights download check: \(String(format: "%.1f", Date().timeIntervalSince(stepStart)))s")
        }

        stepStart = Date()
        LTXDebug.log("Loading unified weights from \(weightsURL.path)...")
        let components = try LTXWeightLoader.loadUnifiedWeights(
            from: weightsURL.path,
            isFP8: model.isFP8
        )
        LTXDebug.log("[TIME] Load + split weights: \(String(format: "%.1f", Date().timeIntervalSince(stepStart)))s")

        // Step 3: Create and load transformer
        progressCallback?(DownloadProgress(progress: 0.5, message: "Loading transformer..."))

        let transformerConfig = model.transformerConfig
        transformer = LTXTransformer(config: transformerConfig)

        stepStart = Date()
        LTXDebug.log("Applying \(components.transformer.count) transformer weights...")
        try LTXWeightLoader.applyTransformerWeights(components.transformer, to: transformer!)
        LTXDebug.log("[TIME] Apply transformer weights: \(String(format: "%.1f", Date().timeIntervalSince(stepStart)))s")

        // Evaluate transformer weights to ensure they're fully materialized
        // FP8 dequantization creates computation graphs that must be evaluated before inference
        stepStart = Date()
        for (i, block) in transformer!.transformerBlocks.enumerated() {
            eval(block.parameters())
            if i == 0 {
                LTXDebug.log("  Block 0 weights evaluated")
            }
        }
        // Also eval non-block parameters (patchify, adaln, caption, proj_out, etc.)
        eval(transformer!.patchifyProj.parameters())
        eval(transformer!.adalnSingle.parameters())
        eval(transformer!.captionProjection.parameters())
        eval(transformer!.projOut.parameters())
        eval(transformer!.normOut.parameters())
        eval(transformer!.scaleShiftTable)
        LTXDebug.log("[TIME] Eval transformer weights: \(String(format: "%.1f", Date().timeIntervalSince(stepStart)))s")

        // Step 4: Create and load VAE decoder
        progressCallback?(DownloadProgress(progress: 0.7, message: "Loading VAE decoder..."))

        vaeDecoder = VideoDecoder()

        stepStart = Date()
        LTXDebug.log("Applying \(components.vaeDecoder.count) VAE weights...")
        try LTXWeightLoader.applyVAEWeights(components.vaeDecoder, to: vaeDecoder!)
        LTXDebug.log("[TIME] VAE load: \(String(format: "%.1f", Date().timeIntervalSince(stepStart)))s")

        // Step 5: Create and load text encoder (feature extractor + connector)
        progressCallback?(DownloadProgress(progress: 0.9, message: "Loading text encoder..."))

        textEncoder = VideoGemmaTextEncoderModel()

        stepStart = Date()
        LTXDebug.log("Applying \(components.textEncoder.count) text encoder weights...")
        try LTXWeightLoader.applyTextEncoderWeights(components.textEncoder, to: textEncoder!)
        LTXDebug.log("[TIME] TextEncoder load: \(String(format: "%.1f", Date().timeIntervalSince(stepStart)))s")

        progressCallback?(DownloadProgress(progress: 1.0, message: "Models loaded successfully"))
        LTXDebug.log("All models loaded successfully")
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
    public func generateVideo(
        prompt: String,
        negativePrompt: String? = nil,
        config: LTXVideoGenerationConfig,
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

        // 1. Encode text prompt
        let useCFG = config.cfgScale > 1.0
        LTXDebug.log("Encoding text prompt... (CFG=\(config.cfgScale), enabled=\(useCFG))")
        let textEncStart = Date()
        let (promptEmbeddings, promptMask) = encodePrompt(prompt, encoder: textEncoder)
        MLX.eval(promptEmbeddings, promptMask)
        LTXDebug.log("promptEmbeddings shape: \(promptEmbeddings.shape)")

        let textEmbeddings: MLXArray
        let contextMask: MLXArray
        if useCFG {
            // CFG mode: stack [negative, positive] embeddings for doubled batch
            let (negEmbeddings, negMask) =
                negativePrompt.map { encodePrompt($0, encoder: textEncoder) }
                ?? (createNullEmbeddings(like: promptEmbeddings), MLXArray.ones(like: promptMask))
            MLX.eval(negEmbeddings, negMask)
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

        timings.textEncoding = Date().timeIntervalSince(textEncStart)

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

        // 4. Get sigma schedule
        let sigmas = scheduler.getSigmas(numSteps: config.numSteps)
        LTXDebug.log("Sigma schedule: \(sigmas.map { String(format: "%.4f", $0) })")

        // 5. Scale initial noise by first sigma
        latent = latent * sigmas[0]

        // 6. Denoising loop
        LTXDebug.log("Starting denoising loop (\(config.numSteps) steps)...")

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

            let latentInput: MLXArray
            let timestep: MLXArray

            if useCFG {
                // CFG: double batch [uncond, cond]
                latentInput = prepareForCFG(latent)
                timestep = MLXArray([sigma, sigma])
            } else {
                // No CFG: single batch
                latentInput = latent
                timestep = MLXArray([sigma])
            }

            // Patchify for transformer
            let patchified = patchify(latentInput)
            LTXDebug.log("Step \(step): patchified \(patchified.shape), σ=\(String(format: "%.4f", sigma))")

            // Run transformer — predicts velocity
            // NOTE: contextMask=nil matches PyTorch behavior — they don't use text masks
            // during denoising. The model was trained this way. Using a mask would change
            // softmax normalization in cross-attention and degrade quality progressively.
            let velocityPred = transformer(
                latent: patchified,
                context: textEmbeddings,
                timesteps: timestep,
                contextMask: nil,
                latentShape: (frames: latentShape.frames, height: latentShape.height, width: latentShape.width)
            )

            // Unpatchify velocity back to (B, C, F, H, W)
            let velocity: MLXArray
            if useCFG {
                let fullVelocity = unpatchify(velocityPred, shape: latentShape.doubled())
                velocity = applyCFG(output: fullVelocity, guidanceScale: config.cfgScale)
            } else {
                velocity = unpatchify(velocityPred, shape: latentShape)
            }

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
                let vMean = velocityPred.mean().item(Float.self)
                let vStd = MLX.sqrt(MLX.variance(velocityPred)).item(Float.self)
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
        LTXDebug.log("Decoding latents to video...")
        let vaeStart = Date()
        let videoTensor = decodeVideo(latent: latent, decoder: vaeDecoder)
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

    // MARK: - Memory Management

    /// Clear all loaded models
    public func clearAll() {
        gemmaModel = nil
        tokenizer = nil
        textEncoder = nil
        transformer = nil
        vaeDecoder = nil
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
    /// 4. Return video encoding [1, 256, 3840] and attention mask [1, 256]
    private func encodePrompt(_ prompt: String, encoder: VideoGemmaTextEncoderModel) -> (encoding: MLXArray, mask: MLXArray) {
        guard let gemma = gemmaModel else {
            LTXDebug.log("Warning: Gemma model not loaded, using placeholder embeddings")
            let placeholder = createPlaceholderEmbeddings(prompt: prompt)
            let mask = MLXArray.ones([1, 256]).asType(.int32)
            return (placeholder, mask)
        }

        // Step 1: Tokenize with left-padding
        let (inputIds, attentionMask) = tokenizePrompt(prompt)
        LTXDebug.log("Tokenized: \(inputIds.shape), padding=\(256 - Int(attentionMask.sum().item(Int32.self)))")

        // Step 2: Run Gemma forward pass to extract all 49 hidden states
        LTXDebug.log("Running Gemma forward pass...")
        let (_, allHiddenStates) = gemma(inputIds, attentionMask: attentionMask, outputHiddenStates: true)

        guard let states = allHiddenStates, states.count == gemma.config.hiddenLayers + 1 else {
            LTXDebug.log("Warning: Expected \(gemma.config.hiddenLayers + 1) hidden states, using placeholder")
            let placeholder = createPlaceholderEmbeddings(prompt: prompt)
            let mask = MLXArray.ones([1, 256]).asType(.int32)
            return (placeholder, mask)
        }
        LTXDebug.log("Got \(states.count) hidden states from Gemma")

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

    /// Tokenize prompt with left-padding (matching validated Encode command)
    private func tokenizePrompt(_ prompt: String, maxLength: Int = 256) -> (inputIds: MLXArray, attentionMask: MLXArray) {
        guard let tokenizer = tokenizer else {
            fatalError("Tokenizer not loaded. Call loadModels() first.")
        }

        // Tokenize (Gemma tokenizer adds BOS=2 automatically)
        let encoded = tokenizer.encode(text: prompt)
        var tokens = Array(encoded.suffix(maxLength)).map { Int32($0) }

        // Left-pad with eos_token_id (106 for Gemma 3)
        let paddingNeeded = maxLength - tokens.count
        let padTokenId = Int32(tokenizer.eosTokenId ?? 106)
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
        let seqLen = 256
        let hiddenDim = 3840
        return MLXArray.zeros([1, seqLen, hiddenDim]).asType(.float32)
    }

    /// Create null embeddings for unconditional generation
    private func createNullEmbeddings(like reference: MLXArray) -> MLXArray {
        return MLXArray.zeros(like: reference)
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
