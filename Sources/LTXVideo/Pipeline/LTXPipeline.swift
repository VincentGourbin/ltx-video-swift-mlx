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

/// Progress information emitted during the denoising phase of generation.
///
/// Passed to the `onProgress` callback of ``LTXPipeline/generateVideo(prompt:negativePrompt:config:onProgress:profile:)``
/// and ``LTXPipeline/generateVideoTwoStage(prompt:config:upscalerWeightsPath:onProgress:profile:)``.
///
/// ## Example
/// ```swift
/// let result = try await pipeline.generateVideo(
///     prompt: "A sunset",
///     config: config,
///     onProgress: { progress in
///         print("[\(Int(progress.progress * 100))%] \(progress.status)")
///     }
/// )
/// ```
public struct GenerationProgress: Sendable {
    /// Current denoising step (0-indexed)
    public let currentStep: Int

    /// Total number of denoising steps
    public let totalSteps: Int

    /// Current noise sigma value (decreases from 1.0 toward 0.0)
    public let sigma: Float

    /// Progress fraction from 0.0 (start) to 1.0 (complete)
    public var progress: Double {
        Double(currentStep + 1) / Double(totalSteps)
    }

    /// Human-readable status string, e.g. `"Step 3/8 (σ=0.7250)"`
    public var status: String {
        "Step \(currentStep + 1)/\(totalSteps) (σ=\(String(format: "%.4f", sigma)))"
    }
}

/// Callback invoked at each denoising step with progress information.
public typealias GenerationProgressCallback = @Sendable (GenerationProgress) -> Void

/// Callback invoked with intermediate frame previews during generation.
/// Parameters: frame index and the rendered CGImage.
public typealias FramePreviewCallback = @Sendable (Int, CGImage) -> Void

// MARK: - LTX Pipeline

/// The main orchestrator for LTX-2 text-to-video generation.
///
/// `LTXPipeline` manages the full generation lifecycle: model loading,
/// text encoding (Gemma 3), iterative denoising (48-block DiT transformer),
/// and VAE decoding to produce video frames.
///
/// ## Typical Usage
/// ```swift
/// let pipeline = LTXPipeline(model: .distilled)
/// try await pipeline.loadModels()
/// let result = try await pipeline.generateVideo(
///     prompt: "A cat walking in a garden",
///     config: LTXVideoGenerationConfig(width: 512, height: 512, numFrames: 25)
/// )
/// ```
///
/// ## Two-Stage Pipeline
/// For higher quality, use the dev model with distilled LoRA and 2x upscaling:
/// ```swift
/// let pipeline = LTXPipeline(model: .dev)
/// try await pipeline.loadModels()
/// let loraPath = try await pipeline.downloadDistilledLoRA()
/// try await pipeline.fuseLoRA(from: loraPath)
/// let upscalerPath = try await pipeline.downloadUpscalerWeights()
/// let result = try await pipeline.generateVideoTwoStage(
///     prompt: "Ocean sunset",
///     config: config,
///     upscalerWeightsPath: upscalerPath
/// )
/// ```
///
/// ## Memory Management
/// The pipeline automatically manages GPU memory between phases. Configure
/// the ``MemoryOptimizationConfig`` preset to control the tradeoff between
/// speed and memory usage.
///
/// - Note: This is an `actor` to ensure thread-safe access to model state.
public actor LTXPipeline {
    // MARK: - Properties

    /// The model variant (``LTXModel/distilled`` or ``LTXModel/dev``)
    public let model: LTXModel

    /// Quantization settings for transformer and text encoder
    public let quantization: LTXQuantizationConfig

    /// Memory optimization settings (eval frequency, cache clearing, component unloading)
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

    /// VAE encoder (loaded only for image-to-video)
    private var vaeEncoder: VideoEncoder?

    /// Audio: dual video/audio transformer (alternative to video-only transformer)
    private var ltx2Transformer: LTX2Transformer?

    /// Audio VAE decoder
    private var audioVAE: AudioVAE?

    /// Audio vocoder (mel → waveform)
    private var vocoder: LTX2Vocoder?

    /// Whether audio models are loaded
    public var isAudioLoaded: Bool {
        ltx2Transformer != nil && audioVAE != nil && vocoder != nil
    }

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

    /// Create a new LTX-2 generation pipeline.
    ///
    /// - Parameters:
    ///   - model: Model variant to use. Defaults to ``LTXModel/distilled``.
    ///   - quantization: Quantization settings. Defaults to ``LTXQuantizationConfig/default``.
    ///   - memoryOptimization: Memory optimization preset. Defaults to ``MemoryOptimizationConfig/default`` (light).
    ///   - hfToken: Optional HuggingFace API token for downloading gated models.
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

        // Step 3b: Apply on-the-fly quantization if configured
        if quantization.transformer.needsQuantization {
            stepStart = Date()
            let bits = quantization.transformer.bits
            let groupSize = quantization.transformer.groupSize
            LTXDebug.log("Quantizing transformer to \(bits)-bit (groupSize=\(groupSize))...")
            progressCallback?(DownloadProgress(progress: 0.6, message: "Quantizing transformer to \(bits)-bit..."))
            quantize(model: transformer!, groupSize: groupSize, bits: bits)
            eval(transformer!.parameters())
            Memory.clearCache()
            LTXDebug.log("[TIME] Transformer quantization: \(String(format: "%.1f", Date().timeIntervalSince(stepStart)))s")
        }

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

    // MARK: - Audio Model Loading

    /// Audio VAE constants
    private static let audioSampleRate: Int = 16000
    private static let audioHopLength: Int = 160
    private static let audioMelBins: Int = 64
    private static let audioLatentChannels: Int = 8
    private static let audioTemporalCompression: Int = 4
    private static let audioMelCompression: Int = 4
    private static let audioLatentMelBins: Int = audioMelBins / audioMelCompression  // 16
    private static let audioPackedChannels: Int = audioLatentChannels * audioLatentMelBins  // 128

    /// Load audio models (Audio VAE, Vocoder, and LTX2 dual transformer)
    ///
    /// This replaces the video-only transformer with the dual video/audio transformer,
    /// and loads the audio VAE decoder and vocoder for waveform synthesis.
    ///
    /// - Important: Call `loadModels()` first, then `loadAudioModels()`. The audio
    ///   transformer weights are in the same unified file and share video weights.
    public func loadAudioModels(
        progressCallback: DownloadProgressCallback? = nil
    ) async throws {
        LTXDebug.log("Loading audio models...")

        // Step 1: Download and load Audio VAE
        progressCallback?(DownloadProgress(progress: 0.1, message: "Downloading audio VAE..."))
        let audioVAEPath = try await downloader.downloadAudioVAE { progress in
            progressCallback?(progress)
        }
        let audioVAEWeights = try LTXWeightLoader.loadAudioVAEWeights(from: audioVAEPath.path)

        audioVAE = AudioVAE()
        try LTXWeightLoader.applyAudioVAEWeights(audioVAEWeights, to: audioVAE!)
        LTXDebug.log("Audio VAE loaded")

        // Step 2: Download and load Vocoder
        progressCallback?(DownloadProgress(progress: 0.4, message: "Downloading vocoder..."))
        let vocoderPath = try await downloader.downloadVocoder { progress in
            progressCallback?(progress)
        }
        let vocoderWeights = try LTXWeightLoader.loadVocoderWeights(from: vocoderPath.path)

        vocoder = LTX2Vocoder()
        try LTXWeightLoader.applyVocoderWeights(vocoderWeights, to: vocoder!)
        LTXDebug.log("Vocoder loaded")

        // Step 3: Create LTX2 dual transformer and load unified weights
        // The LTX2 transformer uses the same weight keys as the video-only transformer
        // plus additional audio-specific keys. We reload from the unified file.
        progressCallback?(DownloadProgress(progress: 0.6, message: "Loading dual audio/video transformer..."))

        let unifiedPath = try await downloader.downloadUnifiedWeights(model: model) { progress in
            progressCallback?(progress)
        }

        // Load and split unified weights
        let (transformerWeights, _, _) = try LTXWeightLoader.splitUnifiedWeightsFile(path: unifiedPath.path)

        // Create LTX2 dual transformer
        let ltx2 = LTX2Transformer(
            config: model.transformerConfig,
            ropeType: .split,
            memoryOptimization: memoryOptimization
        )

        // Apply weights (the key mapping should handle both video and audio keys)
        try LTXWeightLoader.applyTransformerWeights(transformerWeights, to: ltx2)

        // Apply quantization if configured
        if quantization.transformer == .qint8 || quantization.transformer == .int4 {
            let bits = quantization.transformer == .qint8 ? 8 : 4
            LTXDebug.log("Quantizing LTX2 transformer to \(quantization.transformer)...")
            quantize(model: ltx2, groupSize: 64, bits: bits)
            eval(ltx2.parameters())
            Memory.clearCache()
        }

        ltx2Transformer = ltx2

        // Unload the video-only transformer (replaced by LTX2)
        transformer = nil
        Memory.clearCache()

        // Step 4: Update text encoder to include audio connector
        progressCallback?(DownloadProgress(progress: 0.9, message: "Loading audio text connector..."))
        let connectorPath = try await downloader.downloadConnector(model: model)
        let connectorWeights = try LTXWeightLoader.loadConnectorWeights(from: connectorPath.path)

        // Check if audio connector keys exist in the weights
        let hasAudioKeys = connectorWeights.keys.contains { $0.contains("audio") }
        if hasAudioKeys {
            // Recreate text encoder with audio connector
            let newTextEncoder = VideoGemmaTextEncoderModel(
                audioEmbeddingsConnector: Embeddings1DConnector()
            )
            let mapped = LTXWeightLoader.mapTextEncoderWeights(connectorWeights)
            try LTXWeightLoader.applyTextEncoderWeights(mapped, to: newTextEncoder)
            textEncoder = newTextEncoder
            LTXDebug.log("Text encoder updated with audio connector")
        } else {
            LTXDebug.log("No audio connector keys found - audio text encoding will share video embeddings")
        }

        progressCallback?(DownloadProgress(progress: 1.0, message: "Audio models loaded successfully"))
        LTXDebug.log("All audio models loaded successfully")
    }

    /// Compute audio latent frame count from video parameters
    private func computeAudioLatentFrames(videoFrames: Int, fps: Float = 24.0) -> Int {
        let durationS = Float(videoFrames) / fps
        let audioLatentsPerSecond = Float(Self.audioSampleRate) / Float(Self.audioHopLength) / Float(Self.audioTemporalCompression)
        return Int(round(Double(durationS * audioLatentsPerSecond)))
    }

    /// Pack audio latents for transformer input
    ///
    /// - Parameter latents: (B, 8, T, 16) audio latent tensor
    /// - Returns: (B, T, 128) packed audio latents
    private func packAudioLatents(_ latents: MLXArray) -> MLXArray {
        // (B, C, T, M) -> (B, T, C, M) -> (B, T, C*M)
        let transposed = latents.transposed(0, 2, 1, 3)
        return transposed.reshaped([transposed.dim(0), transposed.dim(1), -1])
    }

    /// Unpack audio latents from transformer output
    ///
    /// - Parameters:
    ///   - latents: (B, T, 128) packed audio latents
    ///   - numFrames: Number of audio latent frames
    /// - Returns: (B, 8, T, 16) unpacked audio latents
    private func unpackAudioLatents(_ latents: MLXArray, numFrames: Int) -> MLXArray {
        let b = latents.dim(0)
        // (B, T, C*M) -> (B, T, C, M) -> (B, C, T, M)
        let unflattened = latents.reshaped([b, numFrames, Self.audioLatentChannels, Self.audioLatentMelBins])
        return unflattened.transposed(0, 2, 1, 3)
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
        LTXMemoryManager.logMemoryState("generation start")

        LTXDebug.log("Generating video: \(config.width)x\(config.height), \(config.numFrames) frames")
        LTXDebug.log("Prompt: \(prompt)")

        let textEmbeddings: MLXArray
        let contextMask: MLXArray
        let useCFG = config.cfgScale > 1.0
        let textEncStart = Date()
        LTXMemoryManager.setPhase(.textEncoding)

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
        timings.textEncoding = Date().timeIntervalSince(textEncStart)
        LTXDebug.log("Text encoding: \(String(format: "%.1f", timings.textEncoding))s")
        LTXMemoryManager.logMemoryState("after text encoding")

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

        // 4. Get sigma schedule (with dynamic time shifting matching Diffusers)
        let sigmas: [Float]
        if model.isDistilled {
            scheduler.setTimesteps(
                numSteps: config.numSteps,
                distilled: true,
                latentTokenCount: latentShape.tokenCount
            )
            sigmas = scheduler.sigmas
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
        LTXMemoryManager.setPhase(.denoising)
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

            // Periodic cache clearing to reduce GPU memory fragmentation
            if (step + 1) % 5 == 0 {
                Memory.clearCache()
            }

            // Per-step diagnostics matching Python format
            if profile {
                let vMean = velocity.mean().item(Float.self)
                let vStd = MLX.sqrt(MLX.variance(velocity)).item(Float.self)
                let lMean = latent.mean().item(Float.self)
                let lStd = MLX.sqrt(MLX.variance(latent)).item(Float.self)
                LTXDebug.log("  Step \(step): σ=\(String(format: "%.4f", sigma))→\(String(format: "%.4f", sigmaNext)), vel mean=\(String(format: "%.4f", vMean)), std=\(String(format: "%.4f", vStd)), latent mean=\(String(format: "%.4f", lMean)), std=\(String(format: "%.4f", lStd))")
            }

            timings.denoiseSteps.append(Date().timeIntervalSince(stepStart))
            timings.sampleMemory()
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

        // 7. Unload transformer to free memory for VAE decode
        if memoryOptimization.unloadAfterUse {
            self.transformer = nil
            eval([MLXArray]())
            Memory.clearCache()
            if memoryOptimization.unloadSleepSeconds > 0 {
                try await Task.sleep(for: .seconds(memoryOptimization.unloadSleepSeconds))
                Memory.clearCache()
            }
            LTXDebug.log("Transformer unloaded for VAE decode phase")
            LTXMemoryManager.logMemoryState("after transformer unload")
        }

        // 8. Decode latents to video
        LTXMemoryManager.setPhase(.vaeDecode)
        // Use timestep_conditioning from VAE config.json (Diffusers default: false)
        let vaeTimestep: Float? = vaeDecoder.timestepConditioning ? 0.05 : nil
        LTXDebug.log("Decoding latents to video... (timestep=\(vaeTimestep.map { String($0) } ?? "nil"))")
        let vaeStart = Date()
        let videoTensor = decodeVideo(
            latent: latent, decoder: vaeDecoder, timestep: vaeTimestep,
            temporalTileSize: memoryOptimization.vaeTemporalTileSize,
            temporalTileOverlap: memoryOptimization.vaeTemporalTileOverlap
        )
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

        LTXMemoryManager.logMemoryState("after VAE decode")
        LTXMemoryManager.resetCacheLimit()

        // Capture final peak memory
        timings.capturePeakMemory()

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

    // MARK: - Video + Audio Generation

    /// Result of video+audio generation
    public struct AudioVideoGenerationResult: Sendable {
        /// Video frames tensor (F, H, W, 3)
        public let frames: MLXArray

        /// Audio waveform (B, 2, samples) at 24kHz
        public let audioWaveform: MLXArray

        /// Audio sample rate (24000 Hz)
        public let audioSampleRate: Int

        /// Seed used for generation
        public let seed: UInt64

        /// Total generation time in seconds
        public let generationTime: Double
    }

    /// Generate video with synchronized audio
    ///
    /// Uses the LTX2 dual transformer for joint video/audio denoising.
    /// Requires `loadModels()` + `loadAudioModels()` to have been called.
    ///
    /// - Parameters:
    ///   - prompt: Text description of the video and audio
    ///   - config: Video generation configuration
    ///   - onProgress: Optional progress callback
    /// - Returns: AudioVideoGenerationResult with video frames and audio waveform
    public func generateVideoWithAudio(
        prompt: String,
        config: LTXVideoGenerationConfig,
        onProgress: GenerationProgressCallback? = nil
    ) async throws -> AudioVideoGenerationResult {
        try config.validate()

        guard let textEncoder = textEncoder,
              let ltx2 = ltx2Transformer,
              let vaeDecoder = vaeDecoder,
              let audioVAE = audioVAE,
              let vocoder = vocoder
        else {
            throw LTXError.modelNotLoaded("Audio models not loaded. Call loadModels() + loadAudioModels() first.")
        }

        let generationStart = Date()
        LTXDebug.log("Generating video+audio: \(config.width)x\(config.height), \(config.numFrames) frames")

        // 1. Text encoding (with audio connector)
        LTXMemoryManager.setPhase(.textEncoding)

        if !isGemmaLoaded {
            LTXDebug.log("Reloading Gemma model...")
            let paths = try await downloader.downloadGemma(model: model)
            gemmaModel = try Gemma3WeightLoader.loadModel(from: paths.modelDir)
            tokenizer = try await AutoTokenizer.from(modelFolder: paths.tokenizerDir)
        }

        let effectivePrompt: String
        if config.enhancePrompt {
            effectivePrompt = enhancePrompt(prompt)
        } else {
            effectivePrompt = prompt
        }

        // Encode text with both video and audio connectors
        let (inputIds, attentionMask) = tokenizePrompt(effectivePrompt, maxLength: textMaxLength)

        guard let gemma = gemmaModel else {
            throw LTXError.modelNotLoaded("Gemma model not loaded")
        }

        let (_, allHiddenStates) = gemma(inputIds, attentionMask: attentionMask, outputHiddenStates: true)
        guard let states = allHiddenStates, states.count == gemma.config.hiddenLayers + 1 else {
            throw LTXError.generationFailed("Failed to extract Gemma hidden states")
        }

        // Text encoder produces both video and audio embeddings
        let encoderOutput = textEncoder.encodeFromHiddenStates(
            hiddenStates: states,
            attentionMask: attentionMask,
            paddingSide: "left"
        )
        let videoTextEmbeddings = encoderOutput.videoEncoding
        let audioTextEmbeddings = encoderOutput.audioEncoding ?? videoTextEmbeddings
        let textMask = encoderOutput.attentionMask
        MLX.eval(videoTextEmbeddings, audioTextEmbeddings, textMask)

        LTXDebug.log("Video text: \(videoTextEmbeddings.shape), Audio text: \(audioTextEmbeddings.shape)")

        // Unload Gemma
        self.gemmaModel = nil
        self.tokenizer = nil
        Memory.clearCache()

        // 2. Create latent shapes
        let latentShape = VideoLatentShape.fromPixelDimensions(
            batch: 1, channels: 128,
            frames: config.numFrames, height: config.height, width: config.width
        )

        let audioNumFrames = computeAudioLatentFrames(videoFrames: config.numFrames)
        LTXDebug.log("Audio latent frames: \(audioNumFrames)")

        // 3. Generate noise
        if let seed = config.seed {
            MLXRandom.seed(seed)
        }

        // Video noise (float32)
        var videoLatent = generateNoise(shape: latentShape, seed: config.seed)
        MLX.eval(videoLatent)

        // Audio noise (float32, drawn after video noise from same RNG)
        let audioLatent = MLXRandom.normal(
            [1, Self.audioLatentChannels, audioNumFrames, Self.audioLatentMelBins]
        ).asType(.float32)
        MLX.eval(audioLatent)

        // Pack audio for transformer
        var audioLatentPacked = packAudioLatents(audioLatent)

        // 4. Sigma schedule (distilled only for now)
        scheduler.setTimesteps(
            numSteps: config.numSteps,
            distilled: model.isDistilled,
            latentTokenCount: latentShape.tokenCount
        )
        let sigmas = scheduler.sigmas

        // Create separate audio scheduler (same sigmas)
        let audioScheduler = LTXScheduler(isDistilled: model.isDistilled)
        audioScheduler.setTimesteps(
            numSteps: config.numSteps,
            distilled: model.isDistilled,
            latentTokenCount: latentShape.tokenCount
        )

        // Scale initial noise
        videoLatent = videoLatent * sigmas[0]
        audioLatentPacked = audioLatentPacked * sigmas[0]

        // 5. Denoising loop
        LTXMemoryManager.setPhase(.denoising)
        LTXDebug.log("Starting dual video+audio denoising (\(config.numSteps) steps)...")

        for step in 0..<config.numSteps {
            let stepStart = Date()
            let sigma = sigmas[step]
            let sigmaNext = sigmas[step + 1]

            onProgress?(GenerationProgress(
                currentStep: step, totalSteps: config.numSteps, sigma: sigma
            ))

            let timestep = MLXArray([sigma])

            // Patchify video for transformer
            let videoPatchified = patchify(videoLatent).asType(.bfloat16)
            let audioPatchified = audioLatentPacked.asType(.bfloat16)

            // Forward pass through dual transformer
            let (videoVelPred, audioVelPred) = ltx2(
                videoLatent: videoPatchified,
                audioLatent: audioPatchified,
                videoContext: videoTextEmbeddings.asType(.bfloat16),
                audioContext: audioTextEmbeddings.asType(.bfloat16),
                videoTimesteps: timestep,
                audioTimesteps: timestep,
                videoContextMask: textMask,
                audioContextMask: textMask,
                videoLatentShape: (frames: latentShape.frames, height: latentShape.height, width: latentShape.width),
                audioNumFrames: audioNumFrames
            )

            // Unpatchify video velocity
            let videoVelocity = unpatchify(videoVelPred, shape: latentShape).asType(.float32)
            // Audio velocity is already in packed form (B, T, 128)
            let audioVelocity = audioVelPred.asType(.float32)

            // Euler step for video
            videoLatent = scheduler.step(
                latent: videoLatent, velocity: videoVelocity,
                sigma: sigma, sigmaNext: sigmaNext
            )

            // Euler step for audio (same formula)
            audioLatentPacked = audioLatentPacked + (sigmaNext - sigma) * audioVelocity

            MLX.eval(videoLatent, audioLatentPacked)
            if (step + 1) % 5 == 0 { Memory.clearCache() }

            LTXDebug.log("Step \(step)/\(config.numSteps): σ=\(String(format: "%.4f", sigma))→\(String(format: "%.4f", sigmaNext)), time=\(String(format: "%.1f", Date().timeIntervalSince(stepStart)))s")
        }

        // 6. Unload transformer
        if memoryOptimization.unloadAfterUse {
            self.ltx2Transformer = nil
            Memory.clearCache()
            LTXDebug.log("LTX2 Transformer unloaded")
        }

        // 7. Decode video
        LTXMemoryManager.setPhase(.vaeDecode)
        let vaeTimestep: Float? = vaeDecoder.timestepConditioning ? 0.05 : nil
        let videoTensor = decodeVideo(
            latent: videoLatent, decoder: vaeDecoder, timestep: vaeTimestep,
            temporalTileSize: memoryOptimization.vaeTemporalTileSize,
            temporalTileOverlap: memoryOptimization.vaeTemporalTileOverlap
        )
        MLX.eval(videoTensor)

        // Trim video frames
        let trimmedVideo: MLXArray
        if videoTensor.dim(0) > config.numFrames {
            trimmedVideo = videoTensor[0..<config.numFrames]
        } else {
            trimmedVideo = videoTensor
        }

        // 8. Decode audio: unpack → denormalize → AudioVAE → Vocoder
        LTXDebug.log("Decoding audio latents...")
        let audioLatentUnpacked = unpackAudioLatents(audioLatentPacked, numFrames: audioNumFrames)
        let audioWaveform = decodeAudio(
            latents: audioLatentUnpacked,
            audioVAE: audioVAE,
            vocoder: vocoder
        )
        MLX.eval(audioWaveform)
        LTXDebug.log("Audio waveform: \(audioWaveform.shape)")

        let generationTime = Date().timeIntervalSince(generationStart)
        let usedSeed = config.seed ?? 0

        return AudioVideoGenerationResult(
            frames: trimmedVideo,
            audioWaveform: audioWaveform,
            audioSampleRate: vocoder.outputSampleRate,
            seed: usedSeed,
            generationTime: generationTime
        )
    }

    // MARK: - Image-to-Video Generation

    /// Load VAE encoder weights from the VAE safetensors file
    ///
    /// The encoder weights share the same file as the decoder (vae/diffusion_pytorch_model.safetensors)
    /// but are prefixed with `encoder.` which loadVAEWeights() normally skips.
    private func loadVAEEncoder() async throws {
        if vaeEncoder != nil { return }  // Already loaded

        LTXDebug.log("Loading VAE encoder...")
        let vaePath = try await downloader.downloadVAE(model: model, progress: nil)
        let encoderWeights = try LTXWeightLoader.loadVAEEncoderWeights(from: vaePath.path)

        let encoder = VideoEncoder()
        try LTXWeightLoader.applyVAEEncoderWeights(encoderWeights, to: encoder)
        eval(encoder.parameters())
        Memory.clearCache()

        self.vaeEncoder = encoder
        LTXDebug.log("VAE encoder loaded (\(encoderWeights.count) weights)")
    }

    /// Unload VAE encoder to free memory
    private func unloadVAEEncoder() {
        self.vaeEncoder = nil
        eval([MLXArray]())
        Memory.clearCache()
        LTXDebug.log("VAE encoder unloaded")
    }

    /// Encode an image into latent space using the VAE encoder
    ///
    /// - Parameters:
    ///   - imagePath: Path to input image
    ///   - width: Target video width
    ///   - height: Target video height
    /// - Returns: Image latent tensor (1, 128, 1, H/32, W/32)
    private func encodeImage(path imagePath: String, width: Int, height: Int) async throws -> MLXArray {
        // Load and resize image
        let imageTensor = try loadImage(from: imagePath, width: width, height: height)
        MLX.eval(imageTensor)
        LTXDebug.log("Image loaded: \(imageTensor.shape)")

        // Load encoder if needed
        try await loadVAEEncoder()

        guard let encoder = vaeEncoder else {
            throw LTXError.modelNotLoaded("VAE encoder failed to load")
        }

        // Encode: (1, 3, 1, H, W) -> (1, 128, 1, H/32, W/32)
        let latent = encoder(imageTensor)
        MLX.eval(latent)
        LTXDebug.log("Image encoded to latent: \(latent.shape)")

        // Normalize using VAE per-channel statistics (same as decoder uses to denormalize)
        guard let vaeDecoder = vaeDecoder else {
            throw LTXError.modelNotLoaded("VAE decoder not loaded (needed for latent statistics)")
        }
        let mean5d = vaeDecoder.meanOfMeans.asType(.float32).reshaped([1, -1, 1, 1, 1])
        let std5d = vaeDecoder.stdOfMeans.asType(.float32).reshaped([1, -1, 1, 1, 1])
        let normalizedLatent = (latent.asType(.float32) - mean5d) / std5d
        MLX.eval(normalizedLatent)

        LTXDebug.log("Normalized image latent: mean=\(normalizedLatent.mean().item(Float.self)), std=\(MLX.sqrt(MLX.variance(normalizedLatent)).item(Float.self))")

        return normalizedLatent
    }

    /// Generate video conditioned on an input image (image-to-video)
    ///
    /// The first frame of the generated video matches the input image.
    /// The remaining frames are generated from noise and denoised using
    /// the same transformer and scheduler as text-to-video.
    ///
    /// Memory management:
    /// - Phase 1: Load VAE encoder, encode image, unload encoder
    /// - Phase 2: Encode text (Gemma), unload Gemma
    /// - Phase 3: Denoise with transformer
    /// - Phase 4: Unload transformer, decode with VAE decoder
    ///
    /// - Parameters:
    ///   - prompt: Text description of the desired video motion
    ///   - negativePrompt: Optional negative prompt for CFG
    ///   - config: Video generation configuration (must have imagePath set)
    ///   - onProgress: Optional progress callback
    ///   - profile: Enable performance profiling
    /// - Returns: Video generation result with frames
    public func generateVideoFromImage(
        prompt: String,
        negativePrompt: String? = nil,
        config: LTXVideoGenerationConfig,
        onProgress: GenerationProgressCallback? = nil,
        profile: Bool = false
    ) async throws -> VideoGenerationResult {
        try config.validate()
        var timings = GenerationTimings()

        guard let imagePath = config.imagePath else {
            throw LTXError.invalidConfiguration("imagePath must be set for image-to-video generation")
        }

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
        LTXMemoryManager.logMemoryState("i2v generation start")

        // === Phase 1: Encode image ===
        LTXDebug.log("=== Phase 1: Image Encoding ===")
        let imgEncStart = Date()
        let imageLatent = try await encodeImage(path: imagePath, width: config.width, height: config.height)
        LTXDebug.log("Image encoding: \(String(format: "%.1f", Date().timeIntervalSince(imgEncStart)))s")

        // Unload encoder to free memory
        unloadVAEEncoder()
        LTXMemoryManager.logMemoryState("after image encoding")

        // === Phase 2: Text Encoding ===
        LTXDebug.log("=== Phase 2: Text Encoding ===")
        let textEncStart = Date()
        let useCFG = config.cfgScale > 1.0
        LTXMemoryManager.setPhase(.textEncoding)

        // Reload Gemma if needed
        if !isGemmaLoaded {
            LTXDebug.log("Reloading Gemma model...")
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
        }

        // Optionally enhance prompt
        let effectivePrompt: String
        if config.enhancePrompt {
            effectivePrompt = enhancePrompt(prompt)
        } else {
            effectivePrompt = prompt
        }

        // Encode prompt
        let (promptEmbeddings, promptMask) = encodePrompt(effectivePrompt, encoder: textEncoder)
        MLX.eval(promptEmbeddings, promptMask)

        let textEmbeddings: MLXArray
        let contextMask: MLXArray
        if useCFG {
            let negPromptText = negativePrompt ?? DEFAULT_NEGATIVE_PROMPT
            let (negEmbeddings, negMask) = encodePrompt(negPromptText, encoder: textEncoder)
            MLX.eval(negEmbeddings, negMask)
            textEmbeddings = MLX.concatenated([negEmbeddings, promptEmbeddings], axis: 0)
            contextMask = MLX.concatenated([negMask, promptMask], axis: 0)
            MLX.eval(textEmbeddings, contextMask)
        } else {
            textEmbeddings = promptEmbeddings
            contextMask = promptMask
        }

        // Unload Gemma
        self.gemmaModel = nil
        self.tokenizer = nil
        Memory.clearCache()

        timings.textEncoding = Date().timeIntervalSince(textEncStart)
        LTXDebug.log("Text encoding: \(String(format: "%.1f", timings.textEncoding))s")

        // === Phase 3: Denoising ===
        LTXDebug.log("=== Phase 3: Denoising ===")
        LTXMemoryManager.setPhase(.denoising)

        // Create latent shape
        let latentShape = VideoLatentShape.fromPixelDimensions(
            batch: 1, channels: 128,
            frames: config.numFrames,
            height: config.height,
            width: config.width
        )
        LTXDebug.log("Latent shape: \(latentShape.frames)x\(latentShape.height)x\(latentShape.width)")

        // Seed
        if let seed = config.seed {
            MLXRandom.seed(seed)
        }

        // Cross-attention scaling
        if config.crossAttentionScale != 1.0 {
            transformer.setCrossAttentionScale(config.crossAttentionScale)
        }

        // Get sigma schedule
        // Use distilled sigmas when running distilled-style (8 steps, no CFG) —
        // matches the two-stage pipeline logic. The distilled LoRA was trained
        // with this specific schedule.
        let sigmas: [Float]
        if model.isDistilled || (config.numSteps <= 8 && config.cfgScale <= 1.0) {
            // Apply dynamic time shifting to distilled sigmas (matching Diffusers)
            scheduler.setTimesteps(
                numSteps: config.numSteps,
                distilled: true,
                latentTokenCount: latentShape.tokenCount
            )
            sigmas = scheduler.sigmas
            LTXDebug.log("Using distilled sigma schedule with dynamic shift (\(sigmas.count - 1) steps)")
        } else {
            scheduler.setTimesteps(
                numSteps: config.numSteps,
                distilled: false,
                latentTokenCount: latentShape.tokenCount
            )
            sigmas = scheduler.sigmas
            LTXDebug.log("Using dev sigma schedule (\(sigmas.count - 1) steps)")
        }
        LTXDebug.log("Sigma schedule: \(sigmas.map { String(format: "%.4f", $0) })")

        // Generate noise for full video
        var latent = generateNoise(shape: latentShape, seed: config.seed)

        // Scale noise by first sigma
        latent = latent * sigmas[0]

        // Replace first frame with image latent (conditioned frame, no noise)
        latent[0..., 0..., 0..<1, 0..., 0...] = imageLatent
        MLX.eval(latent)

        // Create per-token conditioning mask: 1.0 for frame 0 tokens, 0.0 for rest
        // In patchified space, tokens are ordered (F, H, W), so frame 0 = first H'*W' tokens
        let tokensPerFrame = latentShape.height * latentShape.width
        let frame0Mask = MLXArray.ones([1, tokensPerFrame])
        let otherMask = MLXArray.zeros([1, latentShape.tokenCount - tokensPerFrame])
        let conditioningMask = MLX.concatenated([frame0Mask, otherMask], axis: 1)
        MLX.eval(conditioningMask)

        LTXDebug.log("I2V conditioning: per-token timestep (frame 0=0, rest=sigma) + SLICE Euler step")
        LTXDebug.log("Conditioning mask: \(tokensPerFrame) conditioned tokens / \(latentShape.tokenCount) total")
        LTXDebug.log("Image cond noise scale: \(config.imageCondNoiseScale)")

        // Denoise with per-token timestep conditioning + slice frame freezing
        latent = denoise(
            latent: latent,
            sigmas: sigmas,
            textEmbeddings: textEmbeddings,
            promptEmbeddings: promptEmbeddings,
            contextMask: contextMask,
            latentShape: latentShape,
            config: config,
            transformer: transformer,
            useCFG: useCFG,
            conditioningMask: conditioningMask,
            conditionedLatent: imageLatent,
            onProgress: onProgress,
            profile: profile,
            timings: &timings
        )

        // === Phase 4: VAE Decode ===
        LTXDebug.log("=== Phase 4: VAE Decode ===")
        if memoryOptimization.unloadAfterUse {
            self.transformer = nil
            eval([MLXArray]())
            Memory.clearCache()
            if memoryOptimization.unloadSleepSeconds > 0 {
                try await Task.sleep(for: .seconds(memoryOptimization.unloadSleepSeconds))
                Memory.clearCache()
            }
            LTXDebug.log("Transformer unloaded for VAE decode")
        }

        LTXMemoryManager.setPhase(.vaeDecode)
        let vaeTimestep: Float? = vaeDecoder.timestepConditioning ? 0.05 : nil
        let vaeStart = Date()
        let videoTensor = decodeVideo(
            latent: latent, decoder: vaeDecoder, timestep: vaeTimestep,
            temporalTileSize: memoryOptimization.vaeTemporalTileSize,
            temporalTileOverlap: memoryOptimization.vaeTemporalTileOverlap
        )
        MLX.eval(videoTensor)
        timings.vaeDecode = Date().timeIntervalSince(vaeStart)

        // Trim to requested frame count
        let trimmedVideo: MLXArray
        if videoTensor.dim(0) > config.numFrames {
            trimmedVideo = videoTensor[0..<config.numFrames]
        } else {
            trimmedVideo = videoTensor
        }

        LTXMemoryManager.logMemoryState("after i2v VAE decode")
        LTXMemoryManager.resetCacheLimit()
        timings.capturePeakMemory()

        let generationTime = Date().timeIntervalSince(generationStart)
        return VideoGenerationResult(
            frames: trimmedVideo,
            seed: config.seed ?? 0,
            generationTime: generationTime,
            timings: profile ? timings : nil
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
        conditioningMask: MLXArray? = nil,
        conditionedLatent: MLXArray? = nil,
        onProgress: GenerationProgressCallback? = nil,
        profile: Bool = false,
        timings: inout GenerationTimings
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

            // Inject noise to conditioned frame BEFORE transformer (Diffusers pattern)
            // noised = init_latents + noise_scale * noise * sigma^2
            // Quadratic decay: more noise at early steps (high sigma), less at late steps
            if let condLatent = conditionedLatent, config.imageCondNoiseScale > 0, sigma > 0 {
                let injectionNoise = MLXRandom.normal(condLatent.shape)
                let noisedFrame0 = condLatent + MLXArray(config.imageCondNoiseScale) * injectionNoise * MLXArray(sigma * sigma)
                currentLatent[0..., 0..., 0..<1, 0..., 0...] = noisedFrame0
            }

            let latentInput: MLXArray
            let timestep: MLXArray
            let isI2V = conditioningMask != nil

            if useCFG {
                latentInput = prepareForCFG(currentLatent)
                if isI2V, let mask = conditioningMask {
                    // Per-token timestep: frame 0 tokens = 0 (clean), others = sigma
                    // Double the mask for CFG (uncond + cond both get same mask)
                    let perToken = MLXArray(sigma) * (1 - mask)  // (1, T)
                    timestep = MLX.concatenated([perToken, perToken], axis: 0)  // (2, T)
                } else {
                    timestep = MLXArray([sigma, sigma])
                }
            } else {
                latentInput = currentLatent
                if isI2V, let mask = conditioningMask {
                    // Per-token timestep: frame 0 tokens = 0 (clean), others = sigma
                    timestep = MLXArray(sigma) * (1 - mask)  // (1, T)
                } else {
                    timestep = MLXArray([sigma])
                }
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

            // I2V diagnostics: print velocity and latent stats per frame for every step
            if isI2V {
                let vf32 = velocity.asType(.float32)
                let lf32 = currentLatent.asType(.float32)
                for f in 0..<min(latentShape.frames, 4) {
                    let fv = vf32[0..., 0..., f..<(f+1), 0..., 0...]
                    let fl = lf32[0..., 0..., f..<(f+1), 0..., 0...]
                    let fvAbsMean = MLX.abs(fv).mean().item(Float.self)
                    let flMean = fl.mean().item(Float.self)
                    let flStd = MLX.sqrt(MLX.variance(fl)).item(Float.self)
                    print("  [I2V] Step \(step) Frame \(f): vel_abs=\(String(format: "%.4f", fvAbsMean)), lat mean=\(String(format: "%.4f", flMean)) std=\(String(format: "%.4f", flStd))")
                }
                // Timestep info on first step
                if step == 0 {
                    let tsEval = timestep.asType(.float32)
                    if timestep.ndim >= 2 {
                        let tokensPerFrame = latentShape.height * latentShape.width
                        let ts0 = tsEval[0, 0].item(Float.self)
                        let ts1 = tsEval[0, tokensPerFrame].item(Float.self)
                        print("  [I2V] timestep shape=\(timestep.shape), frame0_ts=\(String(format: "%.4f", ts0)), frame1_ts=\(String(format: "%.4f", ts1))")
                    }
                }
            }

            // STG
            if config.stgScale > 0 {
                transformer.setSTGSkipFlags(skipSelfAttention: true, blockIndices: config.stgBlocks)
                let perturbedPatchified = patchify(currentLatent).asType(.bfloat16)
                let perturbedTimestep: MLXArray
                if isI2V, let mask = conditioningMask {
                    perturbedTimestep = MLXArray(sigma) * (1 - mask)  // (1, T) per-token
                } else {
                    perturbedTimestep = MLXArray([sigma])
                }
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

            if isI2V {
                // SLICE approach (matches Diffusers LTX2ImageToVideoPipeline exactly):
                // 1. Euler step only on frames 1+ (frame 0 stays clean)
                // 2. Re-attach unchanged frame 0
                let velocitySlice = velocity[0..., 0..., 1..., 0..., 0...]
                let latentSlice = currentLatent[0..., 0..., 1..., 0..., 0...]
                let steppedSlice = scheduler.step(
                    latent: latentSlice,
                    velocity: velocitySlice,
                    sigma: sigma,
                    sigmaNext: sigmaNext
                )
                let frame0 = currentLatent[0..., 0..., 0..<1, 0..., 0...]
                currentLatent = MLX.concatenated([frame0, steppedSlice], axis: 2)
            } else {
                // Standard Euler step for T2V — step all frames
                currentLatent = scheduler.step(
                    latent: currentLatent,
                    velocity: velocity,
                    sigma: sigma,
                    sigmaNext: sigmaNext
                )
            }
            MLX.eval(currentLatent)

            // Periodic cache clearing to reduce GPU memory fragmentation
            if (step + 1) % 5 == 0 {
                Memory.clearCache()
            }

            let stepDuration = Date().timeIntervalSince(stepStart)
            timings.denoiseSteps.append(stepDuration)
            timings.sampleMemory()

            if profile {
                let vMean = velocityPred.mean().item(Float.self)
                let vStd = MLX.sqrt(MLX.variance(velocityPred)).item(Float.self)
                let lMean = currentLatent.mean().item(Float.self)
                let lStd = MLX.sqrt(MLX.variance(currentLatent)).item(Float.self)
                LTXDebug.log("  Step \(step): σ=\(String(format: "%.4f", sigma))→\(String(format: "%.4f", sigmaNext)), vel mean=\(String(format: "%.4f", vMean)), std=\(String(format: "%.4f", vStd)), latent mean=\(String(format: "%.4f", lMean)), std=\(String(format: "%.4f", lStd))")
                LTXDebug.log("  Step \(step) time: \(String(format: "%.2f", stepDuration))s")
                // Per-frame velocity diagnostics for I2V
                if conditionedLatent != nil && step % 2 == 0 {
                    for f in 0..<min(latentShape.frames, 4) {
                        let frameVel = velocity[0..., 0..., f..<(f+1), 0..., 0...]
                        let fvMean = frameVel.mean().item(Float.self)
                        let fvStd = MLX.sqrt(MLX.variance(frameVel)).item(Float.self)
                        let frameLat = currentLatent[0..., 0..., f..<(f+1), 0..., 0...]
                        let flMean = frameLat.mean().item(Float.self)
                        let flStd = MLX.sqrt(MLX.variance(frameLat)).item(Float.self)
                        LTXDebug.log("    Frame \(f): vel(mean=\(String(format: "%.4f", fvMean)), std=\(String(format: "%.4f", fvStd))) lat(mean=\(String(format: "%.4f", flMean)), std=\(String(format: "%.4f", flStd)))")
                    }
                }
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

        // Two-stage requires width/height divisible by 64
        guard config.width % 64 == 0 && config.height % 64 == 0 else {
            throw LTXError.invalidConfiguration("Two-stage requires width and height divisible by 64. Got \(config.width)x\(config.height)")
        }

        let halfWidth = config.width / 2
        let halfHeight = config.height / 2

        LTXDebug.log("Two-stage generation: \(halfWidth)x\(halfHeight) → \(config.width)x\(config.height)")
        LTXMemoryManager.logMemoryState("two-stage start")

        let isI2V = config.imagePath != nil

        // 0. Encode image at half-res if i2v
        var halfResImageLatent: MLXArray? = nil
        if let imagePath = config.imagePath {
            LTXDebug.log("Two-stage I2V: encoding image at \(halfWidth)x\(halfHeight)")
            halfResImageLatent = try await encodeImage(path: imagePath, width: halfWidth, height: halfHeight)
            unloadVAEEncoder()
        }

        // 0b. Optionally enhance prompt
        let effectivePrompt: String
        if config.enhancePrompt {
            LTXDebug.log("Enhancing prompt with Gemma...")
            effectivePrompt = enhancePrompt(prompt)
        } else {
            effectivePrompt = prompt
        }

        // 1. Encode text (shared between both stages)
        LTXMemoryManager.setPhase(.textEncoding)
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
        timings.textEncoding = Date().timeIntervalSince(textEncStart)
        LTXDebug.log("Text encoding: \(String(format: "%.1f", timings.textEncoding))s")

        // Unload Gemma to free memory for denoising
        LTXDebug.log("Unloading Gemma model to free memory...")
        self.gemmaModel = nil
        self.tokenizer = nil
        Memory.clearCache()
        LTXDebug.log("Gemma model unloaded")

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
        LTXMemoryManager.setPhase(.denoising)
        LTXDebug.log("=== Stage 1: Half-resolution denoising (\(config.numSteps) steps) ===")
        let stage1Start = Date()

        let stage1Shape = VideoLatentShape.fromPixelDimensions(
            batch: 1, channels: 128,
            frames: config.numFrames,
            height: halfHeight,
            width: halfWidth
        )
        LTXDebug.log("Stage 1 latent: \(stage1Shape.frames)x\(stage1Shape.height)x\(stage1Shape.width)")

        // Determine stage 1 sigmas based on model configuration
        let stage1Sigmas: [Float]
        if config.numSteps <= 8 && config.cfgScale <= 1.0 {
            // Distilled or LoRA mode: use distilled sigmas with dynamic shifting
            let stage1Scheduler = LTXScheduler(isDistilled: true)
            stage1Scheduler.setTimesteps(
                numSteps: config.numSteps,
                distilled: true,
                latentTokenCount: stage1Shape.tokenCount
            )
            stage1Sigmas = stage1Scheduler.sigmas
            LTXDebug.log("Stage 1 using distilled sigmas with dynamic shift (\(stage1Sigmas.count - 1) steps)")
        } else {
            // Dev model: compute sigmas via scheduler with token-dependent shift
            let stage1Scheduler = LTXScheduler(isDistilled: false)
            stage1Scheduler.setTimesteps(
                numSteps: config.numSteps,
                distilled: false,
                latentTokenCount: stage1Shape.tokenCount
            )
            stage1Sigmas = stage1Scheduler.sigmas
            LTXDebug.log("Stage 1 using dev sigmas (\(stage1Sigmas.count - 1) steps)")
        }

        // Generate noise and scale by initial sigma
        var latent = generateNoise(shape: stage1Shape, seed: config.seed)
        latent = latent * stage1Sigmas[0]

        // I2V: replace first frame with image latent and create conditioning mask
        var stage1CondMask: MLXArray? = nil
        if let imgLatent = halfResImageLatent {
            latent[0..., 0..., 0..<1, 0..., 0...] = imgLatent
            let s1TokensPerFrame = stage1Shape.height * stage1Shape.width
            let s1Frame0Mask = MLXArray.ones([1, s1TokensPerFrame])
            let s1OtherMask = MLXArray.zeros([1, stage1Shape.tokenCount - s1TokensPerFrame])
            stage1CondMask = MLX.concatenated([s1Frame0Mask, s1OtherMask], axis: 1)
            MLX.eval(stage1CondMask!)
            LTXDebug.log("Stage 1 I2V: frame 0 conditioned (\(s1TokensPerFrame)/\(stage1Shape.tokenCount) tokens)")
        }
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
            conditioningMask: stage1CondMask,
            conditionedLatent: halfResImageLatent,
            onProgress: onProgress,
            profile: profile,
            timings: &timings
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

        // I2V stage 2: encode image at full resolution and condition frame 0
        var stage2CondMask: MLXArray? = nil
        var stage2CondLatent: MLXArray? = nil
        if isI2V, let imagePath = config.imagePath {
            LTXDebug.log("Stage 2 I2V: encoding image at \(config.width)x\(config.height)")
            let fullResImageLatent = try await encodeImage(path: imagePath, width: config.width, height: config.height)
            unloadVAEEncoder()
            latent[0..., 0..., 0..<1, 0..., 0...] = fullResImageLatent
            stage2CondLatent = fullResImageLatent
            let s2TokensPerFrame = stage2Shape.height * stage2Shape.width
            let s2Frame0Mask = MLXArray.ones([1, s2TokensPerFrame])
            let s2OtherMask = MLXArray.zeros([1, stage2Shape.tokenCount - s2TokensPerFrame])
            stage2CondMask = MLX.concatenated([s2Frame0Mask, s2OtherMask], axis: 1)
            MLX.eval(stage2CondMask!)
            LTXDebug.log("Stage 2 I2V: frame 0 conditioned (\(s2TokensPerFrame)/\(stage2Shape.tokenCount) tokens)")
        }
        MLX.eval(latent)
        LTXDebug.log("Added stage 2 noise (σ=\(noiseScale))")
        LTXDebug.log("Stage 2 input stats: mean=\(latent.mean().item(Float.self)), std=\(MLX.sqrt(MLX.variance(latent)).item(Float.self))")

        // Denoise stage 2 — ALWAYS without CFG (distilled refinement sigmas)
        // Use prompt-only embeddings, not CFG-concatenated [neg, pos]
        latent = denoise(
            latent: latent,
            sigmas: stage2Sigmas,
            textEmbeddings: promptEmbeddings,
            promptEmbeddings: promptEmbeddings,
            contextMask: promptMask,
            latentShape: stage2Shape,
            config: config,
            transformer: transformer,
            useCFG: false,
            conditioningMask: stage2CondMask,
            conditionedLatent: stage2CondLatent,
            onProgress: onProgress,
            profile: profile,
            timings: &timings
        )
        LTXDebug.log("Stage 2 complete: \(String(format: "%.1f", Date().timeIntervalSince(stage2Start)))s")
        LTXDebug.log("Stage 2 output stats: mean=\(latent.mean().item(Float.self)), std=\(MLX.sqrt(MLX.variance(latent)).item(Float.self)), min=\(latent.min().item(Float.self)), max=\(latent.max().item(Float.self))")

        // Unload transformer to free memory for VAE decode
        if memoryOptimization.unloadAfterUse {
            self.transformer = nil
            eval([MLXArray]())
            Memory.clearCache()
            if memoryOptimization.unloadSleepSeconds > 0 {
                try await Task.sleep(for: .seconds(memoryOptimization.unloadSleepSeconds))
                Memory.clearCache()
            }
            LTXDebug.log("Transformer unloaded for VAE decode phase")
            LTXMemoryManager.logMemoryState("after transformer unload")
        }

        // === VAE Decode ===
        LTXMemoryManager.setPhase(.vaeDecode)
        // Use timestep_conditioning from VAE config.json
        let vaeTimestep2: Float? = vaeDecoder.timestepConditioning ? 0.05 : nil
        LTXDebug.log("Decoding latents to video... (timestep=\(vaeTimestep2.map { String($0) } ?? "nil"))")
        let vaeStart = Date()
        let videoTensor = decodeVideo(
            latent: latent, decoder: vaeDecoder, timestep: vaeTimestep2,
            temporalTileSize: memoryOptimization.vaeTemporalTileSize,
            temporalTileOverlap: memoryOptimization.vaeTemporalTileOverlap
        )
        MLX.eval(videoTensor)
        timings.vaeDecode = Date().timeIntervalSince(vaeStart)
        LTXDebug.log("VAE decode: \(String(format: "%.1f", timings.vaeDecode))s")

        // Trim to requested frame count
        let trimmedVideo: MLXArray
        if videoTensor.dim(0) > config.numFrames {
            trimmedVideo = videoTensor[0..<config.numFrames]
        } else {
            trimmedVideo = videoTensor
        }

        LTXMemoryManager.logMemoryState("after two-stage VAE decode")
        LTXMemoryManager.resetCacheLimit()

        // Capture final peak memory
        timings.capturePeakMemory()

        let generationTime = Date().timeIntervalSince(generationStart)
        LTXDebug.log("Total two-stage time: \(String(format: "%.1f", generationTime))s")

        return VideoGenerationResult(
            frames: trimmedVideo,
            seed: config.seed ?? 0,
            generationTime: generationTime,
            timings: profile ? timings : nil
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
