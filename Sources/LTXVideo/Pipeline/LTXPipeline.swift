// LTXPipeline.swift - Main Video Generation Pipeline for LTX-2
// Copyright 2025

import CoreGraphics
import CoreImage
import Foundation
@preconcurrency import MLX
import MLXLMCommon
import MLXRandom
import MLXNN
import MLXVLM
import MLXHuggingFace
import HuggingFace  // Required: #huggingFaceLoadModelContainer macro expands to HubClient references
import Tokenizers
import Hub

// MARK: - Pipeline Progress

/// Progress information emitted during the denoising phase of generation.
///
/// Passed to the `onProgress` callback of ``LTXPipeline/generateVideo(prompt:config:upscalerWeightsPath:onProgress:)``.
///
/// ## Example
/// ```swift
/// let result = try await pipeline.generateVideo(
///     prompt: "A sunset",
///     config: config,
///     upscalerWeightsPath: upscalerPath,
///     onProgress: { progress in
///         print("[\(Int(progress.progress * 100))%] \(progress.status)")
///     }
/// )
/// ```
public struct GenerationProgress: Sendable {
    /// Pipeline phase
    public enum Phase: String, Sendable {
        /// Main denoising (single-stage or stage 1 of two-stage)
        case denoising = "denoising"
        /// Spatial upscale between stages
        case upscaling = "upscaling"
        /// Refinement at full resolution (stage 2 of two-stage)
        case refinement = "refinement"
        /// VAE decoding latents to pixel frames
        case decoding = "decoding"
        /// MP4 video export (H.264 encoding)
        case exporting = "exporting"
    }

    /// Current step within the current phase (0-indexed)
    public let currentStep: Int

    /// Total steps across all phases
    public let totalSteps: Int

    /// Current noise sigma value (decreases from 1.0 toward 0.0)
    public let sigma: Float

    /// Current phase
    public let phase: Phase

    /// Progress fraction from 0.0 (start) to 1.0 (complete)
    public var progress: Double {
        Double(currentStep + 1) / Double(totalSteps)
    }

    /// Human-readable status, e.g. `"Step 3/11 [denoising] (σ=0.7250)"`
    public var status: String {
        switch phase {
        case .denoising, .refinement:
            return "Step \(currentStep + 1)/\(totalSteps) [\(phase.rawValue)] (σ=\(String(format: "%.4f", sigma)))"
        case .upscaling, .decoding, .exporting:
            return "[\(phase.rawValue)]"
        }
    }
}

/// Callback invoked at each denoising step with progress information.
public typealias GenerationProgressCallback = @Sendable (GenerationProgress) -> Void

/// Callback invoked with intermediate frame previews during generation.
/// Parameters: frame index and the rendered CGImage.
public typealias FramePreviewCallback = @Sendable (Int, CGImage) -> Void

// MARK: - Transformer Reference (for training)

/// A Sendable wrapper for a Module reference, used to pass transformers
/// across actor isolation boundaries for training.
///
/// - Warning: The caller must ensure single-threaded access to the wrapped module.
final class TransformerRef: @unchecked Sendable {
    let module: Module
    init(_ module: Module) { self.module = module }
}

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
/// let upscalerPath = try await pipeline.downloadUpscalerWeights()
/// let result = try await pipeline.generateVideo(
///     prompt: "A cat walking in a garden",
///     config: LTXVideoGenerationConfig(width: 768, height: 512, numFrames: 121),
///     upscalerWeightsPath: upscalerPath
/// )
/// ```
///
/// ## Audio Generation
/// ```swift
/// try await pipeline.loadAudioModels()
/// let result = try await pipeline.generateVideo(
///     prompt: "Birds singing in a forest",
///     config: config,
///     upscalerWeightsPath: upscalerPath
/// )
/// // result.audioWaveform and result.audioSampleRate are populated
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
    private var tokenizer: Tokenizers.Tokenizer?

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

    // MARK: - LoRA State

    /// Original (pre-fusion) transformer weights, stored for unfusing
    private var loraOriginalWeights: [String: MLXArray]? = nil

    /// Path to the currently fused LoRA file
    private var loraFusedPath: String? = nil

    /// Scale used for the currently fused LoRA
    private var loraFusedScale: Float = 1.0

    /// Whether a LoRA is currently fused into the transformer
    public var isLoRAFused: Bool { loraOriginalWeights != nil }

    /// Whether models are loaded (Gemma may be nil after unloading post-encoding)
    public var isLoaded: Bool {
        textEncoder != nil && (transformer != nil || ltx2Transformer != nil) && vaeDecoder != nil
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
        self.scheduler = LTXScheduler(isDistilled: true)
    }


    // MARK: - Model Loading

    /// Load all models required for generation
    ///
    /// Downloads and loads:
    /// 1. Gemma 3 12B (text encoder backbone) — from VLM Gemma 4-bit QAT (shared across variants)
    /// 2. LTX-2.3 unified weights (transformer + VAE + connector) — from `Lightricks/LTX-2.3`
    ///
    /// The unified file is split at load time into transformer, VAE, and connector components.
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
        progressCallback?(DownloadProgress(progress: 0.3, message: "Loading LTX-2.3 weights..."))

        let transformerWeights: [String: MLXArray]
        let vaeWeights: [String: MLXArray]
        let connectorWeights: [String: MLXArray]

        stepStart = Date()
        let unifiedPath: String
        if let path = ltxWeightsPath {
            unifiedPath = path
        } else {
            // Download unified file from Lightricks/LTX-2.3
            LTXDebug.log("Downloading unified weights for \(model.displayName) (if needed)...")
            progressCallback?(DownloadProgress(progress: 0.35, message: "Downloading unified weights..."))
            let downloadedPath = try await downloader.downloadUnifiedWeights(model: model) { progress in
                progressCallback?(progress)
            }
            unifiedPath = downloadedPath.path
        }

        LTXDebug.log("Splitting unified weights from \(unifiedPath)...")
        let split = try LTXWeightLoader.splitUnifiedWeightsFile(path: unifiedPath)
        transformerWeights = split.transformer
        vaeWeights = split.vae
        connectorWeights = split.connector
        LTXDebug.log("[TIME] Download + split unified weights: \(String(format: "%.1f", Date().timeIntervalSince(stepStart)))s")

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
        if let mixedConfig = quantization.mixedPrecision {
            stepStart = Date()
            LTXDebug.log("Applying mixed precision: \(mixedConfig.highPrecisionBlocks.count) blocks at \(mixedConfig.highPrecisionBits)-bit, rest at \(mixedConfig.lowPrecisionBits)-bit...")
            progressCallback?(DownloadProgress(progress: 0.6, message: "Applying mixed precision quantization..."))
            applyMixedPrecisionQuantization(to: transformer!, config: mixedConfig)
            eval(transformer!.parameters())
            Memory.clearCache()
            LTXDebug.log("[TIME] Mixed precision quantization: \(String(format: "%.1f", Date().timeIntervalSince(stepStart)))s")
        } else if quantization.transformer.needsQuantization {
            stepStart = Date()
            let bits = quantization.transformer.bits
            let groupSize = quantization.transformer.groupSize
            let mode = quantization.transformer.quantizationMode
            LTXDebug.log("Quantizing transformer to \(quantization.transformer.rawValue) (groupSize=\(groupSize), mode=\(mode))...")
            progressCallback?(DownloadProgress(progress: 0.6, message: "Quantizing transformer to \(quantization.transformer.displayName)..."))
            quantize(model: transformer!, groupSize: groupSize, bits: bits, mode: mode)
            eval(transformer!.parameters())
            Memory.clearCache()
            LTXDebug.log("[TIME] Transformer quantization: \(String(format: "%.1f", Date().timeIntervalSince(stepStart)))s")
        }

        // Step 4: Create and load VAE decoder
        progressCallback?(DownloadProgress(progress: 0.7, message: "Loading VAE decoder..."))

        vaeDecoder = VideoDecoder()
        // LTX-2.3 unified file doesn't include standalone vae/config.json;
        // timestep_conditioning defaults to false (matching LTX-2.3 behavior)

        stepStart = Date()
        LTXDebug.log("Applying \(vaeWeights.count) VAE weights...")
        try LTXWeightLoader.applyVAEWeights(vaeWeights, to: vaeDecoder!)
        LTXDebug.log("[TIME] VAE load: \(String(format: "%.1f", Date().timeIntervalSince(stepStart)))s")

        // Step 5: Create and load text encoder (connector)
        progressCallback?(DownloadProgress(progress: 0.9, message: "Loading text encoder..."))

        textEncoder = createTextEncoder(
            gatedAttention: model.transformerConfig.gatedAttention
        )

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

        // Step 2: Download unified file and extract connector weights
        progressCallback?(DownloadProgress(progress: 0.7, message: "Loading connector weights..."))
        stepStart = Date()
        let unifiedPath = try await downloader.downloadUnifiedWeights(model: model) { progress in
            progressCallback?(progress)
        }
        let split = try LTXWeightLoader.splitUnifiedWeightsFile(path: unifiedPath.path)
        let connectorWeights = split.connector

        textEncoder = createTextEncoder(
            gatedAttention: model.transformerConfig.gatedAttention
        )
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
        includeEncoder: Bool = false,
        progressCallback: DownloadProgressCallback? = nil
    ) async throws {
        LTXDebug.log("Loading audio models...")

        // Step 1: Download and load Audio VAE
        progressCallback?(DownloadProgress(progress: 0.1, message: "Downloading audio VAE..."))
        let audioVAEPath = try await downloader.downloadAudioVAE { progress in
            progressCallback?(progress)
        }
        let audioVAEWeights = try LTXWeightLoader.loadAudioVAEWeights(from: audioVAEPath.path, includeEncoder: includeEncoder)

        audioVAE = AudioVAE(includeEncoder: includeEncoder)
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

        // Load and split unified weights (includeAudio: true to keep audio transformer keys)
        let (transformerWeights, _, connectorWeightsFromUnified) = try LTXWeightLoader.splitUnifiedWeightsFile(
            path: unifiedPath.path,
            includeAudio: true
        )

        // Create LTX2 dual transformer
        let ltx2 = LTX2Transformer(
            config: model.transformerConfig,
            ropeType: .split,
            memoryOptimization: memoryOptimization
        )

        // Apply weights with audio key mapping enabled
        try LTXWeightLoader.applyTransformerWeights(transformerWeights, to: ltx2, includeAudio: true)

        // Apply quantization if configured
        if let mixedConfig = quantization.mixedPrecision {
            LTXDebug.log("Applying mixed precision to LTX2 transformer...")
            applyMixedPrecisionQuantization(to: ltx2, config: mixedConfig)
            eval(ltx2.parameters())
            Memory.clearCache()
        } else if quantization.transformer.needsQuantization {
            let q = quantization.transformer
            LTXDebug.log("Quantizing LTX2 transformer to \(q.displayName)...")
            quantize(model: ltx2, groupSize: q.groupSize, bits: q.bits, mode: q.quantizationMode)
            eval(ltx2.parameters())
            Memory.clearCache()
        }

        ltx2Transformer = ltx2

        transformer = nil
        Memory.clearCache()

        // Step 4: Recreate text encoder with audio connector and reload all connector weights
        progressCallback?(DownloadProgress(progress: 0.9, message: "Loading audio text connector..."))

        // Create new text encoder with audio connector enabled
        let newTextEncoder = createTextEncoder(
            includeAudioConnector: true,
            gatedAttention: model.transformerConfig.gatedAttention
        )

        // Use connector weights extracted from the unified file (LTX-2.3 has no standalone connector file)
        let connectorWeights = connectorWeightsFromUnified
        LTXDebug.log("Loaded \(connectorWeights.count) connector weights from unified file (video + audio + feature extractor)")

        // Apply all connector weights (video + audio + feature extractor)
        try LTXWeightLoader.applyTextEncoderWeights(connectorWeights, to: newTextEncoder)
        textEncoder = newTextEncoder
        LTXDebug.log("Text encoder updated with audio connector")

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

    /// Generate video using two-stage pipeline (half-res → upscale → refine).
    ///
    /// Supports both video-only and dual video/audio modes:
    /// - **Video-only** (default): Uses `LTXTransformer` for denoising
    /// - **With audio** (after `loadAudioModels()`): Uses `LTX2Transformer` for dual video/audio denoising
    ///
    /// Always two-stage distilled: 8 steps at half resolution → 2x spatial upscale → 3 refinement steps.
    /// I2V supported via `config.imagePath`.
    ///
    /// - Parameters:
    ///   - prompt: Text description of the video
    ///   - config: Video generation configuration (width/height = FINAL resolution)
    ///   - upscalerWeightsPath: Path to spatial upscaler safetensors
    ///   - onProgress: Optional progress callback
    /// - Returns: VideoGenerationResult with video frames and optional audio
    public func generateVideo(
        prompt: String,
        config: LTXVideoGenerationConfig,
        upscalerWeightsPath: String,
        onProgress: GenerationProgressCallback? = nil,
    ) async throws -> VideoGenerationResult {
        try config.validate()

        let hasAudio = isAudioLoaded

        guard let textEncoder = textEncoder,
              let vaeDecoder = vaeDecoder
        else {
            throw LTXError.modelNotLoaded("Models not loaded. Call loadModels() first.")
        }

        // Need either video-only OR dual transformer
        guard transformer != nil || ltx2Transformer != nil else {
            throw LTXError.modelNotLoaded("No transformer loaded. Call loadModels() first.")
        }

        let generationStart = Date()

        // Two-stage requires width/height divisible by 64
        guard config.width % 64 == 0 && config.height % 64 == 0 else {
            throw LTXError.invalidConfiguration("Two-stage requires width and height divisible by 64. Got \(config.width)x\(config.height)")
        }

        let halfWidth = config.width / 2
        let halfHeight = config.height / 2
        let isI2V = config.imagePath != nil

        LTXDebug.log("Two-stage generation: \(halfWidth)x\(halfHeight) → \(config.width)x\(config.height), audio=\(hasAudio)")

        // 0. Encode image at half-res if I2V
        var halfResImageLatent: MLXArray? = nil
        if let imagePath = config.imagePath {
            LTXDebug.log("Two-stage I2V: encoding image at \(halfWidth)x\(halfHeight)")
            halfResImageLatent = try await encodeImage(path: imagePath, width: halfWidth, height: halfHeight)
            unloadVAEEncoder()
        }

        // 0b. Optionally enhance prompt
        let effectivePrompt: String
        if config.enhancePrompt {
            LTXDebug.log("Enhancing prompt with VLM (\(config.imagePath != nil ? "I2V" : "T2V"))...")
            effectivePrompt = try await enhancePromptWithVLM(prompt, imagePath: config.imagePath)
        } else {
            effectivePrompt = prompt
        }

        // 1. Text encoding
        let profiler = LTXVideoProfiler.shared
        profiler.start("Text Encoding")

        profiler.start("Tokenization")
        let (inputIds, attentionMask) = try tokenizePrompt(effectivePrompt, maxLength: textMaxLength)
        MLX.eval(inputIds, attentionMask)
        profiler.end("Tokenization")

        guard let gemma = gemmaModel else {
            throw LTXError.modelNotLoaded("Gemma model not loaded")
        }

        profiler.start("Gemma Forward")
        let (_, allHiddenStates) = gemma(inputIds, attentionMask: attentionMask, outputHiddenStates: true)
        guard let states = allHiddenStates, states.count == gemma.config.hiddenLayers + 1 else {
            throw LTXError.generationFailed("Failed to extract Gemma hidden states")
        }
        MLX.eval(states[states.count - 1])
        profiler.end("Gemma Forward")

        profiler.start("Feature Extractor + Connector")
        let encoderOutput = try textEncoder.encodeFromHiddenStates(
            hiddenStates: states,
            attentionMask: attentionMask,
            paddingSide: "left"
        )
        let videoTextEmbeddings = encoderOutput.videoEncoding
        let audioTextEmbeddings = encoderOutput.audioEncoding ?? videoTextEmbeddings
        let textMask = encoderOutput.attentionMask
        MLX.eval(videoTextEmbeddings, audioTextEmbeddings, textMask)
        profiler.end("Feature Extractor + Connector")

        LTXDebug.log("Video text: \(videoTextEmbeddings.shape), Audio text: \(audioTextEmbeddings.shape)")
        profiler.end("Text Encoding")

        // Unload Gemma
        self.gemmaModel = nil
        self.tokenizer = nil
        Memory.clearCache()

        // 2. Create latent shapes
        let stage1Shape = VideoLatentShape.fromPixelDimensions(
            batch: 1, channels: 128,
            frames: config.numFrames, height: halfHeight, width: halfWidth
        )

        let audioNumFrames = hasAudio ? computeAudioLatentFrames(videoFrames: config.numFrames) : 0
        LTXDebug.log("Stage 1 latent: \(stage1Shape.frames)x\(stage1Shape.height)x\(stage1Shape.width)\(hasAudio ? ", audio frames: \(audioNumFrames)" : "")")

        // 3. Generate noise
        if let seed = config.seed {
            MLXRandom.seed(seed)
        }

        // Video noise (float32) at half resolution
        var videoLatent = generateNoise(shape: stage1Shape, seed: config.seed)
        MLX.eval(videoLatent)

        // Audio noise (float32, drawn after video noise from same RNG)
        var audioLatentPacked: MLXArray? = nil
        if hasAudio {
            let audioLatent = MLXRandom.normal(
                [1, Self.audioLatentChannels, audioNumFrames, Self.audioLatentMelBins]
            ).asType(.float32)
            MLX.eval(audioLatent)
            audioLatentPacked = packAudioLatents(audioLatent)
        }

        // 4. Stage 1 sigma schedule (always distilled: 8 steps, no CFG/STG)
        let stage1Steps = 8
        let stage1Scheduler = LTXScheduler(isDistilled: true)
        stage1Scheduler.setTimesteps(
            numSteps: stage1Steps,
            distilled: true,
            latentTokenCount: stage1Shape.tokenCount
        )
        let stage1Sigmas = stage1Scheduler.sigmas
        LTXDebug.log("Stage 1: \(stage1Sigmas.count - 1) distilled steps, sigmas: \(stage1Sigmas)")

        // Scale initial noise
        videoLatent = videoLatent * stage1Sigmas[0]
        if hasAudio, let ap = audioLatentPacked {
            audioLatentPacked = ap * stage1Sigmas[0]
        }

        // I2V: condition frame 0 with per-token timestep masking
        var stage1CondMask: MLXArray? = nil
        if let imgLatent = halfResImageLatent {
            videoLatent[0..., 0..., 0..<1, 0..., 0...] = imgLatent

            let tokensPerFrame = stage1Shape.height * stage1Shape.width
            let frame0Mask = MLXArray.ones([1, tokensPerFrame])
            let otherMask = MLXArray.zeros([1, stage1Shape.tokenCount - tokensPerFrame])
            stage1CondMask = MLX.concatenated([frame0Mask, otherMask], axis: 1)
            MLX.eval(stage1CondMask!)
        }
        MLX.eval(videoLatent)

        // === STAGE 1: Denoise at half resolution ===
        let stage1NumSteps = stage1Sigmas.count - 1
        let stage2NumStepsForProgress = STAGE_2_DISTILLED_SIGMA_VALUES.count - 1
        let totalStepsForProgress = stage1NumSteps + stage2NumStepsForProgress
        LTXDebug.log("=== Stage 1: Half-resolution \(hasAudio ? "dual" : "video-only") denoising (\(stage1NumSteps) steps) ===")
        profiler.start("Denoising Stage 1")
        profiler.setTotalSteps(stage1NumSteps + stage2NumStepsForProgress)
        let stage1Start = Date()

        for step in 0..<stage1NumSteps {
            let stepStart = Date()
            let sigma = stage1Sigmas[step]
            let sigmaNext = stage1Sigmas[step + 1]

            onProgress?(GenerationProgress(
                currentStep: step, totalSteps: totalStepsForProgress, sigma: sigma, phase: .denoising
            ))

            // I2V: inject noise to conditioned frame
            if let condLatent = halfResImageLatent, config.imageCondNoiseScale > 0, sigma > 0 {
                let injectionNoise = MLXRandom.normal(condLatent.shape)
                let noisedFrame0 = condLatent + MLXArray(config.imageCondNoiseScale) * injectionNoise * MLXArray(sigma * sigma)
                videoLatent[0..., 0..., 0..<1, 0..., 0...] = noisedFrame0
            }

            // Per-token timestep for I2V: frame 0 gets σ=0, others get σ
            let videoTimestep: MLXArray
            if let mask = stage1CondMask {
                videoTimestep = MLXArray(sigma) * (1 - mask)
            } else {
                videoTimestep = MLXArray([sigma])
            }

            let videoPatchified = patchify(videoLatent).asType(.bfloat16)

            if hasAudio, let ltx2 = ltx2Transformer, let ap = audioLatentPacked {
                // Dual video/audio denoising
                let audioTimestep = MLXArray([sigma])
                let audioPatchified = ap.asType(.bfloat16)

                let (videoVelPred, audioVelPred) = ltx2(
                    videoLatent: videoPatchified,
                    audioLatent: audioPatchified,
                    videoContext: videoTextEmbeddings.asType(.bfloat16),
                    audioContext: audioTextEmbeddings.asType(.bfloat16),
                    videoTimesteps: videoTimestep,
                    audioTimesteps: audioTimestep,
                    videoContextMask: textMask,
                    audioContextMask: nil,
                    videoLatentShape: (frames: stage1Shape.frames, height: stage1Shape.height, width: stage1Shape.width),
                    audioNumFrames: audioNumFrames
                )

                let videoVelocity = unpatchify(videoVelPred, shape: stage1Shape).asType(.float32)
                let audioVelocity = audioVelPred.asType(.float32)

                if halfResImageLatent != nil {
                    let velocitySlice = videoVelocity[0..., 0..., 1..., 0..., 0...]
                    let latentSlice = videoLatent[0..., 0..., 1..., 0..., 0...]
                    let steppedSlice = stage1Scheduler.step(
                        latent: latentSlice, velocity: velocitySlice,
                        sigma: sigma, sigmaNext: sigmaNext
                    )
                    let frame0 = videoLatent[0..., 0..., 0..<1, 0..., 0...]
                    videoLatent = MLX.concatenated([frame0, steppedSlice], axis: 2)
                } else {
                    videoLatent = stage1Scheduler.step(
                        latent: videoLatent, velocity: videoVelocity,
                        sigma: sigma, sigmaNext: sigmaNext
                    )
                }
                let updatedAudio = ap + (sigmaNext - sigma) * audioVelocity
                audioLatentPacked = updatedAudio
                MLX.eval(videoLatent, updatedAudio)
            } else if let videoTransformer = transformer {
                // Video-only denoising
                let velocityPred = videoTransformer(
                    latent: videoPatchified,
                    context: videoTextEmbeddings.asType(.bfloat16),
                    timesteps: videoTimestep,
                    contextMask: nil,
                    latentShape: (frames: stage1Shape.frames, height: stage1Shape.height, width: stage1Shape.width)
                )

                let videoVelocity = unpatchify(velocityPred, shape: stage1Shape).asType(.float32)

                if halfResImageLatent != nil {
                    let velocitySlice = videoVelocity[0..., 0..., 1..., 0..., 0...]
                    let latentSlice = videoLatent[0..., 0..., 1..., 0..., 0...]
                    let steppedSlice = stage1Scheduler.step(
                        latent: latentSlice, velocity: velocitySlice,
                        sigma: sigma, sigmaNext: sigmaNext
                    )
                    let frame0 = videoLatent[0..., 0..., 0..<1, 0..., 0...]
                    videoLatent = MLX.concatenated([frame0, steppedSlice], axis: 2)
                } else {
                    videoLatent = stage1Scheduler.step(
                        latent: videoLatent, velocity: videoVelocity,
                        sigma: sigma, sigmaNext: sigmaNext
                    )
                }
                MLX.eval(videoLatent)
            }

            if (step + 1) % 5 == 0 { Memory.clearCache() }
            let stepDur = Date().timeIntervalSince(stepStart)
            profiler.recordStep(duration: stepDur)

            LTXDebug.log("Stage 1 step \(step)/\(stage1NumSteps): σ=\(String(format: "%.4f", sigma))→\(String(format: "%.4f", sigmaNext)), time=\(String(format: "%.1f", stepDur))s")
        }
        LTXDebug.log("Stage 1 complete: \(String(format: "%.1f", Date().timeIntervalSince(stage1Start)))s")

        // Dump stage 1 output for comparison
        if LTXDebug.isEnabled {
            let dumpDir = "/tmp/debug_dumps/swift"
            try? FileManager.default.createDirectory(atPath: dumpDir, withIntermediateDirectories: true)
            let s1 = videoLatent.asType(.float32)
            MLX.eval(s1)
            try? MLX.save(arrays: ["data": s1], url: URL(fileURLWithPath: "\(dumpDir)/stage1_output.safetensors"))
            LTXDebug.log("Dumped stage1_output: \(s1.shape)")
        }

        profiler.end("Denoising Stage 1")

        // === UPSCALE VIDEO 2x (audio unchanged) ===
        onProgress?(GenerationProgress(
            currentStep: totalStepsForProgress, totalSteps: totalStepsForProgress, sigma: 0, phase: .upscaling
        ))
        LTXDebug.log("=== Upscaling video latent 2x ===")
        profiler.start("Upscaler 2x")
        let upscaleStart = Date()

        let upscaler = try loadSpatialUpscaler(from: upscalerWeightsPath)

        let latentMean = vaeDecoder.meanOfMeans
        let latentStd = vaeDecoder.stdOfMeans
        MLX.eval(latentMean, latentStd)

        let mean5d = latentMean.reshaped([1, -1, 1, 1, 1])
        let std5d = latentStd.reshaped([1, -1, 1, 1, 1])

        let denormedLatent = videoLatent * std5d + mean5d
        MLX.eval(denormedLatent)

        let upscaledLatent = upscaler(denormedLatent)
        MLX.eval(upscaledLatent)

        videoLatent = (upscaledLatent - mean5d) / std5d
        MLX.eval(videoLatent)

        LTXDebug.log("Upscale time: \(String(format: "%.1f", Date().timeIntervalSince(upscaleStart)))s, shape: \(videoLatent.shape)")
        profiler.end("Upscaler 2x")

        // Dump upscaled latent
        if LTXDebug.isEnabled {
            let dumpDir = "/tmp/debug_dumps/swift"
            let up = videoLatent.asType(.float32)
            MLX.eval(up)
            try? MLX.save(arrays: ["data": up], url: URL(fileURLWithPath: "\(dumpDir)/after_upscale.safetensors"))
            LTXDebug.log("Dumped after_upscale: \(up.shape)")
        }

        // === STAGE 2: Refine at full resolution ===
        LTXDebug.log("=== Stage 2: Full-resolution \(hasAudio ? "dual" : "video-only") refinement (3 steps) ===")
        profiler.start("Denoising Stage 2")
        let stage2Start = Date()

        let stage2Shape = VideoLatentShape.fromPixelDimensions(
            batch: 1, channels: 128,
            frames: config.numFrames, height: config.height, width: config.width
        )

        // Re-noise video and audio for refinement
        let stage2Sigmas = STAGE_2_DISTILLED_SIGMA_VALUES
        let noiseScale = stage2Sigmas[0]  // 0.909375

        let videoNoise = generateNoise(shape: stage2Shape)
        videoLatent = MLXArray(noiseScale) * videoNoise + MLXArray(1.0 - noiseScale) * videoLatent

        if hasAudio, let ap = audioLatentPacked {
            let audioReNoise = MLXRandom.normal(ap.shape).asType(.float32)
            audioLatentPacked = MLXArray(noiseScale) * audioReNoise + MLXArray(1.0 - noiseScale) * ap
        }

        // I2V stage 2: encode image at full resolution and condition frame 0
        var fullResImageLatent: MLXArray? = nil
        var stage2CondMask: MLXArray? = nil
        if isI2V, let imagePath = config.imagePath {
            LTXDebug.log("Stage 2 I2V: encoding image at \(config.width)x\(config.height)")
            fullResImageLatent = try await encodeImage(path: imagePath, width: config.width, height: config.height)
            unloadVAEEncoder()
            videoLatent[0..., 0..., 0..<1, 0..., 0...] = fullResImageLatent!

            let tokensPerFrame = stage2Shape.height * stage2Shape.width
            let frame0Mask = MLXArray.ones([1, tokensPerFrame])
            let otherMask = MLXArray.zeros([1, stage2Shape.tokenCount - tokensPerFrame])
            stage2CondMask = MLX.concatenated([frame0Mask, otherMask], axis: 1)
            MLX.eval(stage2CondMask!)
        }
        MLX.eval(videoLatent)
        if hasAudio, let ap = audioLatentPacked { MLX.eval(ap) }

        // Dump re-noised latent
        if LTXDebug.isEnabled {
            let dumpDir = "/tmp/debug_dumps/swift"
            let rn = videoLatent.asType(.float32)
            MLX.eval(rn)
            try? MLX.save(arrays: ["data": rn], url: URL(fileURLWithPath: "\(dumpDir)/after_renoise.safetensors"))
            LTXDebug.log("Dumped after_renoise: \(rn.shape)")
        }

        let stage2NumSteps = stage2Sigmas.count - 1
        for step in 0..<stage2NumSteps {
            let stepStart = Date()
            let sigma = stage2Sigmas[step]
            let sigmaNext = stage2Sigmas[step + 1]

            onProgress?(GenerationProgress(
                currentStep: stage1NumSteps + step, totalSteps: totalStepsForProgress, sigma: sigma, phase: .refinement
            ))

            // I2V: inject noise to conditioned frame
            if let condLatent = fullResImageLatent, config.imageCondNoiseScale > 0, sigma > 0 {
                let injectionNoise = MLXRandom.normal(condLatent.shape)
                let noisedFrame0 = condLatent + MLXArray(config.imageCondNoiseScale) * injectionNoise * MLXArray(sigma * sigma)
                videoLatent[0..., 0..., 0..<1, 0..., 0...] = noisedFrame0
            }

            // Per-token timestep for I2V: frame 0 gets σ=0, others get σ
            let videoTimestep: MLXArray
            if let mask = stage2CondMask {
                videoTimestep = MLXArray(sigma) * (1 - mask)
            } else {
                videoTimestep = MLXArray([sigma])
            }

            let videoPatchified = patchify(videoLatent).asType(.bfloat16)

            if hasAudio, let ltx2 = ltx2Transformer, let ap = audioLatentPacked {
                let audioTimestep = MLXArray([sigma])
                let audioPatchified = ap.asType(.bfloat16)

                let (videoVelPred, audioVelPred) = ltx2(
                    videoLatent: videoPatchified,
                    audioLatent: audioPatchified,
                    videoContext: videoTextEmbeddings.asType(.bfloat16),
                    audioContext: audioTextEmbeddings.asType(.bfloat16),
                    videoTimesteps: videoTimestep,
                    audioTimesteps: audioTimestep,
                    videoContextMask: textMask,
                    audioContextMask: nil,
                    videoLatentShape: (frames: stage2Shape.frames, height: stage2Shape.height, width: stage2Shape.width),
                    audioNumFrames: audioNumFrames
                )

                let videoVelocity = unpatchify(videoVelPred, shape: stage2Shape).asType(.float32)
                let audioVelocity = audioVelPred.asType(.float32)

                // Euler step
                let dt = sigmaNext - sigma
                if fullResImageLatent != nil {
                    let velocitySlice = videoVelocity[0..., 0..., 1..., 0..., 0...]
                    let latentSlice = videoLatent[0..., 0..., 1..., 0..., 0...]
                    let steppedSlice = latentSlice + MLXArray(dt) * velocitySlice
                    let frame0 = videoLatent[0..., 0..., 0..<1, 0..., 0...]
                    videoLatent = MLX.concatenated([frame0, steppedSlice], axis: 2)
                } else {
                    videoLatent = videoLatent + MLXArray(dt) * videoVelocity
                }
                let updatedAudio = ap + MLXArray(dt) * audioVelocity
                audioLatentPacked = updatedAudio
                MLX.eval(videoLatent, updatedAudio)
            } else if let videoTransformer = transformer {
                let velocityPred = videoTransformer(
                    latent: videoPatchified,
                    context: videoTextEmbeddings.asType(.bfloat16),
                    timesteps: videoTimestep,
                    contextMask: nil,
                    latentShape: (frames: stage2Shape.frames, height: stage2Shape.height, width: stage2Shape.width)
                )

                let videoVelocity = unpatchify(velocityPred, shape: stage2Shape).asType(.float32)

                let dt = sigmaNext - sigma
                if fullResImageLatent != nil {
                    let velocitySlice = videoVelocity[0..., 0..., 1..., 0..., 0...]
                    let latentSlice = videoLatent[0..., 0..., 1..., 0..., 0...]
                    let steppedSlice = latentSlice + MLXArray(dt) * velocitySlice
                    let frame0 = videoLatent[0..., 0..., 0..<1, 0..., 0...]
                    videoLatent = MLX.concatenated([frame0, steppedSlice], axis: 2)
                } else {
                    videoLatent = videoLatent + MLXArray(dt) * videoVelocity
                }
                MLX.eval(videoLatent)
            }

            let stepDur2 = Date().timeIntervalSince(stepStart)
            profiler.recordStep(duration: stepDur2)

            LTXDebug.log("Stage 2 step \(step)/\(stage2NumSteps): σ=\(String(format: "%.4f", sigma))→\(String(format: "%.4f", sigmaNext)), time=\(String(format: "%.1f", stepDur2))s")
        }
        LTXDebug.log("Stage 2 complete: \(String(format: "%.1f", Date().timeIntervalSince(stage2Start)))s")
        profiler.end("Denoising Stage 2")

        // Unload transformer
        if memoryOptimization.unloadAfterUse {
            self.ltx2Transformer = nil
            self.transformer = nil
            Memory.clearCache()
            LTXDebug.log("Transformer unloaded")
        }

        // Dump latent for Python comparison (debug)
        if LTXDebug.isEnabled {
            let dumpDir = "/tmp/debug_dumps/swift"
            try? FileManager.default.createDirectory(atPath: dumpDir, withIntermediateDirectories: true)
            let latentToSave = videoLatent.asType(.float32)
            MLX.eval(latentToSave)
            try? MLX.save(arrays: ["data": latentToSave], url: URL(fileURLWithPath: "\(dumpDir)/final_latent.safetensors"))
            LTXDebug.log("Saved final latent to \(dumpDir)/final_latent.safetensors: \(latentToSave.shape)")
        }

        // Decode video
        onProgress?(GenerationProgress(
            currentStep: totalStepsForProgress, totalSteps: totalStepsForProgress, sigma: 0, phase: .decoding
        ))
        LTXMemoryManager.setPhase(.vaeDecode)
        profiler.start("VAE Decode")

        profiler.start("VAE Forward Pass")
        let videoTensor = decodeVideo(
            latent: videoLatent, decoder: vaeDecoder, timestep: nil,
            temporalTileSize: memoryOptimization.vaeTemporalTileSize,
            temporalTileOverlap: memoryOptimization.vaeTemporalTileOverlap
        )
        MLX.eval(videoTensor)
        profiler.end("VAE Forward Pass")

        profiler.end("VAE Decode")

        let trimmedVideo: MLXArray
        if videoTensor.dim(0) > config.numFrames {
            trimmedVideo = videoTensor[0..<config.numFrames]
        } else {
            trimmedVideo = videoTensor
        }

        // Decode audio if present
        var audioWaveform: MLXArray? = nil
        var audioSampleRate: Int? = nil
        if hasAudio, let ap = audioLatentPacked, let audioVAE = audioVAE, let vocoder = vocoder {
            LTXDebug.log("Decoding audio latents...")
            let audioLatentUnpacked = unpackAudioLatents(ap, numFrames: audioNumFrames)
            let waveform = decodeAudio(
                latents: audioLatentUnpacked,
                audioVAE: audioVAE,
                vocoder: vocoder
            )
            MLX.eval(waveform)
            audioWaveform = waveform
            audioSampleRate = vocoder.outputSampleRate
            LTXDebug.log("Audio waveform: \(waveform.shape)")
        }

        // Signal export phase
        onProgress?(GenerationProgress(
            currentStep: totalStepsForProgress, totalSteps: totalStepsForProgress, sigma: 0, phase: .exporting
        ))

        LTXMemoryManager.resetCacheLimit()

        let generationTime = Date().timeIntervalSince(generationStart)
        LTXDebug.log("Total two-stage generation time: \(String(format: "%.1f", generationTime))s")

        return VideoGenerationResult(
            frames: trimmedVideo,
            seed: config.seed ?? 0,
            generationTime: generationTime,
            
            audioWaveform: audioWaveform,
            audioSampleRate: audioSampleRate,
            effectivePrompt: effectivePrompt
        )
    }

    // MARK: - Retake (Video-to-Video)

    /// Generate a retake from an existing video (single-stage, matching Lightricks reference).
    ///
    /// Encodes the source video at native resolution, selectively noises the temporal
    /// region to regenerate, then denoises with a new prompt using the full sigma schedule.
    /// Frames outside the retake window are preserved via `post_process_latent` at each step.
    ///
    /// - Parameters:
    ///   - prompt: New text description for the retaken video
    ///   - config: Generation config with `videoPath` set. `retakeStrength` is unused
    ///     (matching Lightricks: regenerated frames always start from pure noise).
    ///   - upscalerWeightsPath: Unused (kept for API compatibility)
    ///   - onProgress: Optional progress callback
    /// - Returns: VideoGenerationResult with retaken video frames
    public func generateRetake(
        prompt: String,
        config: LTXVideoGenerationConfig,
        upscalerWeightsPath: String,
        onProgress: GenerationProgressCallback? = nil,
    ) async throws -> VideoGenerationResult {
        try config.validate()

        guard let videoPath = config.videoPath else {
            throw LTXError.invalidConfiguration("videoPath must be set for retake mode")
        }

        guard let textEncoder = textEncoder,
              let vaeDecoder = vaeDecoder
        else {
            throw LTXError.modelNotLoaded("Models not loaded. Call loadModels() first.")
        }

        guard transformer != nil || ltx2Transformer != nil else {
            throw LTXError.modelNotLoaded("No transformer loaded. Call loadModels() first.")
        }

        let generationStart = Date()

        LTXDebug.log("Retake (single-stage): \(config.width)x\(config.height)")

        // Phase 0: Encode source video at native resolution
        LTXDebug.log("Encoding source video at \(config.width)x\(config.height)...")
        let cleanLatent = try await encodeVideo(
            path: videoPath, width: config.width, height: config.height,
            numFrames: config.numFrames
        )
        unloadVAEEncoder()

        // Phase 0a: Extract source audio (for passthrough and optional cross-modal attention)
        let audioProcessor = AudioProcessor(sampleRate: Self.audioSampleRate)
        var sourceAudioWaveform: MLXArray? = nil
        var frozenAudioLatentPacked: MLXArray? = nil
        var retakeAudioNumFrames: Int = 1

        // Always extract audio from source video for passthrough to output
        let sourceWaveform = try await audioProcessor.loadAudio(from: videoPath)
        if sourceWaveform.dim(0) > 0 {
            sourceAudioWaveform = sourceWaveform
            LTXDebug.log("Source audio extracted: \(sourceWaveform.dim(0)) samples (\(String(format: "%.1f", Float(sourceWaveform.dim(0)) / Float(Self.audioSampleRate)))s)")
        }

        // Encode audio latents for cross-modal attention if LTX2Transformer + AudioVAE encoder are loaded
        let regenerateAudio = config.regenerateAudio
        if ltx2Transformer != nil, let audioVAE = audioVAE, audioVAE.encoder != nil, let waveform = sourceAudioWaveform {
            LTXDebug.log("Encoding source audio for \(regenerateAudio ? "regeneration" : "cross-modal attention")...")
            let melSpec = try audioProcessor.melSpectrogram(waveform)
            eval(melSpec)

            let audioLatent = try audioVAE.encode(melSpec)  // (1, 8, T_latent, 16)
            eval(audioLatent)

            retakeAudioNumFrames = audioLatent.dim(2)
            let packed = audioLatent.transposed(0, 2, 1, 3)  // (1, T, 8, 16)
            frozenAudioLatentPacked = packed.reshaped([1, retakeAudioNumFrames, Self.audioPackedChannels]).asType(DType.bfloat16)
            LTXDebug.log("Audio encoded: \(retakeAudioNumFrames) latent frames")
        }

        // Phase 0b: Optionally enhance prompt
        let effectivePrompt: String
        if config.enhancePrompt {
            LTXDebug.log("Enhancing prompt with VLM...")
            effectivePrompt = try await enhancePromptWithVLM(prompt, imagePath: nil)
        } else {
            effectivePrompt = prompt
        }

        // Phase 1: Text encoding
        let profiler = LTXVideoProfiler.shared
        profiler.start("Text Encoding")
        let (inputIds, attentionMask) = try tokenizePrompt(effectivePrompt, maxLength: textMaxLength)

        guard let gemma = gemmaModel else {
            throw LTXError.modelNotLoaded("Gemma model not loaded")
        }

        let (_, allHiddenStates) = gemma(inputIds, attentionMask: attentionMask, outputHiddenStates: true)
        guard let states = allHiddenStates, states.count == gemma.config.hiddenLayers + 1 else {
            throw LTXError.generationFailed("Failed to extract Gemma hidden states")
        }

        let encoderOutput = try textEncoder.encodeFromHiddenStates(
            hiddenStates: states,
            attentionMask: attentionMask,
            paddingSide: "left"
        )
        let videoTextEmbeddings = encoderOutput.videoEncoding
        let audioTextEmbeddings = encoderOutput.audioEncoding
        MLX.eval(videoTextEmbeddings)
        if let ae = audioTextEmbeddings { MLX.eval(ae) }

        LTXDebug.log("Text encoding: video=\(videoTextEmbeddings.shape), audio=\(audioTextEmbeddings?.shape.description ?? "nil")")
        profiler.end("Text Encoding")

        // Unload Gemma
        self.gemmaModel = nil
        self.tokenizer = nil
        Memory.clearCache()

        // Phase 2: Single-stage denoising at native resolution
        let latentShape = VideoLatentShape.fromPixelDimensions(
            batch: 1, channels: 128,
            frames: config.numFrames, height: config.height, width: config.width
        )

        LTXDebug.log("Retake latent: \(latentShape.frames)x\(latentShape.height)x\(latentShape.width)")

        // Build temporal masks for partial retake
        // condMask: per-token, 1=keep (σ=0), 0=regenerate (σ=sigma)
        // denoiseMask5d: per-frame 5D, 1=regenerate, 0=keep
        let isPartialRetake = config.retakeStartTime != nil || config.retakeEndTime != nil
        var condMask: MLXArray? = nil
        var denoiseMask5d: MLXArray? = nil

        if isPartialRetake {
            let fps: Float = 24.0
            let totalDuration = Float(config.numFrames) / fps
            let startTime = min(config.retakeStartTime ?? 0.0, totalDuration)
            let endTime = min(config.retakeEndTime ?? totalDuration, totalDuration)

            guard startTime < endTime else {
                throw LTXError.invalidConfiguration(
                    "Retake start time (\(startTime)s) must be before end time (\(endTime)s). " +
                    "Output duration is \(String(format: "%.1f", totalDuration))s (\(config.numFrames) frames at \(Int(fps))fps). " +
                    "Increase --frames to cover the full source video duration."
                )
            }

            let latentFrames = latentShape.frames
            let tokensPerFrame = latentShape.height * latentShape.width

            let startPixelFrame = Int(startTime * fps)
            let endPixelFrame = min(Int(endTime * fps), config.numFrames - 1)
            let startLatentFrame = max(0, min(startPixelFrame / 8, latentFrames - 1))
            let endLatentFrame = max(startLatentFrame, min(latentFrames - 1, (endPixelFrame + 7) / 8))

            LTXDebug.log("Partial retake: time \(startTime)s-\(endTime)s → latent frames \(startLatentFrame)-\(endLatentFrame) of \(latentFrames)")

            // Per-token mask: 1=keep (σ=0), 0=regenerate
            var maskValues = [Float](repeating: 1.0, count: latentShape.tokenCount)
            for f in startLatentFrame...endLatentFrame {
                let tokenOffset = f * tokensPerFrame
                for t in 0..<tokensPerFrame {
                    maskValues[tokenOffset + t] = 0.0
                }
            }
            condMask = MLXArray(maskValues, [1, latentShape.tokenCount])
            MLX.eval(condMask!)

            // 5D mask: 1=regenerate, 0=keep
            var mask5dValues = [Float](repeating: 0.0, count: latentFrames)
            for f in startLatentFrame...endLatentFrame {
                mask5dValues[f] = 1.0
            }
            denoiseMask5d = MLXArray(mask5dValues, [1, 1, latentFrames, 1, 1])
            MLX.eval(denoiseMask5d!)

            let regenFrames = endLatentFrame - startLatentFrame + 1
            LTXDebug.log("Regenerating \(regenFrames)/\(latentFrames) latent frames")
        }

        // Determine model mode: dev (30 steps + CFG + STG) or distilled (8 steps, no guidance)
        let useDevModel = (model == .dev)
        let retakeSteps = useDevModel ? 30 : 8
        let cfgScale: Float = useDevModel ? 3.0 : 1.0
        let stgScale: Float = useDevModel ? 1.0 : 0.0
        let stgBlocks: [Int] = useDevModel ? [28] : []  // LTX-2.3 default
        let guidanceRescale: Float = useDevModel ? 0.7 : 0.0

        let scheduler = LTXScheduler(isDistilled: !useDevModel)
        scheduler.setTimesteps(
            numSteps: retakeSteps,
            distilled: !useDevModel,
            latentTokenCount: latentShape.tokenCount
        )
        let sigmas = scheduler.sigmas
        let numSteps = sigmas.count - 1

        // Encode negative prompt for CFG (dev model only)
        var negVideoTextEmbeddings: MLXArray? = nil
        var negAudioTextEmbeddings: MLXArray? = nil
        if useDevModel {
            // Re-load Gemma for negative prompt encoding (was unloaded above)
            // Use empty string as negative prompt (matching Lightricks default for MLX)
            try await loadModels(progressCallback: nil)
            guard let gemma2 = gemmaModel else {
                throw LTXError.modelNotLoaded("Failed to reload Gemma for negative prompt")
            }
            let negPrompt = ""
            let (negInputIds, negAttentionMask) = try tokenizePrompt(negPrompt, maxLength: textMaxLength)
            let (_, negAllHidden) = gemma2(negInputIds, attentionMask: negAttentionMask, outputHiddenStates: true)
            guard let negStates = negAllHidden, negStates.count == gemma2.config.hiddenLayers + 1 else {
                throw LTXError.generationFailed("Failed to extract Gemma hidden states for negative prompt")
            }
            let negEncoderOutput = try textEncoder.encodeFromHiddenStates(
                hiddenStates: negStates,
                attentionMask: negAttentionMask,
                paddingSide: "left"
            )
            negVideoTextEmbeddings = negEncoderOutput.videoEncoding
            negAudioTextEmbeddings = negEncoderOutput.audioEncoding
            MLX.eval(negVideoTextEmbeddings!)
            if let nae = negAudioTextEmbeddings { MLX.eval(nae) }

            // Unload Gemma again
            self.gemmaModel = nil
            self.tokenizer = nil
            Memory.clearCache()
        }

        LTXDebug.log("Retake: \(numSteps) steps, cfg=\(cfgScale), rescale=\(guidanceRescale), sigmas: \(sigmas)")

        // Noise injection: pure noise where denoise_mask=1, clean elsewhere
        // (matching Lightricks GaussianNoiser with noise_scale=1.0)
        if let seed = config.seed {
            MLXRandom.seed(seed)
        }
        let noise = generateNoise(shape: latentShape, seed: config.seed)

        var videoLatent: MLXArray
        if let mask = denoiseMask5d {
            // Partial: pure noise on regen frames, clean on kept frames
            videoLatent = mask * noise + (1 - mask) * cleanLatent
        } else {
            // Full retake: pure noise
            videoLatent = noise
        }
        MLX.eval(videoLatent)

        // Audio regeneration: noise the audio latents (same as video)
        var audioLatentPacked: MLXArray? = nil
        if regenerateAudio, let cleanAudio = frozenAudioLatentPacked {
            // Noise in float32 (matching generateVideo pattern — "Float32 latents" rule)
            let audioNoise = MLXRandom.normal(cleanAudio.shape).asType(.float32)
            audioLatentPacked = audioNoise  // Start from pure noise (full regeneration)
            frozenAudioLatentPacked = nil   // Don't freeze — will be denoised
            LTXDebug.log("Audio latents noised for regeneration: \(audioNoise.shape)")
        }

        // Denoising loop (matching Lightricks euler_denoising_loop)
        let modeStr = isPartialRetake ? "partial" : "full"
        LTXDebug.log("=== Single-stage \(modeStr) retake denoising (\(numSteps) steps) ===")
        profiler.start("Denoising")
        profiler.setTotalSteps(numSteps)
        let denoiseStart = Date()

        for step in 0..<numSteps {
            let stepStart = Date()
            let sigma = sigmas[step]
            let sigmaNext = sigmas[step + 1]

            onProgress?(GenerationProgress(
                currentStep: step, totalSteps: numSteps, sigma: sigma, phase: .denoising
            ))

            // Per-token timestep: kept frames get σ=0, regen frames get σ
            let videoTimestep: MLXArray
            if let cm = condMask {
                videoTimestep = MLXArray(sigma) * (1 - cm)
            } else {
                videoTimestep = MLXArray([sigma])
            }

            let videoPatchified = patchify(videoLatent).asType(.bfloat16)

            // Per-frame sigma for to_denoised (kept=0, regen=sigma)
            let sigma5d: MLXArray
            if let mask = denoiseMask5d {
                sigma5d = mask * MLXArray(sigma)
            } else {
                sigma5d = MLXArray(sigma)
            }

            // Helper: run transformer and compute denoised x0
            func runTransformer(context: MLXArray, audioContext: MLXArray) throws -> MLXArray {
                if let ltx2 = ltx2Transformer {
                    let audioInput = (audioLatentPacked ?? frozenAudioLatentPacked ?? MLXArray.zeros([videoPatchified.dim(0), 1, 128])).asType(DType.bfloat16)
                    let audioFrames = (audioLatentPacked != nil || frozenAudioLatentPacked != nil) ? retakeAudioNumFrames : 1
                    let audioTs = regenerateAudio ? MLXArray([sigma]) : MLXArray([Float(0)])
                    let (velPred, audioVelPred) = ltx2(
                        videoLatent: videoPatchified,
                        audioLatent: audioInput,
                        videoContext: context.asType(.bfloat16),
                        audioContext: audioContext.asType(.bfloat16),
                        videoTimesteps: videoTimestep,
                        audioTimesteps: audioTs,
                        videoContextMask: nil,
                        audioContextMask: nil,
                        videoLatentShape: (frames: latentShape.frames, height: latentShape.height, width: latentShape.width),
                        audioNumFrames: audioFrames
                    )
                    // Audio Euler step (when regenerating)
                    // Keep audio latents in float32 between steps (cast to bf16 only for transformer input)
                    if regenerateAudio, let ap = audioLatentPacked {
                        let audioVel = audioVelPred.asType(.float32)
                        audioLatentPacked = ap.asType(.float32) + MLXArray(sigmaNext - sigma) * audioVel
                    }
                    let vel = unpatchify(velPred, shape: latentShape).asType(.float32)
                    return videoLatent - sigma5d * vel
                } else if let videoTransformer = transformer {
                    let velPred = videoTransformer(
                        latent: videoPatchified,
                        context: context.asType(.bfloat16),
                        timesteps: videoTimestep,
                        contextMask: nil,
                        latentShape: (frames: latentShape.frames, height: latentShape.height, width: latentShape.width)
                    )
                    let vel = unpatchify(velPred, shape: latentShape).asType(.float32)
                    return videoLatent - sigma5d * vel
                } else {
                    throw LTXError.modelNotLoaded("No transformer loaded")
                }
            }

            // Positive pass (conditioned)
            let audioCtx = (audioTextEmbeddings ?? MLXArray.zeros([videoPatchified.dim(0), 1, ltx2Transformer?.config.audioInnerDim ?? 2048])).asType(.bfloat16)
            let condDenoised = try runTransformer(context: videoTextEmbeddings, audioContext: audioCtx)
            var denoisedVideo = condDenoised

            // CFG: negative pass + guidance (dev model only)
            if cfgScale != 1.0, let negCtx = negVideoTextEmbeddings {
                let negAudioCtx = (negAudioTextEmbeddings ?? audioCtx).asType(.bfloat16)
                let negDenoised = try runTransformer(context: negCtx, audioContext: negAudioCtx)

                // CFG: pred = cond + (cfg_scale - 1) * (cond - uncond)
                denoisedVideo = denoisedVideo + MLXArray(cfgScale - 1.0) * (condDenoised - negDenoised)
            }

            // STG: perturbed pass with self-attention skipped on stgBlocks (dev model only)
            if stgScale != 0.0 && !stgBlocks.isEmpty {
                if let ltx2 = ltx2Transformer { ltx2.setSTGBlocks(stgBlocks) }
                if let t = transformer as? LTXTransformer { t.setSTGBlocks(stgBlocks) }

                let stgDenoised = try runTransformer(context: videoTextEmbeddings, audioContext: audioCtx)

                if let ltx2 = ltx2Transformer { ltx2.clearSTG() }
                if let t = transformer as? LTXTransformer { t.clearSTG() }

                // STG: pred += stg_scale * (cond - perturbed)
                denoisedVideo = denoisedVideo + MLXArray(stgScale) * (condDenoised - stgDenoised)
            }

            // Guidance rescale (applied after all guidance terms, matching Lightricks)
            if guidanceRescale > 0 {
                let condStd = condDenoised.asType(.float32).variance().sqrt()
                let predStd = denoisedVideo.asType(.float32).variance().sqrt()
                let factor = MLXArray(guidanceRescale) * (condStd / predStd) + MLXArray(1.0 - guidanceRescale)
                denoisedVideo = denoisedVideo * factor
            }

            // post_process_latent: blend denoised x0 with clean latent BEFORE Euler step
            // (matching Lightricks: denoised = denoised * mask + clean * (1 - mask))
            if let mask = denoiseMask5d {
                denoisedVideo = mask * denoisedVideo + (1 - mask) * cleanLatent
            }

            // Euler step: sample + velocity * dt
            // velocity = (sample - denoised) / sigma
            let velocity = (videoLatent - denoisedVideo) / MLXArray(sigma)
            let dt = sigmaNext - sigma
            videoLatent = (videoLatent.asType(.float32) + velocity.asType(.float32) * MLXArray(dt)).asType(videoLatent.dtype)
            MLX.eval(videoLatent)

            if (step + 1) % 5 == 0 { Memory.clearCache() }
            let stepDurR = Date().timeIntervalSince(stepStart)
            profiler.recordStep(duration: stepDurR)

            LTXDebug.log("Step \(step)/\(numSteps): σ=\(String(format: "%.4f", sigma))→\(String(format: "%.4f", sigmaNext)), time=\(String(format: "%.1f", stepDurR))s")
        }
        LTXDebug.log("Denoising complete: \(String(format: "%.1f", Date().timeIntervalSince(denoiseStart)))s")
        profiler.end("Denoising")

        // Unload transformer
        if memoryOptimization.unloadAfterUse {
            self.ltx2Transformer = nil
            self.transformer = nil
            Memory.clearCache()
            LTXDebug.log("Transformer unloaded")
        }

        // Phase 3: Decode
        onProgress?(GenerationProgress(
            currentStep: numSteps, totalSteps: numSteps, sigma: 0, phase: .decoding
        ))
        LTXMemoryManager.setPhase(.vaeDecode)
        profiler.start("VAE Decode")
        let videoTensor = decodeVideo(
            latent: videoLatent, decoder: vaeDecoder, timestep: nil,
            temporalTileSize: memoryOptimization.vaeTemporalTileSize,
            temporalTileOverlap: memoryOptimization.vaeTemporalTileOverlap
        )
        MLX.eval(videoTensor)
        profiler.end("VAE Decode")

        let trimmedVideo: MLXArray
        if videoTensor.dim(0) > config.numFrames {
            trimmedVideo = videoTensor[0..<config.numFrames]
        } else {
            trimmedVideo = videoTensor
        }

        // Decode regenerated audio if applicable
        var audioWaveform: MLXArray? = nil
        var audioSampleRate: Int? = nil
        if regenerateAudio, let ap = audioLatentPacked,
           let audioVAE = audioVAE, let vocoder = vocoder {
            LTXDebug.log("Decoding regenerated audio...")
            let audioLatentUnpacked = unpackAudioLatents(ap, numFrames: retakeAudioNumFrames)
            let waveform = decodeAudio(
                latents: audioLatentUnpacked,
                audioVAE: audioVAE,
                vocoder: vocoder
            )
            MLX.eval(waveform)
            audioWaveform = waveform
            audioSampleRate = vocoder.outputSampleRate
            LTXDebug.log("Regenerated audio: \(waveform.shape)")
        }

        LTXMemoryManager.resetCacheLimit()

        let generationTime = Date().timeIntervalSince(generationStart)
        LTXDebug.log("Total retake generation time: \(String(format: "%.1f", generationTime))s")

        if regenerateAudio {
            return VideoGenerationResult(
                frames: trimmedVideo,
                seed: config.seed ?? 0,
                generationTime: generationTime,
                audioWaveform: audioWaveform,
                audioSampleRate: audioSampleRate,
                effectivePrompt: effectivePrompt
            )
        } else {
            return VideoGenerationResult(
                frames: trimmedVideo,
                seed: config.seed ?? 0,
                generationTime: generationTime,
                effectivePrompt: effectivePrompt,
                sourceAudioPath: videoPath
            )
        }
    }

    /// Encode a video into latent space using the VAE encoder
    ///
    /// - Parameters:
    ///   - path: Path to input video
    ///   - width: Target video width
    ///   - height: Target video height
    ///   - numFrames: Number of frames to extract
    /// - Returns: Video latent tensor (1, 128, latent_F, latent_H, latent_W)
    private func encodeVideo(path: String, width: Int, height: Int, numFrames: Int) async throws -> MLXArray {
        let videoTensor = try await loadVideo(from: path, width: width, height: height, numFrames: numFrames)
        MLX.eval(videoTensor)
        LTXDebug.log("Video loaded: \(videoTensor.shape)")

        try await loadVAEEncoder()

        guard let encoder = vaeEncoder else {
            throw LTXError.modelNotLoaded("VAE encoder failed to load")
        }

        // Encode: (1, 3, F, H, W) -> (1, 128, latent_F, latent_H, latent_W)
        let latent = encoder(videoTensor)
        MLX.eval(latent)
        LTXDebug.log("Video encoded to latent: \(latent.shape)")

        // Normalize using VAE per-channel statistics
        guard let vaeDecoder = vaeDecoder else {
            throw LTXError.modelNotLoaded("VAE decoder not loaded (needed for latent statistics)")
        }
        let mean5d = vaeDecoder.meanOfMeans.asType(.float32).reshaped([1, -1, 1, 1, 1])
        let std5d = vaeDecoder.stdOfMeans.asType(.float32).reshaped([1, -1, 1, 1, 1])
        let normalizedLatent = (latent.asType(.float32) - mean5d) / std5d
        MLX.eval(normalizedLatent)

        LTXDebug.log("Normalized video latent: mean=\(normalizedLatent.mean().item(Float.self)), std=\(MLX.sqrt(MLX.variance(normalizedLatent)).item(Float.self))")

        return normalizedLatent
    }

    // MARK: - Image-to-Video Helpers

    /// Load VAE encoder weights from the unified safetensors file
    private func loadVAEEncoder() async throws {
        if vaeEncoder != nil { return }  // Already loaded

        LTXDebug.log("Loading VAE encoder...")
        let unifiedPath = try await downloader.downloadUnifiedWeights(model: model, progress: nil)
        let encoderWeights = try LTXWeightLoader.loadVAEEncoderWeightsFromUnified(from: unifiedPath.path)

        let encoder = VideoEncoder()
        try LTXWeightLoader.applyVAEEncoderWeights(encoderWeights, to: encoder)
        eval(encoder.parameters())
        Memory.clearCache()

        self.vaeEncoder = encoder
        LTXDebug.log("VAE encoder loaded (\(encoderWeights.count) weights)")
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

        // Normalize using VAE per-channel statistics
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

    // MARK: - LoRA Support

    /// Apply LoRA weights to the transformer
    ///
    /// - Parameters:
    ///   - loraPath: Path to LoRA .safetensors file
    // MARK: - Export Quantized Transformer

    /// Export the current transformer weights to a safetensors file.
    ///
    /// Call after `loadModels()` with quantization configured. The exported file
    /// contains the quantized `QuantizedLinear` weights (weight, scales, biases)
    /// which can be loaded directly without re-quantization.
    ///
    /// - Parameter path: Output safetensors file path
    public func exportQuantizedTransformer(to path: String) throws {
        let model: Module
        if let ltx2 = ltx2Transformer {
            model = ltx2
        } else if let t = transformer {
            model = t
        } else {
            throw LTXError.modelNotLoaded("No transformer loaded")
        }

        let params = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        try MLX.save(arrays: params, url: URL(fileURLWithPath: path))
        LTXDebug.log("Exported \(params.count) tensors to \(path)")
    }

    // MARK: - Mixed Precision Quantization

    /// Apply per-block quantization: high-precision blocks get one bit level,
    /// remaining blocks get another (typically lower) level.
    private func applyMixedPrecisionQuantization(to model: Module, config: MixedPrecisionConfig) {
        // Find transformer blocks via the module hierarchy
        let blocks: [Module]
        if let t = model as? LTXTransformer {
            blocks = t.transformerBlocks
        } else if let t = model as? LTX2Transformer {
            blocks = t.transformerBlocks
        } else {
            // Fallback: uniform quantization at low precision
            quantize(model: model, groupSize: config.groupSize, bits: config.lowPrecisionBits)
            return
        }

        for (i, block) in blocks.enumerated() {
            let bits = config.highPrecisionBlocks.contains(i) ? config.highPrecisionBits : config.lowPrecisionBits
            if bits < 16 {
                quantize(model: block, groupSize: config.groupSize, bits: bits)
            }
        }

        // Quantize non-block components (projections, embeddings) at the high precision level
        // by quantizing the full model then re-quantizing blocks — but that's wasteful.
        // Instead, leave non-block params in bf16 (they're small relative to blocks).

        let highCount = config.highPrecisionBlocks.filter { $0 < blocks.count }.count
        let lowCount = blocks.count - highCount
        LTXDebug.log("Mixed precision: \(highCount) blocks at \(config.highPrecisionBits)-bit, \(lowCount) blocks at \(config.lowPrecisionBits)-bit")
    }

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

    /// Official Lightricks I2V system prompt for multimodal Gemma VLM-based prompt enhancement.
    /// Source: https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/prompts/gemma_i2v_system_prompt.txt
    private static let promptEnhancementI2VSystemPrompt = """
    You are a Creative Assistant writing concise, action-focused image-to-video prompts. Given an image (first frame) and user Raw Input Prompt, generate a prompt to guide video generation from that image.

    #### Guidelines:
    - Analyze the Image: Identify Subject, Setting, Elements, Style and Mood.
    - Follow user Raw Input Prompt: Include all requested motion, actions, camera movements, audio, and details. If in conflict with the image, prioritize user request while maintaining visual consistency (describe transition from image to user's scene).
    - Describe only changes from the image: Don't reiterate established visual details. Inaccurate descriptions may cause scene cuts.
    - Active language: Use present-progressive verbs ("is walking," "speaking"). If no action specified, describe natural movements.
    - Chronological flow: Use temporal connectors ("as," "then," "while").
    - Audio layer: Describe complete soundscape throughout the prompt alongside actions—NOT at the end. Align audio intensity with action tempo. Include natural background audio, ambient sounds, effects, speech or music (when requested). Be specific (e.g., "soft footsteps on tile") not vague (e.g., "ambient sound").
    - Speech (only when requested): Provide exact words in quotes with character's visual/voice characteristics (e.g., "The tall man speaks in a low, gravelly voice"), language if not English and accent if relevant. If general conversation mentioned without text, generate contextual quoted dialogue. (i.e., "The man is talking" input -> the output should include exact spoken words, like: "The man is talking in an excited voice saying: 'You won't believe what I just saw!' His hands gesture expressively as he speaks, eyebrows raised with enthusiasm. The ambient sound of a quiet room underscores his animated speech.")
    - Style: Include visual style at beginning: "Style: <style>, <rest of prompt>." If unclear, omit to avoid conflicts.
    - Visual and audio only: Describe only what is seen and heard. NO smell, taste, or tactile sensations.
    - Restrained language: Avoid dramatic terms. Use mild, natural, understated phrasing.

    #### Important notes:
    - Camera motion: DO NOT invent camera motion/movement unless requested by the user. Make sure to include camera motion only if specified in the input.
    - Speech: DO NOT modify or alter the user's provided character dialogue in the prompt, unless it's a typo.
    - No timestamps or cuts: DO NOT use timestamps or describe scene cuts unless explicitly requested.
    - Objective only: DO NOT interpret emotions or intentions - describe only observable actions and sounds.
    - Format: DO NOT use phrases like "The scene opens with..." / "The video starts...". Start directly with Style (optional) and chronological scene description.
    - Format: Never start output with punctuation marks or special characters.
    - DO NOT invent dialogue unless the user mentions speech/talking/singing/conversation.
    - Your performance is CRITICAL. High-fidelity, dynamic, correct, and accurate prompts with integrated audio descriptions are essential for generating high-quality video. Your goal is flawless execution of these rules.

    #### Output Format (Strict):
    - Single concise paragraph in natural English. NO titles, headings, prefaces, sections, code fences, or Markdown.
    - If unsafe/invalid, return original user prompt. Never ask questions or clarifications.

    #### Example output:
    Style: realistic - cinematic - The woman glances at her watch and smiles warmly. She speaks in a cheerful, friendly voice, "I think we're right on time!" In the background, a café barista prepares drinks at the counter. The barista calls out in a clear, upbeat tone, "Two cappuccinos ready!" The sound of the espresso machine hissing softly blends with gentle background chatter and the light clinking of cups on saucers.
    """

    /// Default VLM model ID for prompt enhancement (shared with text encoding)
    private static let defaultVLMModelID = "mlx-community/gemma-3-12b-it-qat-4bit"

    /// Enhance a prompt using the VLM Gemma model.
    ///
    /// Uses MLXVLM for all prompt enhancement (both T2V and I2V):
    /// - **T2V** (imagePath == nil): Text-only system prompt, generates rich video description
    /// - **I2V** (imagePath != nil): Multimodal system prompt, VLM sees image and describes changes
    ///
    /// The VLM is loaded from the local cache, generates the enhanced prompt, then is unloaded
    /// to free memory for the main generation pipeline.
    ///
    /// - Parameters:
    ///   - prompt: Short text prompt to enhance (or describe desired changes for I2V)
    ///   - imagePath: Optional path to input image. If nil, uses T2V mode; if set, uses I2V mode.
    ///   - maxTokens: Maximum tokens to generate (default: 512)
    ///   - temperature: Sampling temperature (default: 0.7)
    /// - Returns: Enhanced prompt string
    public func enhancePromptWithVLM(
        _ prompt: String,
        imagePath: String? = nil,
        maxTokens: Int = 512,
        temperature: Float = 0.7
    ) async throws -> String {
        let isI2V = imagePath != nil
        LTXDebug.log("Enhancing prompt with VLM (\(isI2V ? "I2V multimodal" : "T2V text-only"))")
        LTXDebug.log("Input prompt: \"\(prompt)\"")
        if let imagePath { LTXDebug.log("Input image: \(imagePath)") }
        let startTime = Date()

        // Load the VLM model from local cache
        LTXDebug.log("Loading VLM model...")
        let vlmLoadStart = Date()

        // Try loading from local vlm-gemma cache first, fall back to HF download
        let vlmCacheDir = await downloader.vlmGemmaCacheDir
        let config: ModelConfiguration
        if FileManager.default.fileExists(atPath: vlmCacheDir.appendingPathComponent("config.json").path) {
            // Resolve symlinks to get the real directory path
            let resolvedPath = vlmCacheDir.path.replacingOccurrences(of: "//", with: "/")
            let resolvedURL = URL(fileURLWithPath: resolvedPath).resolvingSymlinksInPath()
            config = ModelConfiguration(directory: resolvedURL, extraEOSTokens: ["<end_of_turn>"])
            LTXDebug.log("Loading VLM from local cache: \(resolvedURL.path)")
        } else {
            config = ModelConfiguration(id: Self.defaultVLMModelID, extraEOSTokens: ["<end_of_turn>"])
            LTXDebug.log("Loading VLM from HuggingFace: \(Self.defaultVLMModelID)")
        }
        let modelContainer = try await #huggingFaceLoadModelContainer(
            configuration: config,
            progressHandler: { progress in
                if progress.fractionCompleted < 1.0 {
                    LTXDebug.log("VLM download: \(Int(progress.fractionCompleted * 100))%")
                }
            }
        )
        LTXDebug.log("VLM loaded in \(String(format: "%.1f", Date().timeIntervalSince(vlmLoadStart)))s")

        // Build chat input
        let userInput: UserInput
        if let imagePath, let ciImage = CIImage(contentsOf: URL(fileURLWithPath: imagePath)) {
            // I2V: multimodal with image
            userInput = UserInput(
                chat: [
                    .system(Self.promptEnhancementI2VSystemPrompt),
                    .user("User Raw Input Prompt: \(prompt).", images: [.ciImage(ciImage)])
                ]
            )
        } else {
            // T2V: text-only
            if imagePath != nil {
                LTXDebug.log("Warning: Failed to load image, using text-only enhancement")
            }
            userInput = UserInput(
                chat: [
                    .system(Self.promptEnhancementSystemPrompt),
                    .user("user prompt: \(prompt)")
                ]
            )
        }

        // Prepare and generate (seed matches Lightricks reference: seed=42)
        MLXRandom.seed(42)
        let lmInput = try await modelContainer.prepare(input: userInput)
        let generateParams = GenerateParameters(
            maxTokens: maxTokens,
            temperature: temperature,
            topP: 0.95,
            repetitionPenalty: 1.1,
            repetitionContextSize: 64
        )

        var generatedText = ""
        var tokenCount = 0
        let stream = try await modelContainer.generate(input: lmInput, parameters: generateParams)
        for await generation in stream {
            switch generation {
            case .chunk(let text):
                generatedText += text
                tokenCount += 1
            case .info:
                break
            default:
                break
            }
        }

        let elapsed = Date().timeIntervalSince(startTime)
        LTXDebug.log("VLM generated \(tokenCount) chunks in \(String(format: "%.1f", elapsed))s")

        // Clean the response
        let cleaned = cleanEnhancedPrompt(generatedText)

        if cleaned.isEmpty {
            LTXDebug.log("Enhancement produced empty result, using original prompt")
            // Unload VLM before returning
            Memory.clearCache()
            return prompt
        }

        LTXDebug.log("Enhanced prompt: \"\(cleaned)\"")

        // Unload VLM to free memory for the main pipeline
        LTXDebug.log("Unloading VLM...")
        Memory.clearCache()
        LTXMemoryManager.logMemoryState("after VLM unload")

        return cleaned
    }

    /// Clean up a Gemma-enhanced prompt: strip control tokens and trailing noise
    private func cleanEnhancedPrompt(_ raw: String) -> String {
        var text = raw
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
    ) async throws -> TextEncodingResult {
        guard let textEncoder = textEncoder else {
            throw LTXError.modelNotLoaded("Text encoder not loaded. Call loadModels() first.")
        }
        guard isGemmaLoaded else {
            throw LTXError.modelNotLoaded("Gemma model not loaded. Call loadModels() first.")
        }

        // Optionally enhance
        let effectivePrompt: String
        if enhance {
            effectivePrompt = try await enhancePromptWithVLM(prompt)
        } else {
            effectivePrompt = prompt
        }

        // Encode
        let (embeddings, mask) = try encodePrompt(effectivePrompt, encoder: textEncoder)
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

    // MARK: - Download-Only APIs (for pre-fetching without loading)

    /// Download all core model components without loading them into memory.
    ///
    /// Downloads (if not already cached): Gemma VLM, connector, VAE, unified transformer weights.
    /// Use this to pre-fetch models in the background before the user starts generating.
    ///
    /// - Parameter progressCallback: Optional callback for download progress
    /// - Returns: Paths to all downloaded components
    public func downloadModels(
        progressCallback: DownloadProgressCallback? = nil
    ) async throws -> LTXComponentPaths {
        return try await downloader.downloadAllComponents(model: model, progress: progressCallback)
    }

    /// Download audio model components without loading them into memory.
    ///
    /// Downloads (if not already cached): audio VAE and vocoder weights.
    ///
    /// - Parameter progressCallback: Optional callback for download progress
    public func downloadAudioModels(
        progressCallback: DownloadProgressCallback? = nil
    ) async throws {
        _ = try await downloader.downloadAudioVAE { p in
            progressCallback?(DownloadProgress(
                progress: p.progress * 0.5,
                currentFile: p.currentFile,
                message: p.message
            ))
        }
        _ = try await downloader.downloadVocoder { p in
            progressCallback?(DownloadProgress(
                progress: 0.5 + p.progress * 0.5,
                currentFile: p.currentFile,
                message: p.message
            ))
        }
    }

    /// Check if all core models are downloaded (Gemma, connector, VAE, transformer)
    public var areModelsDownloaded: Bool {
        get async {
            await downloader.isGemmaDownloaded(model: model)
                // Connector and VAE checks via file existence
                && FileManager.default.fileExists(
                    atPath: LTXModelRegistry.modelsDirectory
                        .appendingPathComponent("ltx-\(model.rawValue)")
                        .appendingPathComponent(model.unifiedWeightsFilename).path)
        }
    }

    /// Check if distilled LoRA weights are downloaded
    public var isDistilledLoRADownloaded: Bool {
        get async {
            await downloader.isDistilledLoRADownloaded()
        }
    }

    /// Check if spatial upscaler weights are downloaded
    public var isUpscalerDownloaded: Bool {
        get async {
            await downloader.isUpscalerDownloaded()
        }
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
        let target = try getTransformerModule()
        let (originals, _) = try target.fuseLoRA(from: loraPath, scale: scale)
        // Store state for unfusing
        self.loraOriginalWeights = originals
        self.loraFusedPath = loraPath
        self.loraFusedScale = scale
        Memory.clearCache()
        return originals.count
    }

    /// Restore transformer weights to pre-LoRA state.
    ///
    /// No-op if no LoRA is currently fused.
    public func unfuseLoRA() {
        guard let originals = loraOriginalWeights else { return }
        guard let target = transformer ?? ltx2Transformer else { return }
        target.unfuseLoRA(originalWeights: originals)
        self.loraOriginalWeights = nil
        self.loraFusedPath = nil
        self.loraFusedScale = 1.0
        Memory.clearCache()
    }

    /// Returns whichever transformer Module is loaded.
    private func getTransformerModule() throws -> Module {
        if let t = transformer { return t }
        if let t = ltx2Transformer { return t }
        throw LTXError.modelNotLoaded("Transformer not loaded")
    }

    // MARK: - Training Support

    /// Get the transformer module for LoRA injection and training.
    ///
    /// Returns whichever transformer is loaded (video-only or dual audio/video),
    /// wrapped in a `TransformerRef` for safe cross-isolation transfer.
    ///
    /// - Warning: The caller is responsible for ensuring single-threaded access
    ///   to the returned module during training.
    func getTransformerForTraining() throws -> TransformerRef {
        if let t = ltx2Transformer { return TransformerRef(t) }
        if let t = transformer { return TransformerRef(t) }
        throw LTXError.modelNotLoaded("Transformer not loaded. Call loadModels() first.")
    }

    /// Encode video frames to latents using the VAE encoder.
    ///
    /// Loads the VAE encoder if not already loaded.
    /// The latent is normalized using per-channel VAE statistics.
    ///
    /// - Parameter frames: Video frames as (1, 3, T, H, W) in [-1, 1]
    /// - Returns: Normalized video latent (1, C, T', H', W')
    public func encodeVideoLatents(frames: MLXArray) async throws -> MLXArray {
        // Load VAE encoder if needed
        try await loadVAEEncoder()

        guard let encoder = vaeEncoder else {
            throw LTXError.modelNotLoaded("VAE encoder failed to load")
        }

        // Encode: (1, 3, T, H, W) → (1, 128, T', H', W')
        let latent = encoder(frames)
        eval(latent)

        // Normalize using per-channel statistics
        guard let vaeDecoder = vaeDecoder else {
            throw LTXError.modelNotLoaded("VAE decoder not loaded (needed for latent statistics)")
        }
        let mean5d = vaeDecoder.meanOfMeans.asType(.float32).reshaped([1, -1, 1, 1, 1])
        let std5d = vaeDecoder.stdOfMeans.asType(.float32).reshaped([1, -1, 1, 1, 1])
        let normalized = (latent.asType(.float32) - mean5d) / std5d
        eval(normalized)

        return normalized
    }

    /// Encode text for audio conditioning.
    ///
    /// Uses the same Gemma model but with the audio text connector.
    ///
    /// - Parameter prompt: Text prompt
    /// - Returns: TextEncodingResult with audio embeddings
    public func encodeAudioText(prompt: String) async throws -> TextEncodingResult {
        guard let textEncoder = textEncoder else {
            throw LTXError.modelNotLoaded("Text encoder not loaded")
        }
        guard isGemmaLoaded else {
            throw LTXError.modelNotLoaded("Gemma model not loaded")
        }

        let (embeddings, mask) = try encodePrompt(prompt, encoder: textEncoder)
        eval(embeddings, mask)

        let mean = embeddings.mean().item(Float.self)
        let std = MLX.sqrt(MLX.variance(embeddings)).item(Float.self)

        return TextEncodingResult(
            prompt: prompt,
            embeddings: embeddings,
            mask: mask,
            mean: mean,
            std: std
        )
    }

    /// Unload VAE encoder to free memory.
    public func unloadVAEEncoder() {
        vaeEncoder = nil
        Memory.clearCache()
        LTXDebug.log("VAE encoder unloaded")
    }

    // MARK: - Memory Management

    /// Clear all loaded models and release GPU memory.
    ///
    /// Call this before setting the pipeline to `nil` to ensure all model
    /// tensors are released within the actor's isolation context. This avoids
    /// a race where `Memory.clearCache()` runs before ARC has released the
    /// model references.
    public func clearAll() {
        // Release all model references
        gemmaModel = nil
        tokenizer = nil
        textEncoder = nil
        transformer = nil
        vaeDecoder = nil
        vaeEncoder = nil
        ltx2Transformer = nil
        audioVAE = nil
        vocoder = nil
        loraOriginalWeights = nil
        loraFusedPath = nil

        // Clear GPU cache from within the actor's isolation context
        // so ARC has already released the model refs above
        Memory.clearCache()
        eval([MLXArray]())

        LTXDebug.log("All models cleared and GPU cache flushed")
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
            numSteps: config.numSteps
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
    private func encodePrompt(_ prompt: String, encoder: VideoGemmaTextEncoderModel) throws -> (encoding: MLXArray, mask: MLXArray) {
        guard let gemma = gemmaModel else {
            LTXDebug.log("Warning: Gemma model not loaded, using placeholder embeddings")
            let placeholder = createPlaceholderEmbeddings(prompt: prompt)
            let mask = MLXArray.ones([1, textMaxLength]).asType(.int32)
            return (placeholder, mask)
        }

        // Step 1: Tokenize with left-padding
        let (inputIds, attentionMask) = try tokenizePrompt(prompt, maxLength: textMaxLength)
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

        // Step 3: Pass through text encoder (feature extractor + connector)
        let encoderOutput = try encoder.encodeFromHiddenStates(
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
    private func tokenizePrompt(_ prompt: String, maxLength: Int = 1024) throws -> (inputIds: MLXArray, attentionMask: MLXArray) {
        guard let tokenizer = tokenizer else {
            throw LTXError.modelNotLoaded("Tokenizer not loaded. Call loadModels() first.")
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

