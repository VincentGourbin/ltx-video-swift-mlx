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
import Gemma4Swift
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

    /// Resolved unified weights file paths associated with this pipeline, keyed by model.
    ///
    /// When the caller supplies a local unified safetensors file via `loadModels`,
    /// later lazy loads such as the I2V VAE encoder should reuse that same file
    /// instead of falling back to the downloader.
    private var unifiedWeightsPathCache = UnifiedWeightsPathCache()

    /// Flow-matching scheduler
    private let scheduler: LTXScheduler

    /// Prompt encoder: Gemma 3 for LTX-2.3, Gemma 4 for LTX-2.5. Both produce the
    /// 49 hidden states the feature extractor consumes, so every downstream stage
    /// is generation-agnostic.
    private var gemmaEncoder: (any LTXGemmaEncoding)?

    /// Text encoder (feature extractor + connector)
    private var textEncoder: VideoGemmaTextEncoderModel?

    /// Diffusion transformer
    internal var transformer: LTXTransformer?

    /// VAE decoder
    internal var vaeDecoder: VideoDecoder?

    /// LTX-2.5's diffusion video decoder, when the caller opted into it. The
    /// conv decoder stays loaded either way: it owns the latent statistics the
    /// rest of the pipeline reads, and it is the fallback.
    internal var diffusionVAEDecoder: DiffusionVideoDecoder?

    /// VAE encoder (loaded only for image-to-video)
    private var vaeEncoder: VideoEncoder?

    /// Audio: dual video/audio transformer (alternative to video-only transformer)
    internal var ltx2Transformer: LTX2Transformer?

    /// Audio VAE decoder
    private var audioVAE: AudioVAE?

    /// Audio vocoder (mel → waveform)
    private var vocoder: (any LTXVocoding)?

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

    /// Identity of the LipDub IC-LoRA currently fused into `ltx2Transformer`
    /// (nil = pristine weights). Unlike `loraFusedPath`, no original weights are
    /// kept (too large for the 22B transformer): consecutive `generateLipDub`
    /// runs with the SAME LoRA file AND the same scale reuse the fused
    /// transformer as-is; switching LoRA, changing the scale — or the file
    /// changing under the same path — requires reloading the models. Cleared
    /// wherever the transformer is unloaded or recreated.
    private struct LipDubFusionRecord {
        /// Canonical path (symlinks resolved, standardized) so path spelling
        /// differences between segments don't force a needless 22B reload.
        let path: String
        /// File mtime at fusion time — detects the file being overwritten in
        /// place between segments (same path, different weights).
        let modificationDate: Date?
        /// Scale the delta was fused at. Part of the identity: the same file
        /// fused at a different scale is a different set of weights, and the
        /// fusion is destructive (no originals kept), so it cannot be adjusted
        /// in place.
        let scale: Float
    }
    private var lipdubFusion: LipDubFusionRecord? = nil

    /// Canonical path of the LipDub IC-LoRA currently fused into the loaded
    /// transformer, or nil when the weights are pristine. Exposed so host apps
    /// can decide whether the next LipDub segment can reuse the loaded pipeline
    /// (same LoRA → no reload needed) or must call `loadModels()` again.
    public var fusedLipDubLoRAPath: String? { lipdubFusion?.path }

    /// Scale the currently-fused LipDub IC-LoRA was applied at, or nil when the
    /// weights are pristine. A host app reusing the pipeline across segments
    /// must pass this same scale — see `fusedLipDubLoRAPath`.
    public var fusedLipDubLoRAScale: Float? { lipdubFusion?.scale }

    /// Validate a LipDub IC-LoRA scale, returning the warning to print when the
    /// value is usable but far from the published 1.0 (nil when it is in range).
    ///
    /// Static and separate from `generateLipDub` so it can be unit-tested on its
    /// own: inside the pipeline this check necessarily runs *after* the
    /// models-loaded guard, which a fast test cannot satisfy — a test calling
    /// `generateLipDub` on an unloaded pipeline would pass on `modelNotLoaded`
    /// and prove nothing about this rule.
    static func validateLipDubLoRAScale(_ scale: Float) throws -> String? {
        guard scale > 0 else {
            throw LTXError.invalidConfiguration(
                "lipdubLoRAScale must be > 0 (got \(scale)); 0 would run the base weights, "
                + "which have never seen the appended reference tokens.")
        }
        guard scale < 0.5 || scale > 1.5 else { return nil }
        return String(
            format: "[lipdub] WARNING: LoRA scale %.2f is far from the published 1.0. "
                + "This IC-LoRA carries the reference-token conditioning itself, not a "
                + "style — expect degraded lip-sync rather than a softer effect.",
            scale)
    }

    private static func canonicalLoRAPath(_ path: String) -> String {
        URL(fileURLWithPath: path).resolvingSymlinksInPath().standardizedFileURL.path
    }

    private static func loraModificationDate(_ path: String) -> Date? {
        (try? FileManager.default.attributesOfItem(atPath: path))?[.modificationDate] as? Date
    }

    /// Throws when the LipDub IC-LoRA is fused into the loaded transformer: its
    /// delta is destructive (no pristine weights kept) and would corrupt any
    /// non-LipDub use of the weights.
    private func ensureNoLipDubLoRAFused(wouldCorrupt operation: String) throws {
        if let fused = lipdubFusion {
            throw LTXError.invalidConfiguration(
                "The LipDub IC-LoRA is fused into the loaded transformer " +
                "(\(fused.path)) and would corrupt \(operation). " +
                "Call loadModels() + loadAudioModels() to restore pristine weights."
            )
        }
    }

    /// Unload Gemma + tokenizer (~7.5 GB) when the memory config asks for it.
    /// Kept resident with `unloadAfterUse == false` so consecutive runs can
    /// re-encode text without reloading models.
    internal func unloadGemmaIfConfigured() {
        guard memoryOptimization.unloadAfterUse else { return }
        gemmaEncoder = nil
        Memory.clearCache()
    }

    /// Whether models are loaded (Gemma may be nil after unloading post-encoding)
    public var isLoaded: Bool {
        textEncoder != nil && (transformer != nil || ltx2Transformer != nil) && vaeDecoder != nil
    }

    /// Whether Gemma model is available for text encoding
    public var isGemmaLoaded: Bool {
        gemmaEncoder != nil
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

    /// Resolve the unified safetensors path for a model.
    ///
    /// Resolution order is: explicit caller override, cached path for the model
    /// if it still exists, then downloader/cache lookup. Explicit overrides are
    /// cached as-is so later lazy component loads use the same caller-supplied file.
    /// Resolve every file the current model needs, downloading what is missing.
    ///
    /// `overrideTransformerPath` names the transformer file (the whole checkpoint
    /// for unified layouts); the other components still resolve through the cache.
    private func resolveCheckpoint(
        overrideTransformerPath: String? = nil,
        progressCallback: DownloadProgressCallback? = nil
    ) async throws -> LTXCheckpointPaths {
        try model.validateRunnable()

        switch model.weightsLayout {
        case .unified:
            let path = try await resolveUnifiedWeightsPath(
                for: model, overridePath: overrideTransformerPath,
                progressCallback: progressCallback)
            let url = URL(fileURLWithPath: path)
            return LTXCheckpointPaths(transformer: url, videoVAE: url)

        case .split:
            var paths = try await downloader.downloadCheckpoint(model: model) { progress in
                progressCallback?(progress)
            }
            if let overrideTransformerPath {
                // Keep the resolved audio bundle: the default init aliases it to
                // the transformer path, which for an override is a file with no
                // vocoder.* keys — the silent 24 kHz-fallback path.
                paths = LTXCheckpointPaths(
                    transformer: URL(fileURLWithPath: overrideTransformerPath),
                    videoVAE: paths.videoVAE,
                    textEncoder: paths.textEncoder,
                    audioBundle: paths.audioBundle)
            }
            unifiedWeightsPathCache.store(paths.transformer.path, for: model)
            return paths
        }
    }

    /// x₀-space guidance combination shared by every dev-guided loop (retake,
    /// dev single-stage): CFG toward `neg`, STG toward `stg`, then variance
    /// rescale toward `cond` — matching the Lightricks order, all on x₀.
    nonisolated static func combineGuidance(
        cond: MLXArray, neg: MLXArray?, stg: MLXArray?,
        cfgScale: Float, stgScale: Float, guidanceRescale: Float
    ) -> MLXArray {
        var combined = cond
        if cfgScale != 1.0, let neg {
            // CFG: pred = cond + (cfg_scale - 1) * (cond - uncond)
            combined = combined + MLXArray(cfgScale - 1.0) * (cond - neg)
        }
        if stgScale != 0.0, let stg {
            // STG: pred += stg_scale * (cond - perturbed)
            combined = combined + MLXArray(stgScale) * (cond - stg)
        }
        if guidanceRescale > 0 {
            let condStd = cond.asType(.float32).variance().sqrt()
            let predStd = combined.asType(.float32).variance().sqrt()
            let factor = MLXArray(guidanceRescale) * (condStd / predStd)
                + MLXArray(1.0 - guidanceRescale)
            combined = combined * factor
        }
        return combined
    }

    /// Load LTX-2.5's diffusion video decoder and use it for every decode from
    /// then on.
    ///
    /// Opt-in: it is a separate ~1.5 GB download, and it costs more per decode
    /// than the convolutional decoder (a full attention pass over the pixel
    /// volume rather than a stack of convolutions). The conv decoder stays
    /// loaded — the pipeline reads its latent statistics elsewhere, and it is
    /// the fallback if this one is unloaded.
    ///
    /// - Throws: when the checkpoint's generation ships no diffusion decoder.
    public func loadDiffusionDecoder(
        progressCallback: DownloadProgressCallback? = nil
    ) async throws {
        guard model.family == .ltx25 else {
            throw LTXError.invalidConfiguration(
                "The diffusion video decoder ships from LTX-2.5 onward; "
                + "\(model.displayName) has only the convolutional one.")
        }
        let path = try await downloader.downloadDiffusionVideoVAE(
            model: model, progress: progressCallback)
        LTXDebug.log("Loading diffusion video decoder from \(path.lastPathComponent)...")
        diffusionVAEDecoder = try DiffVAEWeightLoader.load(from: path.path)
        LTXDebug.log("Diffusion video decoder ready")
    }

    /// Drop the diffusion decoder, reverting to the convolutional one.
    public func unloadDiffusionDecoder() {
        diffusionVAEDecoder = nil
        Memory.clearCache()
    }

    /// Decode a video latent to `[F, H, W, C]` frames in `[0, 1]`.
    ///
    /// One funnel for every path (generate, retake, dev single-stage, IC-LoRA)
    /// so the decoder choice is made once: the diffusion decoder when it was
    /// loaded, the convolutional one otherwise.
    func decodeFrames(latent: MLXArray, timestep: Float? = nil) -> MLXArray {
        if let diffusion = diffusionVAEDecoder {
            return decodeVideo(
                latent: latent, decoder: diffusion,
                temporalTileSize: memoryOptimization.vaeTemporalTileSize,
                temporalTileOverlap: memoryOptimization.vaeTemporalTileOverlap)
        }
        guard let conv = vaeDecoder else {
            return MLXArray.zeros([0])
        }
        return decodeVideo(
            latent: latent, decoder: conv, timestep: timestep,
            temporalTileSize: memoryOptimization.vaeTemporalTileSize,
            temporalTileOverlap: memoryOptimization.vaeTemporalTileOverlap)
    }

    /// Build the prompt encoder this checkpoint was trained with.
    private func loadGemmaEncoder(
        checkpoint: LTXCheckpointPaths,
        source: LTXCheckpointSource,
        textEncoderAssets: LTX25TextEncoderAssets? = nil,
        gemmaModelPath: String?,
        tokenizerPath: String?,
        progressCallback: DownloadProgressCallback?
    ) async throws -> any LTXGemmaEncoding {
        switch model.textEncoder {
        case .gemma4_12bLTX:
            guard let textEncoderPath = checkpoint.textEncoder else {
                throw LTXError.modelNotLoaded(
                    "\(model.displayName) needs its bundled Gemma 4 encoder, which was not resolved")
            }
            LTXDebug.log("Loading Gemma 4 encoder from \(textEncoderPath.lastPathComponent)...")
            return try await Gemma4TextEncoder.load(
                assets: textEncoderAssets ?? LTX25TextEncoderAssets(fileURL: textEncoderPath),
                tokenizerCacheDirectory: LTXModelRegistry.modelsDirectory
                    .appendingPathComponent("ltx25-gemma4-tokenizer", isDirectory: true),
                transformerMetadata: try source.transformerMetadata(),
                quantization: quantization.textEncoder)

        case .gemma3_12b:
            let gemmaURL: URL
            let tokenizerURL: URL
            if let gemmaModelPath {
                gemmaURL = URL(fileURLWithPath: gemmaModelPath)
                tokenizerURL = tokenizerPath.map { URL(fileURLWithPath: $0) } ?? gemmaURL
            } else {
                LTXDebug.log("Downloading Gemma text encoder for \(model.displayName) (if needed)...")
                let paths = try await downloader.downloadGemma(model: model) { progress in
                    progressCallback?(progress)
                }
                gemmaURL = paths.modelDir
                tokenizerURL = paths.tokenizerDir
            }
            LTXDebug.log("Loading Gemma3 model from \(gemmaURL.path)...")
            return Gemma3Encoder(
                model: try Gemma3WeightLoader.loadModel(from: gemmaURL),
                tokenizer: try await AutoTokenizer.from(modelFolder: tokenizerURL))
        }
    }

    private func resolveUnifiedWeightsPath(
        for model: LTXModel,
        overridePath: String? = nil,
        progressCallback: DownloadProgressCallback? = nil
    ) async throws -> String {
        // Fail here rather than after a 40 GB download or with a wall of unmatched
        // weight keys: catalogued-but-unimplemented variants have no runnable path.
        try model.validateRunnable()

        if let overridePath {
            unifiedWeightsPathCache.store(overridePath, for: model)
            return overridePath
        }

        if let cachedPath = unifiedWeightsPathCache.cachedPath(for: model) {
            return cachedPath
        }

        LTXDebug.log("Downloading unified weights for \(model.displayName) (if needed)...")
        let downloadedPath = try await downloader.downloadUnifiedWeights(model: model) { progress in
            progressCallback?(progress)
        }
        unifiedWeightsPathCache.store(downloadedPath.path, for: model)
        return downloadedPath.path
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
        let beacon = RuntimeBeacon.begin(task: "load-models", model: model.rawValue)
        defer { beacon?.end() }
        var stepStart = Date()

        // Step 1: Resolve the checkpoint's files (one for 2.3, several for 2.5)
        progressCallback?(DownloadProgress(progress: 0.1, message: "Resolving checkpoint..."))
        let checkpoint = try await resolveCheckpoint(
            overrideTransformerPath: ltxWeightsPath, progressCallback: progressCallback)
        let source = LTXCheckpointSource(model: model, paths: checkpoint)

        // Step 2: Load the prompt encoder — Gemma 4 ships inside 2.5 checkpoints,
        // Gemma 3 is an external download for 2.3.
        progressCallback?(DownloadProgress(progress: 0.2, message: "Loading text encoder..."))
        stepStart = Date()
        // Parse the (mmap'd) text-encoder bundle once: both the Gemma encoder
        // and the aggregate projections read from it.
        let textEncoderAssets = try checkpoint.textEncoder.map { try LTX25TextEncoderAssets(fileURL: $0) }
        gemmaEncoder = try await loadGemmaEncoder(
            checkpoint: checkpoint,
            source: source,
            textEncoderAssets: textEncoderAssets,
            gemmaModelPath: gemmaModelPath,
            tokenizerPath: tokenizerPath,
            progressCallback: progressCallback)
        LTXDebug.log("[TIME] Text encoder load: \(String(format: "%.1f", Date().timeIntervalSince(stepStart)))s")

        // Step 3: Load LTX component weights
        progressCallback?(DownloadProgress(progress: 0.3, message: "Loading \(model.family.displayName) weights..."))
        stepStart = Date()
        let split = try source.loadComponents(textEncoderAssets: textEncoderAssets)
        let transformerWeights = split.transformer
        let vaeWeights = split.vae
        let connectorWeights = split.connector
        LTXDebug.log("[TIME] Load checkpoint components: \(String(format: "%.1f", Date().timeIntervalSince(stepStart)))s")

        // Step 3: Create and load transformer
        progressCallback?(DownloadProgress(progress: 0.5, message: "Loading transformer..."))

        let transformerConfig = model.transformerConfig
        LTXVideoProfiler.shared.start("Load Transformer")
        transformer = LTXTransformer(config: transformerConfig, memoryOptimization: memoryOptimization)

        stepStart = Date()
        LTXDebug.log("Applying \(transformerWeights.count) transformer weights...")
        try LTXWeightLoader.applyTransformerWeights(transformerWeights, to: transformer!)
        LTXVideoProfiler.shared.end("Load Transformer")
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

        LTXVideoProfiler.shared.start("Load Video VAE")
        vaeDecoder = VideoDecoder()
        // LTX-2.3 unified file doesn't include standalone vae/config.json;
        // timestep_conditioning defaults to false (matching LTX-2.3 behavior)

        stepStart = Date()
        LTXDebug.log("Applying \(vaeWeights.count) VAE weights...")
        try LTXWeightLoader.applyVAEWeights(vaeWeights, to: vaeDecoder!)
        LTXVideoProfiler.shared.end("Load Video VAE")
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
    ///
    /// Family-aware: 2.5 checkpoints load their bundled Gemma 4, 2.3 the external
    /// Gemma 3 — the same routing as `loadModels()`. Component dicts are lazily
    /// mmap'd, so resolving the checkpoint here does not read transformer bytes.
    public func loadTextEncoderModels(
        progressCallback: DownloadProgressCallback? = nil,
        gemmaModelPath: String? = nil,
        tokenizerPath: String? = nil
    ) async throws {
        LTXDebug.log("Loading text encoder models for \(model.displayName)...")
        var stepStart = Date()

        progressCallback?(DownloadProgress(progress: 0.1, message: "Resolving checkpoint..."))
        let checkpoint = try await resolveCheckpoint(progressCallback: progressCallback)
        let source = LTXCheckpointSource(model: model, paths: checkpoint)
        let textEncoderAssets = try checkpoint.textEncoder.map { try LTX25TextEncoderAssets(fileURL: $0) }

        progressCallback?(DownloadProgress(progress: 0.3, message: "Loading Gemma model..."))
        gemmaEncoder = try await loadGemmaEncoder(
            checkpoint: checkpoint,
            source: source,
            textEncoderAssets: textEncoderAssets,
            gemmaModelPath: gemmaModelPath,
            tokenizerPath: tokenizerPath,
            progressCallback: progressCallback)
        LTXDebug.log("[TIME] Gemma load: \(String(format: "%.1f", Date().timeIntervalSince(stepStart)))s")

        progressCallback?(DownloadProgress(progress: 0.7, message: "Loading connector weights..."))
        stepStart = Date()
        let connectorWeights = try source.loadComponents(textEncoderAssets: textEncoderAssets).connector

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
        let beacon = RuntimeBeacon.begin(task: "load-audio-models", model: model.rawValue)
        defer { beacon?.end() }

        // Resolve the checkpoint up front: the audio bundle path and the audio
        // VAE source both depend on it, and resolving late meant a standalone
        // loadAudioModels() call fell back to the 24 kHz legacy vocoder.
        let checkpointEarly = try await resolveCheckpoint(progressCallback: progressCallback)

        // Step 1: Download and load Audio VAE. Split checkpoints (2.5) carry
        // their own audio_vae.* tensors in the audio bundle; only unified-era
        // checkpoints use the shared Lightricks/LTX-2 file.
        progressCallback?(DownloadProgress(progress: 0.1, message: "Downloading audio VAE..."))
        let audioVAEPath: URL
        if model.weightsLayout == .split {
            audioVAEPath = checkpointEarly.audioBundle
        } else {
            audioVAEPath = try await downloader.downloadAudioVAE { progress in
                progressCallback?(progress)
            }
        }
        let audioVAEWeights = try LTXWeightLoader.loadAudioVAEWeights(from: audioVAEPath.path, includeEncoder: includeEncoder)

        LTXVideoProfiler.shared.start("Load Audio VAE")
        audioVAE = AudioVAE(includeEncoder: includeEncoder)
        try LTXWeightLoader.applyAudioVAEWeights(audioVAEWeights, to: audioVAE!)
        LTXVideoProfiler.shared.end("Load Audio VAE")
        LTXDebug.log("Audio VAE loaded from \(audioVAEPath.lastPathComponent)")

        // Step 2: Download and load Vocoder
        progressCallback?(DownloadProgress(progress: 0.4, message: "Downloading vocoder..."))
        let vocoderPath = try await downloader.downloadVocoder { progress in
            progressCallback?(progress)
        }
        vocoder = try loadVocoder(
            bundle: checkpointEarly.audioBundle, audioVAEPath: audioVAEPath, legacyPath: vocoderPath)
        LTXDebug.log("Vocoder loaded (\(vocoder!.outputSampleRate) Hz)")

        // Step 3: Create LTX2 dual transformer and load unified weights
        // The LTX2 transformer uses the same weight keys as the video-only transformer
        // plus additional audio-specific keys. We reload from the unified file.
        progressCallback?(DownloadProgress(progress: 0.6, message: "Loading dual audio/video transformer..."))

        // Same source as loadModels: on a split checkpoint the aggregate projections
        // live with the text encoder, so resolving only the transformer file here
        // would rebuild the encoder below with randomly-initialised projections.
        let checkpoint = checkpointEarly
        let source = LTXCheckpointSource(model: model, paths: checkpoint)
        let (transformerWeights, _, connectorWeightsFromUnified) =
            try source.loadComponents(includeAudio: true)

        // Create LTX2 dual transformer
        let ltx2 = LTX2Transformer(
            config: model.transformerConfig,
            ropeType: .split,
            memoryOptimization: memoryOptimization
        )

        // Apply weights with audio key mapping enabled
        LTXVideoProfiler.shared.start("Load Dual Transformer")
        try LTXWeightLoader.applyTransformerWeights(transformerWeights, to: ltx2, includeAudio: true)
        LTXVideoProfiler.shared.end("Load Dual Transformer")

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
        lipdubFusion = nil  // fresh weights from the unified file

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
    internal func packAudioLatents(_ latents: MLXArray) -> MLXArray {
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
    internal func unpackAudioLatents(_ latents: MLXArray, numFrames: Int) -> MLXArray {
        let b = latents.dim(0)
        // (B, T, C*M) -> (B, T, C, M) -> (B, C, T, M)
        let unflattened = latents.reshaped([b, numFrames, Self.audioLatentChannels, Self.audioLatentMelBins])
        return unflattened.transposed(0, 2, 1, 3)
    }

    // MARK: - Denoise step helper

    /// Result of one denoise-loop forward pass: predicted velocities for the
    /// video latent (always present) and the audio packed latent (when running
    /// dual-stream `LTX2Transformer`). Both are unpatchified and float32 — ready
    /// for the scheduler step. Velocities are already cropped back to the
    /// original token counts when their respective `*AppendCtx` was provided.
    internal struct StepVelocity {
        let video: MLXArray
        let audio: MLXArray?
        /// Velocity for the generated-keyframe slots, when the stage has any.
        /// The caller steps it with the same rule it uses for `video`.
        let slots: MLXArray?

        init(video: MLXArray, audio: MLXArray?, slots: MLXArray? = nil) {
            self.video = video
            self.audio = audio
            self.slots = slots
        }
    }

    /// One forward pass through the active transformer (video-only `LTXTransformer`
    /// or dual `LTX2Transformer`) at the given `sigma`.
    ///
    /// Two independent append paths are supported:
    /// - `videoAppendCtx` extends the video sequence (used by keyframe interpolation
    ///   and by the LipDub IC-LoRA video reference).
    /// - `audioRefCtx` extends the audio sequence (used by LipDub audio reference).
    ///
    /// Either, both, or neither may be set. When set, per-token timesteps put σ on
    /// the original tokens and 0 on the appended tokens, and the matching
    /// `precomputed*RoPE` overrides bypass the transformer's internal cache.
    /// Predicted velocities are cropped back to the original token counts before
    /// the scheduler step.
    ///
    /// Caller is responsible for the scheduler step and post-step `MLX.eval`.
    /// This split keeps the Stage 1 (Euler scheduler) and Stage 2 (manual Euler)
    /// integration code untouched, since their numerical contracts differ.
    internal func runDenoiseStep(
        sigma: Float,
        videoLatent: MLXArray,
        audioLatentPacked: MLXArray?,
        shape: VideoLatentShape,
        videoAppendCtx: AppendKeyframeContext?,
        audioRefCtx: AudioReferenceContext?,
        audioNumFrames: Int,
        videoTextEmbeddings: MLXArray,
        audioTextEmbeddings: MLXArray,
        textMask: MLXArray?,
        slotLatent: MLXArray? = nil
    ) -> StepVelocity {
        let videoPatchified = patchify(videoLatent).asType(.bfloat16)

        // --- Extend the video stream when videoAppendCtx is provided ---
        let extTokensVideo: MLXArray
        let videoTimestep: MLXArray
        var keyframeRange: Range<Int>? = nil
        if let ctx = videoAppendCtx {
            var pieces = [videoPatchified]
            if let guides = ctx.guideTokens { pieces.append(guides) }
            if let layout = ctx.slots {
                guard let slotLatent else {
                    fatalError("stage declares \(layout.slotCount) keyframe slots but no slot latent "
                        + "was passed — the slots would be denoised from nothing")
                }
                pieces.append(slotLatent.asType(.bfloat16))
                keyframeRange = layout.tokenRange
            }
            extTokensVideo = pieces.count == 1 ? videoPatchified : MLX.concatenated(pieces, axis: 1)
            videoTimestep = buildExtendedTimestep(
                sigma: sigma,
                originalCount: ctx.originalCount,
                guideCount: ctx.guideCount,
                slotCount: ctx.slots?.tokenCount ?? 0
            )
        } else {
            extTokensVideo = videoPatchified
            videoTimestep = MLXArray([sigma])
        }

        if let ltx2 = ltx2Transformer, let ap = audioLatentPacked {
            // --- Extend the audio stream when audioRefCtx is provided ---
            let audioPatchified = ap.asType(.bfloat16)
            let extTokensAudio: MLXArray
            let audioTimestep: MLXArray
            if let aCtx = audioRefCtx {
                extTokensAudio = MLX.concatenated([audioPatchified, aCtx.guideTokens], axis: 1)
                audioTimestep = buildExtendedTimestep(
                    sigma: sigma,
                    originalCount: aCtx.originalCount,
                    guideCount: aCtx.guideCount
                )
            } else {
                extTokensAudio = audioPatchified
                audioTimestep = MLXArray([sigma])
            }

            // Match Python `context_mask=None` (LipDub/keyframe/IC-LoRA paths) — the
            // dual-stream LTX2 model was trained without masking padding tokens.
            // textMask intentionally unused; kept in the signature for symmetry.
            _ = textMask
            let (videoVelExt, audioVelExt) = ltx2(
                videoLatent: extTokensVideo,
                audioLatent: extTokensAudio,
                videoContext: videoTextEmbeddings.asType(.bfloat16),
                audioContext: audioTextEmbeddings.asType(.bfloat16),
                videoTimesteps: videoTimestep,
                audioTimesteps: audioTimestep,
                videoContextMask: nil,
                audioContextMask: nil,
                videoLatentShape: (frames: shape.frames, height: shape.height, width: shape.width),
                audioNumFrames: audioNumFrames,
                precomputedVideoRoPE: videoAppendCtx?.extRoPE,
                precomputedCrossVideoRoPE: videoAppendCtx?.extCrossVideoRoPE,
                precomputedAudioRoPE: audioRefCtx?.extRoPE,
                precomputedCrossAudioRoPE: audioRefCtx?.extCrossRoPE,
                keyframeTokenRange: keyframeRange
            )
            let videoVel = videoAppendCtx
                .map { cropToOriginal(velocity: videoVelExt, originalCount: $0.originalCount) }
                ?? videoVelExt
            let audioVel = audioRefCtx
                .map { cropToOriginal(velocity: audioVelExt, originalCount: $0.originalCount) }
                ?? audioVelExt
            return StepVelocity(
                video: unpatchify(videoVel, shape: shape).asType(.float32),
                audio: audioVel.asType(.float32),
                slots: sliceSlotVelocity(videoVelExt, layout: videoAppendCtx?.slots)
            )
        } else if let videoTransformer = transformer {
            // Audio reference is meaningless without ltx2Transformer — caller's bug if set.
            assert(audioRefCtx == nil, "audioRefCtx requires LTX2Transformer (audio enabled)")
            let velExt = videoTransformer(
                latent: extTokensVideo,
                context: videoTextEmbeddings.asType(.bfloat16),
                timesteps: videoTimestep,
                contextMask: nil,
                latentShape: (frames: shape.frames, height: shape.height, width: shape.width),
                precomputedRoPE: videoAppendCtx?.extRoPE,
                keyframeTokenRange: keyframeRange
            )
            let vel = videoAppendCtx
                .map { cropToOriginal(velocity: velExt, originalCount: $0.originalCount) }
                ?? velExt
            return StepVelocity(
                video: unpatchify(vel, shape: shape).asType(.float32),
                audio: nil,
                slots: sliceSlotVelocity(velExt, layout: videoAppendCtx?.slots)
            )
        } else {
            fatalError("runDenoiseStep called without any transformer loaded — guarded by generateVideo")
        }
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
        keyframeSlots: [Int] = [],
        slotsOutputPath: String? = nil
    ) async throws -> VideoGenerationResult {
        try config.validate()

        let beacon = RuntimeBeacon.begin(task: "generate", model: model.rawValue)
        defer { beacon?.end() }
        let userProgress = onProgress
        let onProgress: GenerationProgressCallback?
        if let beacon {
            onProgress = { progress in
                beacon.update(
                    phase: progress.phase.rawValue,
                    step: progress.currentStep + 1,
                    totalSteps: progress.totalSteps
                )
                userProgress?(progress)
            }
        } else {
            onProgress = userProgress
        }

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

        try ensureNoLipDubLoRAFused(wouldCorrupt: "regular generation")

        let generationStart = Date()

        // Two-stage requires width/height divisible by 64
        guard config.width % 64 == 0 && config.height % 64 == 0 else {
            throw LTXError.invalidConfiguration("Two-stage requires width and height divisible by 64. Got \(config.width)x\(config.height)")
        }

        // The two-stage pipeline is always distilled: stage 1 runs the fixed
        // trained 9-value sigma schedule (8 steps) and stage 2 the fixed 4-value
        // refinement schedule. Custom step counts are not honored here (they
        // produce artifacts on a distilled model — issue #33); reject explicitly
        // instead of silently ignoring config.numSteps. Configurable steps are
        // available on the dev retake path.
        guard config.numSteps == LTXModel.distilled.defaultSteps else {
            throw LTXError.invalidConfiguration(
                "generateVideo always runs the two-stage distilled schedule " +
                "(fixed \(LTXModel.distilled.defaultSteps)+3 steps); numSteps=\(config.numSteps) " +
                "cannot be honored. Use generateRetake with the dev model for configurable steps."
            )
        }

        let halfWidth = config.width / 2
        let halfHeight = config.height / 2

        // Resolve keyframes: explicit list takes priority, --image is sugar for [(image, 0)]
        let effectiveKeyframes: [KeyframeInput]
        if !config.keyframes.isEmpty {
            effectiveKeyframes = config.keyframes
        } else if let imagePath = config.imagePath {
            effectiveKeyframes = [KeyframeInput(path: imagePath, pixelFrameIndex: 0)]
        } else {
            effectiveKeyframes = []
        }
        let isI2V = !effectiveKeyframes.isEmpty

        LTXDebug.log("Two-stage generation: \(halfWidth)x\(halfHeight) → \(config.width)x\(config.height), audio=\(hasAudio), keyframes=\(effectiveKeyframes.count)")

        // 0. Encode keyframes at half-res
        var halfResKeyframes: [EncodedKeyframe] = []
        if isI2V {
            LTXDebug.log("Two-stage: encoding \(effectiveKeyframes.count) keyframe(s) at \(halfWidth)x\(halfHeight)")
            halfResKeyframes = try await encodeKeyframes(effectiveKeyframes, width: halfWidth, height: halfHeight)
            unloadVAEEncoder()
        }

        // 0b. Optionally enhance prompt (uses first keyframe's image when available)
        let effectivePrompt: String
        if config.enhancePrompt {
            let promptImage = effectiveKeyframes.first?.path
            LTXDebug.log("Enhancing prompt with VLM (\(promptImage != nil ? "I2V" : "T2V"))...")
            effectivePrompt = try await enhancePromptWithVLM(prompt, imagePath: promptImage)
        } else {
            effectivePrompt = prompt
        }

        // 1. Text encoding
        let profiler = LTXVideoProfiler.shared
        profiler.start("Text Encoding")

        profiler.start("Gemma Forward")
        let (states, attentionMask) = try encodeHiddenStates(effectivePrompt)
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

        unloadGemmaIfConfigured()

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

        // Keyframe conditioning (issue #21 fix): appended guide tokens with shifted RoPE
        // positions, matching Lightricks `VideoConditionByKeyframeIndex` semantics.
        //
        // The keyframe latents stay separate from the main video sequence; they're
        // concatenated as extra tokens with their own temporal/spatial positions and
        // σ=0 timestep. The model "sees" them via attention, and we crop the predicted
        // velocity to the original token count before the scheduler step. This is the
        // only keyframe code path — the previous "inject into latent slot" approach
        // produced grainy artifacts at non-zero keyframe positions (issue #21) because
        // a 1-frame VAE encoding placed at a slot representing 8 pixel frames is
        // structurally incompatible with what the decoder expects.
        MLX.eval(videoLatent)

        // Pre-build guide tokens, extended positions, and RoPE for the append path.
        // These are constant across denoising steps, so compute once. Returns nil
        // when there are no keyframes — `runDenoiseStep` handles both paths.
        // Generated keyframe slots (LTX-2.5 only): extra denoised tokens at chosen
        // pixel frames. They cost K × H × W tokens and come back as full-quality
        // single frames that later stages and later temporal tiles anchor on.
        if !keyframeSlots.isEmpty {
            // Checked before anything expensive happens, the way upstream checks it:
            // the answer comes from the checkpoint config alone, and a slot denoised
            // without the marker is off-distribution rather than merely unmarked.
            guard model.transformerConfig.keyframesAbsPosEmbedding else {
                throw LTXError.invalidConfiguration(
                    "\(model.rawValue) has no keyframe absolute-position embedding, so it cannot "
                    + "generate keyframe slots. Use an LTX-2.5 checkpoint.")
            }
        }
        let slotIndices = keyframeSlots.isEmpty
            ? [] : try validatedSlotIndices(keyframeSlots, numFrames: config.numFrames)
        let stage1AppendCtx: AppendKeyframeContext? = prepareKeyframeAppend(
            encoded: halfResKeyframes,
            shape: stage1Shape,
            hasAudio: hasAudio,
            refConfig: transformer?.config ?? ltx2Transformer?.config ?? .default,
            stageLabel: "Stage 1",
            slotIndices: slotIndices
        )

        // Slots start as pure noise scaled by σ₀, exactly like the video latent:
        // upstream's `denoise_mask = 1` means the noiser ignores any clean content
        // and treats the slot as something to generate from scratch.
        var slotLatent: MLXArray? = stage1AppendCtx?.slots.map { layout in
            MLXRandom.normal([1, layout.tokenCount, stage1Shape.channels])
                .asType(.float32) * stage1Sigmas[0]
        }
        if let s = slotLatent { MLX.eval(s) }

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

            // One forward pass through the active transformer. Stage 1 uses the Euler
            // scheduler for the video latent and a manual flow-matching update for audio.
            let vel = runDenoiseStep(
                sigma: sigma,
                videoLatent: videoLatent,
                audioLatentPacked: audioLatentPacked,
                shape: stage1Shape,
                videoAppendCtx: stage1AppendCtx,
                audioRefCtx: nil,
                audioNumFrames: audioNumFrames,
                videoTextEmbeddings: videoTextEmbeddings,
                audioTextEmbeddings: audioTextEmbeddings,
                textMask: textMask,
                slotLatent: slotLatent
            )
            videoLatent = stage1Scheduler.step(
                latent: videoLatent, velocity: vel.video,
                sigma: sigma, sigmaNext: sigmaNext
            )
            if let sv = vel.slots, let current = slotLatent {
                // Plain Euler on the same schedule. The scheduler's token-count shift
                // is already folded into the sigmas the two streams share.
                let stepped = current + MLXArray(sigmaNext - sigma) * sv
                slotLatent = stepped
                MLX.eval(stepped)
            }
            if let av = vel.audio, let ap = audioLatentPacked {
                let updatedAudio = ap + (sigmaNext - sigma) * av
                audioLatentPacked = updatedAudio
                MLX.eval(videoLatent, updatedAudio)
            } else {
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

        // The slots ride the same upscale so stage 2 can start from them: they are
        // latent frames like any other, just one pixel frame wide.
        var upscaledSlots: MLXArray? = nil
        if let layout = stage1AppendCtx?.slots, let tokens = slotLatent {
            let packed = GeneratedKeyframeSlots.unpack(
                tokens: tokens, layout: layout, shape: stage1Shape)
            let denormed = packed * std5d + mean5d
            let up = upscaler(denormed)
            upscaledSlots = (up - mean5d) / std5d
            MLX.eval(upscaledSlots!)
            LTXDebug.log("[slots] upscaled \(layout.slotCount) keyframe(s) to \(upscaledSlots!.shape)")
        }

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

        // I2V stage 2: encode keyframes at full resolution
        var fullResKeyframes: [EncodedKeyframe] = []
        if isI2V {
            LTXDebug.log("Stage 2: encoding \(effectiveKeyframes.count) keyframe(s) at \(config.width)x\(config.height)")
            fullResKeyframes = try await encodeKeyframes(effectiveKeyframes, width: config.width, height: config.height)
            unloadVAEEncoder()
        }
        MLX.eval(videoLatent)
        if hasAudio, let ap = audioLatentPacked { MLX.eval(ap) }

        // Pre-build guide tokens, extended positions, and RoPE for the Stage 2 append path.
        let stage2AppendCtx: AppendKeyframeContext? = prepareKeyframeAppend(
            encoded: fullResKeyframes,
            shape: stage2Shape,
            hasAudio: hasAudio,
            refConfig: transformer?.config ?? ltx2Transformer?.config ?? .default,
            stageLabel: "Stage 2",
            slotIndices: slotIndices,
            slotInitial: upscaledSlots
        )

        // Stage 2 re-noises the slots to the same level as the video, so they are
        // refined at full resolution rather than carried over untouched.
        if let layout = stage2AppendCtx?.slots, let initial = stage2AppendCtx?.slotInitialTokens {
            let slotNoise = MLXRandom.normal([1, layout.tokenCount, stage2Shape.channels])
                .asType(.float32)
            let renoised = MLXArray(noiseScale) * slotNoise
                + MLXArray(1.0 - noiseScale) * initial.asType(.float32)
            slotLatent = renoised
            MLX.eval(renoised)
        }

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

            // One forward pass through the active transformer. Stage 2 uses a manual
            // Euler step (not the Stage 1 scheduler) for both video and audio.
            let vel = runDenoiseStep(
                sigma: sigma,
                videoLatent: videoLatent,
                audioLatentPacked: audioLatentPacked,
                shape: stage2Shape,
                videoAppendCtx: stage2AppendCtx,
                audioRefCtx: nil,
                audioNumFrames: audioNumFrames,
                videoTextEmbeddings: videoTextEmbeddings,
                audioTextEmbeddings: audioTextEmbeddings,
                textMask: textMask,
                slotLatent: slotLatent
            )
            let dt = sigmaNext - sigma
            videoLatent = videoLatent + MLXArray(dt) * vel.video
            if let sv = vel.slots, let current = slotLatent {
                let stepped = current + MLXArray(dt) * sv
                slotLatent = stepped
                MLX.eval(stepped)
            }
            if let av = vel.audio, let ap = audioLatentPacked {
                let updatedAudio = ap + MLXArray(dt) * av
                audioLatentPacked = updatedAudio
                MLX.eval(videoLatent, updatedAudio)
            } else {
                MLX.eval(videoLatent)
            }

            let stepDur2 = Date().timeIntervalSince(stepStart)
            profiler.recordStep(duration: stepDur2)

            LTXDebug.log("Stage 2 step \(step)/\(stage2NumSteps): σ=\(String(format: "%.4f", sigma))→\(String(format: "%.4f", sigmaNext)), time=\(String(format: "%.1f", stepDur2))s")
        }
        LTXDebug.log("Stage 2 complete: \(String(format: "%.1f", Date().timeIntervalSince(stage2Start)))s")
        profiler.end("Denoising Stage 2")

        // Hand the generated keyframes back as latents. They are what a later
        // temporal round anchors on, and re-encoding them from decoded pixels
        // would put a VAE round trip between the anchor and what produced it.
        if let path = slotsOutputPath, let layout = stage2AppendCtx?.slots,
           let tokens = slotLatent {
            let keyframes = GeneratedKeyframeSlots.unpack(
                tokens: tokens, layout: layout, shape: stage2Shape)
            MLX.eval(keyframes)
            try MLX.save(
                arrays: [
                    "keyframes": keyframes,
                    "pixel_frame_indices": MLXArray(layout.pixelFrameIndices.map { Int32($0) }),
                ],
                url: URL(fileURLWithPath: path))
            LTXDebug.log("[slots] wrote \(layout.slotCount) generated keyframe(s) to \(path)")
        }

        // Unload transformer
        if memoryOptimization.unloadAfterUse {
            self.ltx2Transformer = nil
            self.transformer = nil
            self.lipdubFusion = nil
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
        let videoTensor = decodeFrames(latent: videoLatent)
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
            profiler.start("Audio Decode")
            let audioLatentUnpacked = unpackAudioLatents(ap, numFrames: audioNumFrames)
            let waveform = decodeAudio(
                latents: audioLatentUnpacked,
                audioVAE: audioVAE,
                vocoder: vocoder
            )
            MLX.eval(waveform)
            profiler.end("Audio Decode")
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

        let beacon = RuntimeBeacon.begin(task: "retake", model: model.rawValue)
        defer { beacon?.end() }
        let userProgress = onProgress
        let onProgress: GenerationProgressCallback?
        if let beacon {
            onProgress = { progress in
                beacon.update(
                    phase: progress.phase.rawValue,
                    step: progress.currentStep + 1,
                    totalSteps: progress.totalSteps
                )
                userProgress?(progress)
            }
        } else {
            onProgress = userProgress
        }

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

        try ensureNoLipDubLoRAFused(wouldCorrupt: "retake generation")

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
        // loadAudio returns (samples,) for mono or (channels, samples) for stereo —
        // sample count is the LAST axis in either case.
        let sourceSampleCount = sourceWaveform.dim(sourceWaveform.ndim - 1)
        if sourceSampleCount > 0 {
            sourceAudioWaveform = sourceWaveform
            LTXDebug.log("Source audio extracted: \(sourceSampleCount) samples × \(sourceWaveform.ndim == 1 ? 1 : sourceWaveform.dim(0)) ch (\(String(format: "%.1f", Float(sourceSampleCount) / Float(Self.audioSampleRate)))s)")
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
        let (states, attentionMask) = try encodeHiddenStates(effectivePrompt)

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

        unloadGemmaIfConfigured()

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

        // Determine model mode: dev (CFG + STG, configurable steps) or distilled
        // (8 steps, no guidance). Step count is only configurable on the dev
        // path: the non-distilled scheduler computes a proper token-shifted
        // sigma schedule for any count, whereas the distilled model was trained
        // to jump the fixed 9-value sigma schedule — arbitrary counts there
        // produce artifacts (issue #33).
        let useDevModel = (model == .dev)
        if !useDevModel && config.numSteps != LTXModel.distilled.defaultSteps {
            throw LTXError.invalidConfiguration(
                "The distilled model runs a fixed \(LTXModel.distilled.defaultSteps)-step " +
                "trained sigma schedule; custom step counts (got \(config.numSteps)) produce " +
                "artifacts. Use the dev model for configurable steps."
            )
        }
        let retakeSteps = useDevModel ? config.numSteps : 8
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
            // Re-load Gemma for negative prompt encoding — only when it was
            // actually unloaded above (with unloadAfterUse == false it is still
            // resident, and an unconditional loadModels() would rebuild the full
            // Gemma + transformer + VAE stack mid-run on top of the live one).
            if gemmaEncoder == nil {
                try await loadModels(progressCallback: nil)
            }
            // The official DEFAULT_NEGATIVE_PROMPT — the CFG direction is part of
            // the trained contract (docs/knowledge: empty-cfg-negative pitfall).
            let (negStates, negAttentionMask) = try encodeHiddenStates(Self.defaultNegativePrompt)
            let negEncoderOutput = try textEncoder.encodeFromHiddenStates(
                hiddenStates: negStates,
                attentionMask: negAttentionMask,
                paddingSide: "left"
            )
            negVideoTextEmbeddings = negEncoderOutput.videoEncoding
            negAudioTextEmbeddings = negEncoderOutput.audioEncoding
            MLX.eval(negVideoTextEmbeddings!)
            if let nae = negAudioTextEmbeddings { MLX.eval(nae) }

            unloadGemmaIfConfigured()
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

            // CFG negative pass (dev model only)
            var negDenoised: MLXArray? = nil
            if cfgScale != 1.0, let negCtx = negVideoTextEmbeddings {
                let negAudioCtx = (negAudioTextEmbeddings ?? audioCtx).asType(.bfloat16)
                negDenoised = try runTransformer(context: negCtx, audioContext: negAudioCtx)
            }

            // STG: perturbed pass with self-attention skipped on stgBlocks (dev model only)
            var stgDenoised: MLXArray? = nil
            if stgScale != 0.0 && !stgBlocks.isEmpty {
                if let ltx2 = ltx2Transformer { ltx2.setSTGBlocks(stgBlocks) }
                if let t = transformer { t.setSTGBlocks(stgBlocks) }
                stgDenoised = try runTransformer(context: videoTextEmbeddings, audioContext: audioCtx)
                if let ltx2 = ltx2Transformer { ltx2.clearSTG() }
                if let t = transformer { t.clearSTG() }
            }

            var denoisedVideo = Self.combineGuidance(
                cond: condDenoised, neg: negDenoised, stg: stgDenoised,
                cfgScale: cfgScale, stgScale: stgScale, guidanceRescale: guidanceRescale)

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
            self.lipdubFusion = nil
            Memory.clearCache()
            LTXDebug.log("Transformer unloaded")
        }

        // Phase 3: Decode
        onProgress?(GenerationProgress(
            currentStep: numSteps, totalSteps: numSteps, sigma: 0, phase: .decoding
        ))
        LTXMemoryManager.setPhase(.vaeDecode)
        profiler.start("VAE Decode")
        let videoTensor = decodeFrames(latent: videoLatent)
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

    // MARK: - LipDub

    /// Generate a lip-synced video from a reference video + prompt using the LipDub IC-LoRA.
    ///
    /// Two-stage distilled pipeline that conditions BOTH streams on the reference:
    /// - **Video stream** is conditioned on the reference video frames via the IC-LoRA
    ///   (`buildVideoReference`, downscaled per `reference_downscale_factor` from the
    ///   LoRA's safetensors metadata).
    /// - **Audio stream** is conditioned on the reference audio via appended audio tokens
    ///   with negative RoPE positions (`buildAudioReference`).
    ///
    /// Stage 2 reuses the Stage 1 audio output as the new audio reference and keeps the
    /// audio latent FROZEN (no further denoising — Python `lipdub.py` `frozen=True`).
    /// The decoded audio is the Stage 1 denoised result (matches Python `lipdub.py` line 264).
    ///
    /// - Parameters:
    ///   - prompt: Text description of the desired output (typically describing what is being said).
    ///   - referenceVideoPath: Path to the source video file (.mp4) — both video frames and
    ///     (by default) audio track are extracted from this file. Provide either this OR
    ///     `referenceImagePath`, not both.
    ///   - continuationTailPath: Segment chaining (issue #35, image mode only):
    ///     path to the **previous segment's video**. Its last 9 frames are read
    ///     natively (no clip preparation, no external tool) and their last latent
    ///     frame replaces the still image as the frame-0 anchor, preserving
    ///     position AND velocity across the cut. A pre-trimmed 9-frame clip is
    ///     also accepted (its last 9 frames are all of them). Requires
    ///     `targetAudioPath`; combinable with `referenceImagePath` (then only
    ///     used for prompt enhancement). First output frame duplicates the
    ///     anchor — trim one frame when concatenating.
    ///   - referenceImagePath: Path to a still image (.png/.jpg) used as the identity
    ///     anchor. Internally encoded as a single-frame I2V keyframe at pixel index 0
    ///     (same code path as `generate --image`), NOT as a multi-frame IC-LoRA video
    ///     reference. This frees the rest of the timeline to respond to the prompt while
    ///     the LipDub LoRA + audio reference still drive lip-sync. When set,
    ///     `targetAudioPath` is REQUIRED (the image has no audio) and the speech-window
    ///     alignment step is skipped — the target audio is used directly.
    ///   - lipdubLoraPath: Path to the LipDub IC-LoRA safetensors file.
    ///   - config: Width / height / seed / etc. `numFrames` **must** be `8n+1`
    ///     already (no automatic snap is performed here — the CLI enforces the
    ///     constraint up front, and library callers are expected to do the same).
    ///   - upscalerWeightsPath: Path to spatial upscaler safetensors (used between stages).
    ///   - targetAudioPath: Optional path to a separate audio file (e.g. TTS in a
    ///     different language) to lip-sync to. When set (and `referenceVideoPath` is
    ///     used), the audio is loaded as mono, its speech-active window is detected, and
    ///     it is time-stretched (pitch preserved) so the speech occupies the same window
    ///     as the source video's speech. The aligned waveform replaces the source video's
    ///     audio as the LipDub reference. When nil and using a video reference, the
    ///     source video's audio is used as-is. REQUIRED when using `referenceImagePath`.
    ///   - onProgress: Optional progress callback.
    /// - Returns: `VideoGenerationResult` with the generated video frames and the decoded
    ///   audio waveform.
    ///
    /// > NOTE: `reference_strength` (Python `VideoConditionByReferenceLatent.strength`,
    /// > default 1.0) is hard-coded to 1.0 here — the reference is fully clean
    /// > (`denoise_mask = 0`). Partial-strength reference conditioning would require
    /// > adding per-token denoise-mask blending in `runDenoiseStep`; not implemented.
    ///
    /// ## Consecutive runs (segmentation)
    /// The IC-LoRA is fused destructively (no pristine weights are kept for the 22B
    /// transformer). Consecutive calls with the **same** `lipdubLoraPath` and the
    /// same `lipdubLoRAScale` reuse the fused transformer without re-fusing or
    /// reloading — provided the transformer survives between runs, i.e.
    /// `memoryOptimization.unloadAfterUse == false`
    /// (use ``MemoryOptimizationConfig/disabled``). Switching to a different LoRA
    /// or scale, or calling `generateVideo`/`generateRetake` while fused, throws
    /// until `loadModels()` + `loadAudioModels()` restore pristine weights. Check
    /// ``fusedLipDubLoRAPath`` / ``fusedLipDubLoRAScale`` to know the current state.
    ///
    /// - Parameter lipdubLoRAScale: Multiplier on the IC-LoRA delta
    ///   (`W' = W + scale · B·A`). **Leave at 1.0 unless experimenting**: this is
    ///   an in-context LoRA, not a style LoRA — it teaches the transformer how to
    ///   read the appended reference tokens (audio at negative positions, video
    ///   reference), so scaling it down weakens the conditioning mechanism itself
    ///   rather than just softening an effect. Lightricks publishes it for use at
    ///   1.0. Values outside 0.5…1.5 log a warning; ≤ 0 throws.
    public func generateLipDub(
        prompt: String,
        referenceVideoPath: String? = nil,
        referenceImagePath: String? = nil,
        continuationTailPath: String? = nil,
        lipdubLoraPath: String,
        lipdubLoRAScale: Float = 1.0,
        config: LTXVideoGenerationConfig,
        upscalerWeightsPath: String,
        targetAudioPath: String? = nil,
        enhancePrompt: Bool = false,
        onProgress: GenerationProgressCallback? = nil
    ) async throws -> VideoGenerationResult {
        try config.validate()

        let beacon = RuntimeBeacon.begin(task: "lipdub", model: model.rawValue)
        defer { beacon?.end() }
        let userProgress = onProgress
        let onProgress: GenerationProgressCallback?
        if let beacon {
            onProgress = { progress in
                beacon.update(
                    phase: progress.phase.rawValue,
                    step: progress.currentStep + 1,
                    totalSteps: progress.totalSteps
                )
                userProgress?(progress)
            }
        } else {
            onProgress = userProgress
        }

        guard config.width % 64 == 0 && config.height % 64 == 0 else {
            throw LTXError.invalidConfiguration("LipDub requires width and height divisible by 64. Got \(config.width)x\(config.height)")
        }
        // LipDub runs the fixed distilled two-stage schedules (see generateVideo).
        guard config.numSteps == LTXModel.distilled.defaultSteps else {
            throw LTXError.invalidConfiguration(
                "generateLipDub always runs the fixed distilled schedules; " +
                "numSteps=\(config.numSteps) cannot be honored."
            )
        }
        guard let textEncoder = textEncoder,
              let vaeDecoder = vaeDecoder,
              let audioVAE = audioVAE,
              let vocoder = vocoder else {
            throw LTXError.modelNotLoaded("LipDub requires text encoder + VAE + AudioVAE + vocoder. Call loadModels() then loadAudioModels().")
        }
        guard let ltx2 = ltx2Transformer else {
            throw LTXError.modelNotLoaded("LipDub requires the dual-stream LTX2Transformer (audio enabled). Call loadAudioModels().")
        }
        // Exactly one of referenceVideoPath / referenceImagePath must be set.
        // After this block:
        //   - `isImageMode == true`  iff `referenceImagePath != nil`
        //   - `videoRefPath`         is the non-nil video path in video mode, or
        //                            an empty sentinel in image mode (never read).
        // The trailing helpers below assume these two invariants — use
        // `videoRefPath` directly inside `!isImageMode` branches instead of
        // re-unwrapping `referenceVideoPath`.
        // Continuation (issue #35) is an image-mode variant: the frame-0 anchor
        // comes from the PREVIOUS segment's tail clip instead of the still image,
        // so chained segments join with position AND motion continuity. Video
        // mode never needs it — its continuity comes from the source video.
        if let tailPath = continuationTailPath {
            guard referenceVideoPath == nil else {
                throw LTXError.invalidConfiguration(
                    "continuationTailPath is for image-mode segment chaining; video mode " +
                    "gets continuity from the source — segment the reference video instead.")
            }
            guard FileManager.default.fileExists(atPath: tailPath) else {
                throw LTXError.fileNotFound("Continuation tail clip not found: \(tailPath)")
            }
            guard targetAudioPath != nil else {
                throw LTXError.invalidConfiguration(
                    "continuationTailPath requires targetAudioPath (the tail clip anchors " +
                    "the visuals; the audio must come from the target track).")
            }
        }
        let isImageMode: Bool
        let videoRefPath: String
        switch (referenceVideoPath, referenceImagePath) {
        case (nil, nil):
            guard continuationTailPath != nil else {
                throw LTXError.invalidConfiguration("LipDub requires referenceVideoPath, referenceImagePath, or continuationTailPath.")
            }
            videoRefPath = ""  // sentinel — image mode never reads videoRefPath
            isImageMode = true
        case (.some, .some):
            throw LTXError.invalidConfiguration("LipDub: pass referenceVideoPath OR referenceImagePath, not both.")
        case (.some(let vp), nil):
            guard FileManager.default.fileExists(atPath: vp) else {
                throw LTXError.fileNotFound("Reference video not found: \(vp)")
            }
            videoRefPath = vp
            isImageMode = false
        case (nil, .some(let ip)):
            guard FileManager.default.fileExists(atPath: ip) else {
                throw LTXError.fileNotFound("Reference image not found: \(ip)")
            }
            guard targetAudioPath != nil else {
                throw LTXError.invalidConfiguration("LipDub image mode requires targetAudioPath (the image has no audio track).")
            }
            videoRefPath = ""  // sentinel — image mode never reads videoRefPath
            isImageMode = true
        }
        guard FileManager.default.fileExists(atPath: lipdubLoraPath) else {
            throw LTXError.fileNotFound("LipDub LoRA not found: \(lipdubLoraPath)")
        }
        if let warning = try Self.validateLipDubLoRAScale(lipdubLoRAScale) { print(warning) }
        if let targetAudioPath = targetAudioPath {
            guard FileManager.default.fileExists(atPath: targetAudioPath) else {
                throw LTXError.fileNotFound("Target audio not found: \(targetAudioPath)")
            }
        }

        let generationStart = Date()
        let halfWidth = config.width / 2
        let halfHeight = config.height / 2

        // 0. Optional VLM-based prompt enhancement (image mode only — VLM needs an image).
        // Done BEFORE LoRA fusion so the ~7.5 GB VLM container doesn't stack on top of
        // the LoRA-fused dual-stream transformer in resident memory. The VLM is unloaded
        // inside `enhancePromptWithVLM` via Memory.clearCache before this returns.
        // The I2V system prompt instructs the VLM to preserve the user's quoted dialogue
        // verbatim; if the speaking-verb wrapper is nonetheless dropped from the output,
        // `applyLipDubSignatureFallback` re-appends the original tail so the LipDub LoRA
        // still sees its trained trigger.
        let effectivePrompt: String
        if enhancePrompt, let imagePath = referenceImagePath {
            print("[lipdub] enhancing prompt via VLM (analyzing reference image)...")
            let enhanced = try await enhancePromptWithVLM(prompt, imagePath: imagePath)
            let (final, reappended) = Self.applyLipDubSignatureFallback(
                enhanced: enhanced, original: prompt
            )
            if let reappended = reappended {
                print("[lipdub] VLM dropped speaking/dialogue hint — re-appended: \(reappended)")
            }
            effectivePrompt = final
            print("[lipdub] enhanced prompt: \(effectivePrompt)")
        } else {
            effectivePrompt = prompt
        }

        // 1. Read LoRA metadata and fuse into the dual-stream transformer.
        let downscaleFactor = LoRALoader.referenceDownscaleFactor(from: lipdubLoraPath)
        LTXDebug.log("[lipdub] reference_downscale_factor=\(downscaleFactor) from LoRA metadata")
        // Downscale factor only matters for the video-reference path; in image mode
        // we use the I2V keyframe-append pattern at the full target resolution.
        if !isImageMode {
            guard halfWidth % downscaleFactor == 0 && halfHeight % downscaleFactor == 0 else {
                throw LTXError.invalidConfiguration(
                    "Half-resolution \(halfWidth)x\(halfHeight) must be divisible by downscale_factor=\(downscaleFactor)"
                )
            }
        }
        // [DIAGNOSTIC] LTX_LIPDUB_SKIP_LORA=1 bypasses LoRA fusion to isolate IC-LoRA contribution.
        let skipLoRA = ProcessInfo.processInfo.environment["LTX_LIPDUB_SKIP_LORA"] == "1"
        let canonicalLoRA = Self.canonicalLoRAPath(lipdubLoraPath)
        if skipLoRA {
            // The diagnostic is only meaningful on pristine weights: a transformer
            // still fused from a previous run would silently compare fused vs fused.
            try ensureNoLipDubLoRAFused(wouldCorrupt: "the LTX_LIPDUB_SKIP_LORA A/B diagnostic")
            print("[lipdub][DIAG] LTX_LIPDUB_SKIP_LORA=1 — skipping LoRA fusion")
        } else if let fused = lipdubFusion {
            // Consecutive runs with the same LoRA (e.g. app-side segmentation of a
            // long dialogue) reuse the fused transformer — re-fusing would apply
            // the delta twice, and reloading the 22B per segment is needless.
            guard fused.path == canonicalLoRA else {
                throw LTXError.invalidConfiguration(
                    "A different LipDub LoRA is already fused into the transformer " +
                    "(\(fused.path)). No pristine weights are kept for the 22B model, " +
                    "so switching LoRA requires reloading: call loadModels() + loadAudioModels() first."
                )
            }
            guard fused.modificationDate == Self.loraModificationDate(canonicalLoRA) else {
                throw LTXError.invalidConfiguration(
                    "The LipDub LoRA file changed on disk since it was fused " +
                    "(\(fused.path)). Reusing the fused transformer would run stale " +
                    "weights: call loadModels() + loadAudioModels() to re-fuse."
                )
            }
            // Same file at a different scale is a different set of weights, and the
            // fusion keeps no originals to rescale from.
            guard fused.scale == lipdubLoRAScale else {
                throw LTXError.invalidConfiguration(
                    "The LipDub LoRA is already fused at scale \(fused.scale), but this " +
                    "call asks for \(lipdubLoRAScale). The fusion is destructive (no " +
                    "pristine weights kept for the 22B model), so changing the scale " +
                    "requires reloading: call loadModels() + loadAudioModels() first."
                )
            }
            print("[lipdub] LoRA already fused (same file, scale \(fused.scale)) — reusing fused transformer")
        } else {
            // A generic LoRA fused via fuseLoRA() would be baked under the IC-LoRA
            // delta, and a later unfuseLoRA() would partially wipe it — refuse the
            // ambiguous stack instead of corrupting silently.
            guard loraOriginalWeights == nil else {
                throw LTXError.invalidConfiguration(
                    "A LoRA is already fused via fuseLoRA() (\(loraFusedPath ?? "?")). " +
                    "Call unfuseLoRA() before generateLipDub, or reload the models."
                )
            }
            LTXDebug.log("[lipdub] fusing LipDub IC-LoRA into LTX2Transformer...")
            let loraDebug = ProcessInfo.processInfo.environment["LTX_LIPDUB_LORA_DEBUG"] == "1"
            let prevDebug = LTXDebug.isEnabled
            defer { LTXDebug.isEnabled = prevDebug }
            if loraDebug { LTXDebug.enableDebugMode() }
            // Record the mtime BEFORE fusing so an overwrite racing the fusion is
            // detected as "changed" on the next run rather than missed.
            let fusedMtime = Self.loraModificationDate(canonicalLoRA)
            let (_, fuseResult) = try ltx2.fuseLoRA(from: lipdubLoraPath, scale: lipdubLoRAScale)
            let coverage = fuseResult.totalLayerCount > 0
                ? Float(fuseResult.modifiedLayerCount) / Float(fuseResult.totalLayerCount) * 100.0
                : 0
            print("[lipdub] LoRA fused: \(fuseResult.modifiedLayerCount) / \(fuseResult.totalLayerCount) layer-pairs (\(String(format: "%.1f", coverage))%) at scale \(lipdubLoRAScale) — \(fuseResult.loraName)")
            eval(ltx2.parameters())
            Memory.clearCache()
            lipdubFusion = LipDubFusionRecord(
                path: canonicalLoRA, modificationDate: fusedMtime, scale: lipdubLoRAScale)
        }

        // 2. Snap target frame count to 8k+1 based on the reference video.
        // We respect config.numFrames as the upper bound but never exceed the ref video.
        let numFrames = config.numFrames  // CLI is responsible for snapping; trust the caller
        let stage1Shape = VideoLatentShape.fromPixelDimensions(
            batch: 1, channels: 128,
            frames: numFrames, height: halfHeight, width: halfWidth
        )
        let stage2Shape = VideoLatentShape.fromPixelDimensions(
            batch: 1, channels: 128,
            frames: numFrames, height: config.height, width: config.width
        )
        let audioNumFrames = computeAudioLatentFrames(videoFrames: numFrames)
        LTXDebug.log("[lipdub] target frames=\(numFrames), Stage1 latent=\(stage1Shape.frames)x\(stage1Shape.height)x\(stage1Shape.width), audio frames=\(audioNumFrames)")

        // 3. Text encode the prompt.
        let (states, attentionMask) = try encodeHiddenStates(effectivePrompt)
        let encoderOutput = try textEncoder.encodeFromHiddenStates(
            hiddenStates: states,
            attentionMask: attentionMask,
            paddingSide: "left"
        )
        let videoTextEmbeddings = encoderOutput.videoEncoding
        let audioTextEmbeddings = encoderOutput.audioEncoding ?? videoTextEmbeddings
        let textMask = encoderOutput.attentionMask
        MLX.eval(videoTextEmbeddings, audioTextEmbeddings, textMask)
        // Frees ~7.5 GB before the dual-stream denoising loops; kept resident
        // with .disabled so consecutive LipDub segments can re-encode their
        // prompt — the whole point of the fused-LoRA reuse.
        unloadGemmaIfConfigured()

        // 4. Stage 1 reference: VIDEO MODE uses the IC-LoRA multi-frame reference
        // at downscaled resolution; IMAGE MODE uses the I2V keyframe-append pattern
        // (single VAE-encoded frame at index 0, full target resolution). The
        // keyframe pattern doesn't pin every output frame to the reference, so the
        // model is free to animate motion from the prompt while the LipDub LoRA +
        // audio reference still drive lip-sync.
        let refS1Width = halfWidth / downscaleFactor
        let refS1Height = halfHeight / downscaleFactor
        var refLatentS1: MLXArray? = nil
        var s1ImageKeyframes: [EncodedKeyframe] = []
        if let tailPath = continuationTailPath {
            // Continuation overrides the still image as the frame-0 anchor: the
            // tail latent carries the previous segment's final 8 frames of
            // motion, not just a pose.
            LTXDebug.log("[lipdub] continuation — encoding tail anchor at Stage 1 res \(halfWidth)x\(halfHeight)")
            s1ImageKeyframes = [try await encodeContinuationTail(
                path: tailPath, width: halfWidth, height: halfHeight)]
        } else if let imagePath = referenceImagePath {
            LTXDebug.log("[lipdub] image mode — encoding I2V keyframe at Stage 1 res \(halfWidth)x\(halfHeight)")
            s1ImageKeyframes = try await encodeKeyframes(
                [KeyframeInput(path: imagePath, pixelFrameIndex: 0)],
                width: halfWidth, height: halfHeight
            )
        } else {
            LTXDebug.log("[lipdub] encoding reference video at Stage 1 res \(refS1Width)x\(refS1Height)")
            refLatentS1 = try await encodeVideo(
                path: videoRefPath, width: refS1Width, height: refS1Height, numFrames: numFrames
            )
        }
        if isImageMode && ProcessInfo.processInfo.environment["LTX_LIPDUB_DUMP_VIDEO_REF"] == "1" {
            print("[lipdub][DIAG] LTX_LIPDUB_DUMP_VIDEO_REF=1 ignored in image mode — the I2V keyframe path produces no multi-frame reference latent to dump")
        }
        if !isImageMode, let f = refLatentS1, ProcessInfo.processInfo.environment["LTX_LIPDUB_DUMP_VIDEO_REF"] == "1" {
            let f32 = f.asType(.float32)
            MLX.eval(f32)
            try? MLX.save(arrays: ["data": f32], url: URL(fileURLWithPath: "/tmp/swift_video_ref_latent_s1.safetensors"))
            print("[lipdub][DIAG] dumped video ref latent S1 \(f32.shape) to /tmp/swift_video_ref_latent_s1.safetensors")
            let m = f32.mean().item(Float.self)
            let v = MLX.mean(MLX.square(f32 - m)).item(Float.self)
            print("[lipdub][DIAG]   stats: mean=\(m), std=\(sqrt(v)), min=\(f32.min().item(Float.self)), max=\(f32.max().item(Float.self))")
            return VideoGenerationResult(frames: MLXArray.zeros([1,3,1,64,64]), seed: 0, generationTime: 0, audioWaveform: nil, audioSampleRate: nil, effectivePrompt: effectivePrompt)
        }
        unloadVAEEncoder()  // free encoder; we'll reload it for Stage 2

        // 5. Extract reference audio + encode via AudioVAE.
        //
        // Three sources, in priority order:
        //   - Image mode: target audio is the ONLY audio source — load it directly,
        //     no speech-window alignment (the static reference has no speech timing).
        //   - Video mode + targetAudio: align target to source's speech window
        //     (silence-aware, pitch-preserving time-stretch).
        //   - Video mode without targetAudio: use the source video's own audio.
        let audioProcessor = AudioProcessor()
        let refWaveform: MLXArray
        if isImageMode {
            // `targetAudioPath` is non-nil here (validated above).
            let audioPath = targetAudioPath!
            print("[lipdub] image mode — using target audio \(audioPath) directly (no alignment)")
            refWaveform = try await audioProcessor.loadAudio(from: audioPath)
        } else if let targetAudioPath = targetAudioPath {
            print("[lipdub] aligning target audio \(targetAudioPath) to source video speech window")
            let sourceMonoMLX = try await audioProcessor.loadAudio(from: videoRefPath)
            let targetMonoMLX = try await audioProcessor.loadAudio(from: targetAudioPath)
            let sourceMono = AudioPreprocessor.mlxToMonoFloats(sourceMonoMLX)
            let targetMono = AudioPreprocessor.mlxToMonoFloats(targetMonoMLX)
            let (aligned, rate, srcWin, tgtWin) = try AudioPreprocessor.alignTargetToSource(
                source: sourceMono,
                target: targetMono,
                sampleRate: 16000
            )
            let srcStart = Float(srcWin.0) / 16000.0
            let srcEnd = Float(srcWin.1) / 16000.0
            let tgtStart = Float(tgtWin.0) / 16000.0
            let tgtEnd = Float(tgtWin.1) / 16000.0
            let srcSpeech = srcEnd - srcStart
            let tgtSpeech = tgtEnd - tgtStart
            print(String(format: "[lipdub] source speech window: %.3fs..%.3fs (%.3fs)", srcStart, srcEnd, srcSpeech))
            print(String(format: "[lipdub] target speech window: %.3fs..%.3fs (%.3fs)", tgtStart, tgtEnd, tgtSpeech))
            print(String(format: "[lipdub] time-stretch rate=%.3f (pitch preserved)", rate))
            refWaveform = MLXArray(aligned)  // mono (samples,)
        } else {
            refWaveform = try await audioProcessor.loadAudio(from: videoRefPath)
        }
        let refChannels = refWaveform.ndim == 1 ? 1 : refWaveform.dim(0)
        let refSamples = refWaveform.dim(refWaveform.ndim - 1)
        print("[lipdub] reference audio: \(refChannels) ch × \(refSamples) samples (\(String(format: "%.2f", Float(refSamples) / 16000.0))s)")
        // The audio reference sits at NEGATIVE RoPE positions (see buildAudioReference),
        // so the audio stream spans [-(refDur + 0.04), targetDur] — roughly TWICE the
        // segment duration, against the same audioMaxPos window the video side uses.
        // Measured: a 15.7 s segment lip-syncs with a constant ~0.75 s lag; the same
        // source at 9.7 s is in sync. 481 frames remains correct for generate/retake —
        // it is only LipDub that pays the doubled span.
        let lipdubAudioSpan = Float(refSamples) / 16000.0 + Float(numFrames) / 24.0 + 0.04
        let lipdubAudioBudget = Float(ltx2.config.audioMaxPos.first ?? 20)
        if lipdubAudioSpan > lipdubAudioBudget {
            let maxFrames = Int(((lipdubAudioBudget - 0.04) / 2 * 24 - 1) / 8) * 8 + 1
            // The remedy differs by mode: continuationTailPath is image-mode only
            // (video mode throws on it — continuity there comes from slicing the
            // source), so naming it unconditionally would send a video-mode caller
            // straight into that throw.
            let remedy = isImageMode
                ? "chain them with continuationTailPath"
                : "slice the reference video the same way and re-run per slice"
            print(String(
                format: "[lipdub] WARNING: audio reference + target span %.1fs exceeds the "
                    + "%.0fs RoPE window — expect a growing lip-sync lag. Split the dialogue "
                    + "into segments of at most %d frames (%.1fs) and %@.",
                lipdubAudioSpan, lipdubAudioBudget, maxFrames, Float(maxFrames) / 24.0, remedy))
        }
        let refMel = try audioProcessor.melSpectrogram(refWaveform)
        print("[lipdub] reference mel spectrogram: \(refMel.shape)")
        let refAudioLatent = try audioVAE.encode(refMel)  // (1, 8, T_audio_ref, 16)
        if ProcessInfo.processInfo.environment["LTX_LIPDUB_DUMP_AUDIO"] == "1" {
            let melF32 = refMel.asType(.float32)
            let latF32 = refAudioLatent.asType(.float32)
            MLX.eval(melF32, latF32)
            try? MLX.save(arrays: ["data": melF32], url: URL(fileURLWithPath: "/tmp/swift_audio_mel.safetensors"))
            try? MLX.save(arrays: ["data": latF32], url: URL(fileURLWithPath: "/tmp/swift_audio_latent.safetensors"))
            print("[lipdub][DIAG] dumped mel \(melF32.shape) and latent \(latF32.shape) to /tmp/swift_audio_*.safetensors — exiting")
            return VideoGenerationResult(frames: MLXArray.zeros([1,3,1,64,64]), seed: 0, generationTime: 0, audioWaveform: nil, audioSampleRate: nil, effectivePrompt: effectivePrompt)
        }
        MLX.eval(refAudioLatent)

        // 6. Build Stage 1 reference contexts.
        let s1VideoRefCtx: AppendKeyframeContext?
        if isImageMode {
            s1VideoRefCtx = prepareKeyframeAppend(
                encoded: s1ImageKeyframes,
                shape: stage1Shape,
                hasAudio: true,
                refConfig: ltx2.config,
                stageLabel: "LipDub Stage 1 (image keyframe)"
            )
        } else {
            s1VideoRefCtx = buildVideoReference(
                referenceLatent: refLatentS1!,
                targetShape: stage1Shape,
                downscaleFactor: downscaleFactor,
                hasAudio: true,
                refConfig: ltx2.config
            )
        }
        // [DIAGNOSTIC] LTX_LIPDUB_SKIP_AUDIO_REF=1 disables the audio reference (audio is still
        // denoised but with no negative-position reference tokens) to isolate audio-ref contribution.
        let skipAudioRef = ProcessInfo.processInfo.environment["LTX_LIPDUB_SKIP_AUDIO_REF"] == "1"
        let s1AudioRefCtx: AudioReferenceContext? = skipAudioRef ? nil : buildAudioReference(
            referenceLatent: refAudioLatent,
            targetAudioFrames: audioNumFrames,
            refConfig: ltx2.config
        )
        if skipAudioRef {
            print("[lipdub][DIAG] LTX_LIPDUB_SKIP_AUDIO_REF=1 — audio reference disabled")
        }
        // [DIAGNOSTIC] LTX_LIPDUB_SKIP_VIDEO_REF=1 disables the video reference (no IC-LoRA append).
        let skipVideoRef = ProcessInfo.processInfo.environment["LTX_LIPDUB_SKIP_VIDEO_REF"] == "1"
        if skipVideoRef {
            print("[lipdub][DIAG] LTX_LIPDUB_SKIP_VIDEO_REF=1 — video reference disabled")
        }
        let s1VideoRefCtxEffective: AppendKeyframeContext? = skipVideoRef ? nil : s1VideoRefCtx
        LTXDebug.log("[lipdub] Stage 1 video ref tokens=\(s1VideoRefCtxEffective?.guideCount ?? 0), audio ref tokens=\(s1AudioRefCtx?.guideCount ?? 0)")

        // 7. Initial noise + Stage 1 sigma schedule (always distilled).
        if let seed = config.seed { MLXRandom.seed(seed) }
        var videoLatent = generateNoise(shape: stage1Shape, seed: config.seed)
        let initialAudio = MLXRandom.normal(
            [1, Self.audioLatentChannels, audioNumFrames, Self.audioLatentMelBins]
        ).asType(.float32)
        var audioLatentPacked: MLXArray? = packAudioLatents(initialAudio)

        let stage1Scheduler = LTXScheduler(isDistilled: true)
        stage1Scheduler.setTimesteps(
            numSteps: 8, distilled: true, latentTokenCount: stage1Shape.tokenCount
        )
        let stage1Sigmas = stage1Scheduler.sigmas
        videoLatent = videoLatent * stage1Sigmas[0]
        if let ap = audioLatentPacked { audioLatentPacked = ap * stage1Sigmas[0] }
        MLX.eval(videoLatent)
        if let ap = audioLatentPacked { MLX.eval(ap) }

        // 8. Stage 1 denoise loop — dual stream, both refs appended.
        LTXMemoryManager.setPhase(.denoising)
        let totalSteps = (stage1Sigmas.count - 1) + (STAGE_2_DISTILLED_SIGMA_VALUES.count - 1)
        let stage1NumSteps = stage1Sigmas.count - 1
        for step in 0..<stage1NumSteps {
            let sigma = stage1Sigmas[step]
            let sigmaNext = stage1Sigmas[step + 1]
            onProgress?(GenerationProgress(
                currentStep: step, totalSteps: totalSteps, sigma: sigma, phase: .denoising
            ))
            let vel = runDenoiseStep(
                sigma: sigma,
                videoLatent: videoLatent,
                audioLatentPacked: audioLatentPacked,
                shape: stage1Shape,
                videoAppendCtx: s1VideoRefCtxEffective,
                audioRefCtx: s1AudioRefCtx,
                audioNumFrames: audioNumFrames,
                videoTextEmbeddings: videoTextEmbeddings,
                audioTextEmbeddings: audioTextEmbeddings,
                textMask: textMask
            )
            videoLatent = stage1Scheduler.step(
                latent: videoLatent, velocity: vel.video,
                sigma: sigma, sigmaNext: sigmaNext
            )
            if let av = vel.audio, let ap = audioLatentPacked {
                let updatedAudio = ap + (sigmaNext - sigma) * av
                audioLatentPacked = updatedAudio
                MLX.eval(videoLatent, updatedAudio)
            } else {
                MLX.eval(videoLatent)
            }
            if (step + 1) % 5 == 0 { Memory.clearCache() }
            LTXDebug.log("[lipdub] Stage 1 step \(step)/\(stage1NumSteps) σ=\(String(format: "%.4f", sigma))→\(String(format: "%.4f", sigmaNext))")
        }
        guard let s1AudioLatentPacked = audioLatentPacked else {
            throw LTXError.generationFailed("Stage 1 audio latent missing after denoising")
        }

        // 9. Upscale video latent (denormalize → spatial upscale 2x → renormalize).
        onProgress?(GenerationProgress(
            currentStep: stage1NumSteps, totalSteps: totalSteps, sigma: 0, phase: .upscaling
        ))
        let upscaler = try loadSpatialUpscaler(from: upscalerWeightsPath)
        let mean5d = vaeDecoder.meanOfMeans.asType(.float32).reshaped([1, -1, 1, 1, 1])
        let std5d = vaeDecoder.stdOfMeans.asType(.float32).reshaped([1, -1, 1, 1, 1])
        let denormedS1 = videoLatent * std5d + mean5d
        let upscaled = upscaler(denormedS1)
        videoLatent = (upscaled - mean5d) / std5d
        MLX.eval(videoLatent)

        // 10. Stage 2 reference: same dual-path as Stage 1, at full resolution.
        let refS2Width = config.width / downscaleFactor
        let refS2Height = config.height / downscaleFactor
        var refLatentS2: MLXArray? = nil
        var s2ImageKeyframes: [EncodedKeyframe] = []
        if let tailPath = continuationTailPath {
            LTXDebug.log("[lipdub] continuation — encoding tail anchor at Stage 2 res \(config.width)x\(config.height)")
            s2ImageKeyframes = [try await encodeContinuationTail(
                path: tailPath, width: config.width, height: config.height)]
        } else if let imagePath = referenceImagePath {
            LTXDebug.log("[lipdub] image mode — encoding I2V keyframe at Stage 2 res \(config.width)x\(config.height)")
            s2ImageKeyframes = try await encodeKeyframes(
                [KeyframeInput(path: imagePath, pixelFrameIndex: 0)],
                width: config.width, height: config.height
            )
        } else {
            LTXDebug.log("[lipdub] re-encoding reference video at Stage 2 res \(refS2Width)x\(refS2Height)")
            refLatentS2 = try await encodeVideo(
                path: videoRefPath, width: refS2Width, height: refS2Height, numFrames: numFrames
            )
        }
        unloadVAEEncoder()

        // 11. Build Stage 2 reference contexts.
        let s2VideoRefCtx: AppendKeyframeContext?
        if isImageMode {
            s2VideoRefCtx = prepareKeyframeAppend(
                encoded: s2ImageKeyframes,
                shape: stage2Shape,
                hasAudio: true,
                refConfig: ltx2.config,
                stageLabel: "LipDub Stage 2 (image keyframe)"
            )
        } else {
            s2VideoRefCtx = buildVideoReference(
                referenceLatent: refLatentS2!,
                targetShape: stage2Shape,
                downscaleFactor: downscaleFactor,
                hasAudio: true,
                refConfig: ltx2.config
            )
        }
        // Stage 2 audio reference = Stage 1 denoised audio (already packed).
        let s2AudioRefCtx = buildAudioReferenceFromPacked(
            packed: s1AudioLatentPacked,
            targetAudioFrames: audioNumFrames,
            refConfig: ltx2.config
        )

        // 12. Re-noise video for Stage 2; audio is FROZEN to s1 output.
        let stage2Sigmas = STAGE_2_DISTILLED_SIGMA_VALUES
        let videoNoise = generateNoise(shape: stage2Shape)
        let s2NoiseScale = stage2Sigmas[0]
        videoLatent = MLXArray(s2NoiseScale) * videoNoise + MLXArray(1.0 - s2NoiseScale) * videoLatent
        audioLatentPacked = s1AudioLatentPacked  // frozen for Stage 2
        MLX.eval(videoLatent)

        // 13. Stage 2 denoise loop — manual Euler for video, audio frozen (discard vel.audio).
        let stage2NumSteps = stage2Sigmas.count - 1
        for step in 0..<stage2NumSteps {
            let sigma = stage2Sigmas[step]
            let sigmaNext = stage2Sigmas[step + 1]
            onProgress?(GenerationProgress(
                currentStep: stage1NumSteps + step, totalSteps: totalSteps, sigma: sigma, phase: .refinement
            ))
            let vel = runDenoiseStep(
                sigma: sigma,
                videoLatent: videoLatent,
                audioLatentPacked: audioLatentPacked,
                shape: stage2Shape,
                videoAppendCtx: skipVideoRef ? nil : s2VideoRefCtx,
                audioRefCtx: skipAudioRef ? nil : s2AudioRefCtx,
                audioNumFrames: audioNumFrames,
                videoTextEmbeddings: videoTextEmbeddings,
                audioTextEmbeddings: audioTextEmbeddings,
                textMask: textMask
            )
            let dt = sigmaNext - sigma
            videoLatent = videoLatent + MLXArray(dt) * vel.video
            // AUDIO FROZEN — discard vel.audio (matches Python lipdub.py frozen=True)
            MLX.eval(videoLatent)
            LTXDebug.log("[lipdub] Stage 2 step \(step)/\(stage2NumSteps) σ=\(String(format: "%.4f", sigma))→\(String(format: "%.4f", sigmaNext))")
        }

        // 14. Unload transformer before VAE decode if memory pressure is a concern.
        // (With unloadAfterUse the fused transformer is gone — the next LipDub run
        // must reload models. Use MemoryOptimizationConfig.disabled to keep the
        // fused transformer across consecutive same-LoRA segments.)
        if memoryOptimization.unloadAfterUse {
            self.ltx2Transformer = nil
            self.lipdubFusion = nil
            Memory.clearCache()
        }

        // 15. Decode video.
        onProgress?(GenerationProgress(
            currentStep: totalSteps, totalSteps: totalSteps, sigma: 0, phase: .decoding
        ))
        LTXMemoryManager.setPhase(.vaeDecode)
        let videoTensor = decodeFrames(latent: videoLatent)
        MLX.eval(videoTensor)
        let trimmedVideo: MLXArray
        if videoTensor.dim(0) > numFrames {
            trimmedVideo = videoTensor[0..<numFrames]
        } else {
            trimmedVideo = videoTensor
        }

        // 16. Decode audio (from Stage 1 packed latent — matches Python lipdub.py:264).
        let audioLatentUnpacked = unpackAudioLatents(s1AudioLatentPacked, numFrames: audioNumFrames)
        let audioWaveform = decodeAudio(
            latents: audioLatentUnpacked,
            audioVAE: audioVAE,
            vocoder: vocoder
        )
        MLX.eval(audioWaveform)

        LTXMemoryManager.resetCacheLimit()
        // Free MLX workspace buffers between runs (weights stay resident). With
        // unloadAfterUse == false nothing else clears the cache, and decode
        // buffers accumulating across consecutive segments degrade later runs.
        Memory.clearCache()
        let generationTime = Date().timeIntervalSince(generationStart)
        LTXDebug.log("[lipdub] total generation time: \(String(format: "%.1f", generationTime))s")

        return VideoGenerationResult(
            frames: trimmedVideo,
            seed: config.seed ?? 0,
            generationTime: generationTime,
            audioWaveform: audioWaveform,
            audioSampleRate: vocoder.outputSampleRate,
            effectivePrompt: effectivePrompt
        )
    }

    /// Repair a VLM-enhanced LipDub prompt when the speaking-verb hint is missing.
    ///
    /// The LipDub IC-LoRA was trained on **English-wrapped** prompts of the form
    /// `<scene>, speaking in <LANGUAGE> saying: "<DIALOGUE>"` — the wrapper is
    /// always English even when the dialogue is in another language. The I2V
    /// system prompt the VLM runs under is instructed to preserve the user's
    /// quoted dialogue verbatim, but the wrapper itself may be rephrased
    /// (`"speaks in Spanish"`) or, rarely, dropped.
    ///
    /// We treat the enhanced output as still valid when it contains any common
    /// English speaking-verb variant (`speaking|speaks|saying|says` followed by
    /// `in` as a whole word — substrings inside `speaking individually` /
    /// `says intermittently` are NOT matches). If none survive AND the original
    /// prompt did contain a `speaking in` wrapper, that wrapper's tail is glued
    /// to the enhanced text so the LoRA stays engaged. Joiner picks `" "` after
    /// any terminal punctuation (`.,;?!"')]…—`) and `", "` otherwise.
    ///
    /// > **Cross-language caveat.** Authors using non-English wrappers
    /// > (`parlant en …`, `hablando en …`, `sprechend in …`) get no fallback —
    /// > but those wrappers also don't match the LoRA's trained distribution,
    /// > so the right fix is the prompt, not this helper.
    ///
    /// - Parameters:
    ///   - enhanced: VLM-enhanced prompt.
    ///   - original: User's input prompt (the one passed to the VLM).
    /// - Returns: A tuple `(final, reappended)` — `final` is the prompt to send onward;
    ///   `reappended` is non-nil only when the fallback fired, set to the signature
    ///   string that was glued onto the end (useful for logging / tests).
    static func applyLipDubSignatureFallback(
        enhanced: String, original: String
    ) -> (final: String, reappended: String?) {
        if Self.containsSpeakingVerbWrapper(enhanced) {
            return (enhanced, nil)
        }
        // Search the original directly with case-insensitive match — avoids the
        // Swift String-index portability hazard of indexing `original` with an
        // index obtained from `original.lowercased()` (lowercasing can change
        // unit length: Turkish `İ` → `i\u{0307}`, German `ẞ` → `ss`, etc.).
        guard let sigRange = Self.firstSpeakingInRange(in: original) else {
            return (enhanced, nil)
        }
        let signature = String(original[sigRange.lowerBound...])
        let trimmed = enhanced.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            return (signature, signature)
        }
        let terminalChars: Set<Character> =
            [".", ",", ";", ":", "?", "!", "\"", "'", ")", "]", "}", "…", "—", "–"]
        let endsTerminally = trimmed.last.map { terminalChars.contains($0) } ?? false
        let joiner = endsTerminally ? " " : ", "
        return (trimmed + joiner + signature, signature)
    }

    /// Whether `text` contains any common English speaking-verb wrapper —
    /// `(speaking|speaks|saying|says)` followed by `in` as a WHOLE WORD.
    /// `"speaking individually"` / `"says intermittently"` do NOT match.
    private static func containsSpeakingVerbWrapper(_ text: String) -> Bool {
        let verbs = ["speaking", "speaks", "saying", "says"]
        for verb in verbs {
            var searchStart = text.startIndex
            while let r = text.range(
                of: verb, options: .caseInsensitive, range: searchStart..<text.endIndex
            ) {
                // After the verb we want: 1+ whitespace, then literal "in",
                // then a word-boundary char (whitespace, punctuation, EOS).
                var i = r.upperBound
                // Skip whitespace.
                while i < text.endIndex, text[i].isWhitespace { i = text.index(after: i) }
                let afterWs = i
                let inEnd = text.index(afterWs, offsetBy: 2, limitedBy: text.endIndex)
                if let inEnd = inEnd, afterWs < text.endIndex {
                    let twoChars = text[afterWs..<inEnd]
                    if twoChars.lowercased() == "in" {
                        // Boundary: end-of-string OR next char is non-letter.
                        if inEnd == text.endIndex || !text[inEnd].isLetter {
                            return true
                        }
                    }
                }
                searchStart = r.upperBound
            }
        }
        return false
    }

    /// Find the first `speaking in` (as whole-word wrapper) in `original`.
    private static func firstSpeakingInRange(in original: String) -> Range<String.Index>? {
        var searchStart = original.startIndex
        while let r = original.range(
            of: "speaking", options: .caseInsensitive, range: searchStart..<original.endIndex
        ) {
            var i = r.upperBound
            while i < original.endIndex, original[i].isWhitespace { i = original.index(after: i) }
            let afterWs = i
            if let inEnd = original.index(afterWs, offsetBy: 2, limitedBy: original.endIndex),
               afterWs < original.endIndex,
               original[afterWs..<inEnd].lowercased() == "in",
               inEnd == original.endIndex || !original[inEnd].isLetter {
                return r.lowerBound..<inEnd
            }
            searchStart = r.upperBound
        }
        return nil
    }

    /// Encode a video into latent space using the VAE encoder
    ///
    /// - Parameters:
    ///   - path: Path to input video
    ///   - width: Target video width
    ///   - height: Target video height
    ///   - numFrames: Number of frames to extract
    /// - Returns: Video latent tensor (1, 128, latent_F, latent_H, latent_W)
    internal func encodeVideo(
        path: String, width: Int, height: Int, numFrames: Int, tail: Bool = false
    ) async throws -> MLXArray {
        let videoTensor = try await loadVideo(
            from: path, width: width, height: height, numFrames: numFrames, tail: tail)
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

    /// Load VAE encoder weights from wherever this checkpoint keeps them.
    ///
    /// Unified checkpoints hold them under `vae.encoder.*`; split ones put them in
    /// the VAE file. Reading the wrong file yields *zero* matching keys and leaves
    /// the encoder at its random initialisation, which does not fail loudly — it
    /// silently encodes every conditioning image to noise. Hence the guard below.
    private func loadVAEEncoder() async throws {
        if vaeEncoder != nil { return }  // Already loaded

        LTXDebug.log("Loading VAE encoder...")
        let checkpoint = try await resolveCheckpoint()
        let source = LTXCheckpointSource(model: model, paths: checkpoint)
        let encoderWeights = try source.loadVAEEncoderWeights()
        guard !encoderWeights.isEmpty else {
            throw LTXError.weightLoadingFailed(
                "No VAE encoder weights found in \(checkpoint.videoVAE.lastPathComponent)")
        }

        let encoder = VideoEncoder()
        try LTXWeightLoader.applyVAEEncoderWeights(encoderWeights, to: encoder)
        eval(encoder.parameters())
        Memory.clearCache()

        self.vaeEncoder = encoder
        LTXDebug.log("VAE encoder loaded (\(encoderWeights.count) weights)")
    }


    /// Encode a list of keyframes into latent space, returning each one tagged with
    /// its target latent slot. The VAE encoder is loaded once and reused.
    ///
    /// Each keyframe is encoded independently (single-frame input) so the result is
    /// always a `(1, 128, 1, H/32, W/32)` latent that can be placed at any latent slot.
    /// Encode a segment-continuation anchor from the PREVIOUS segment (issue
    /// #35): its **last 9 pixel frames** are VAE-encoded (2 latent frames) and
    /// the LAST latent frame — the one carrying 8 pixel frames of actual motion
    /// — becomes a frame-0 guide keyframe. Unlike a still-image anchor, it
    /// preserves velocity across the segment cut.
    ///
    /// Contract for callers: **pass the previous segment's video file itself**.
    /// The tail is located here, natively (`loadVideo(tail: true)` reads the
    /// last 9 frames from the track's own frame rate), so no clip preparation
    /// is required. A pre-trimmed 9-frame clip still works — its last 9 frames
    /// are all of them — which keeps callers written against the old contract
    /// valid. The new segment's first output frame reproduces the anchor: drop
    /// one frame at concatenation (overlap-and-trim).
    private func encodeContinuationTail(
        path: String,
        width: Int,
        height: Int
    ) async throws -> EncodedKeyframe {
        let latent = try await encodeVideo(
            path: path, width: width, height: height, numFrames: 9, tail: true)
        let lastIdx = latent.dim(2) - 1
        let tail = latent[0..., 0..., lastIdx..<(lastIdx + 1), 0..., 0...]
        MLX.eval(tail)
        LTXDebug.log("[lipdub] continuation tail latent \(tail.shape) (last of \(latent.dim(2)) latent frames)")
        return EncodedKeyframe(latentIdx: 0, latent: tail, pixelFrameIndex: 0)
    }

    internal func encodeKeyframes(
        _ keyframes: [KeyframeInput],
        width: Int,
        height: Int
    ) async throws -> [EncodedKeyframe] {
        guard !keyframes.isEmpty else { return [] }

        try await loadVAEEncoder()
        guard let encoder = vaeEncoder else {
            throw LTXError.modelNotLoaded("VAE encoder failed to load")
        }
        guard let vaeDecoder = vaeDecoder else {
            throw LTXError.modelNotLoaded("VAE decoder not loaded (needed for latent statistics)")
        }
        let mean5d = vaeDecoder.meanOfMeans.asType(.float32).reshaped([1, -1, 1, 1, 1])
        let std5d = vaeDecoder.stdOfMeans.asType(.float32).reshaped([1, -1, 1, 1, 1])

        var encoded: [EncodedKeyframe] = []
        encoded.reserveCapacity(keyframes.count)
        for kf in keyframes {
            let imageTensor = try loadImage(from: kf.path, width: width, height: height)
            MLX.eval(imageTensor)
            let latent = encoder(imageTensor)
            let normalized = (latent.asType(.float32) - mean5d) / std5d
            MLX.eval(normalized)
            let slot = pixelFrameToLatentFrame(kf.pixelFrameIndex)
            LTXDebug.log("Keyframe \(kf.path) → pixel \(kf.pixelFrameIndex) → latent slot \(slot), shape=\(normalized.shape)")
            encoded.append(EncodedKeyframe(latentIdx: slot, latent: normalized, pixelFrameIndex: kf.pixelFrameIndex))
        }
        return encoded
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
        // Exporting a LipDub-fused transformer would persist the destructive
        // IC-LoRA delta to disk as if it were pristine base weights.
        try ensureNoLipDubLoRAFused(wouldCorrupt: "the exported weights (persisted to disk)")
        let beacon = RuntimeBeacon.begin(task: "export-quantized", model: self.model.rawValue)
        defer { beacon?.end() }
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

        // Enhancer choice mirrors upstream: on 2.3 the (full-headed) Gemma 3
        // self-enhances, so the shared Gemma 3 VLM plays that role; on 2.5 the
        // bundled Gemma 4 encoder is encode-only (vestigial LM head, measured —
        // docs/knowledge), so upstream mandates a separate small Gemma 4
        // instruct enhancer (`--prompt-enhancer-gemma-root`, E2B-it).
        if model.family == .ltx25 {
            return try await enhancePromptWithGemma4(
                prompt, imagePath: imagePath, startTime: startTime)
        }
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
            print("Prompt enhancer: Gemma 3 12B (downloading if needed, ~7.5GB)...")
            fflush(stdout)
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
        // Note: repetitionPenalty disabled — mlx-swift-lm TokenRing.loadPrompt has a bug
        // where prompt.dim(0) returns batch dim (1) instead of seq length for 2D prompts,
        // causing a shape mismatch in the ring buffer. Tracked upstream.
        let generateParams = GenerateParameters(
            maxTokens: maxTokens,
            temperature: temperature,
            topP: 0.95
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

        // Always shown: the enhanced text is what actually gets encoded, and
        // hiding it behind --debug made runs impossible to audit.
        print("Enhanced prompt (Gemma 3 12B):\n\(cleaned)")
        fflush(stdout)

        // Unload VLM to free memory for the main pipeline
        LTXDebug.log("Unloading VLM...")
        Memory.clearCache()
        LTXMemoryManager.logMemoryState("after VLM unload")

        return cleaned
    }

    /// Upstream's dedicated Gemma 4 enhancer prompts (gemma4_i2v/t2v_system_prompt.txt)
    /// — caption-style output, image grounding, framing triple; distinct from the
    /// Gemma 3 self-enhance prompts above, which stay on the 2.3 path.
    private static let promptEnhancementGemma4I2VSystemPrompt = """
You are given a REFERENCE IMAGE (the exact first frame of the video) and a user's short image-to-video request. Write a single, highly detailed audio-visual caption describing the video that BEGINS from this exact reference image and best fulfills that request, in the EXACT style of the training captions used for this video model. The generated video is scored against the user's ORIGINAL request, so preserve every element the user stated; expand faithfully into the full caption style without contradicting or dropping anything they asked for.

FIRST-FRAME / IMAGE GROUNDING (do this first): the opening of your caption must match the reference image exactly — same subject(s), identity, appearance, clothing, setting, lighting, and composition as shown. The video starts on this frame; describe it faithfully, then narrate chronologically as the user's requested action unfolds from it. Never contradict, replace, or invent things not consistent with the image. Single continuous take — no hard cuts.

Match this captioning style precisely:

1. Begin immediately with the action or visual detail. Do NOT use "The scene opens…", "We see…", "There is…".

2. Objective, observable description only. Do not infer emotions or intentions — describe what is visible and audible (e.g. not "he looks sad" but "his eyebrows angle downward and his lips are pressed together").

3. Full visual detail: environment (materials, textures, lighting, colors), character appearance (clothing, posture, facial details), and the spatial positioning of all elements — grounded in and consistent with the reference image. When a human appears, identify them specifically (gendered terms when clearly implied; differentiate multiple people consistently) and describe visible physical attributes — apparent gender presentation, skin tone, estimated age group, hair color/length/style, build, clothing and accessories. Do not infer ethnicity, nationality, religion, or culture.

4. Precise motion and cinematic description. For every shot you MUST include, woven naturally into the prose (never as tags or labels):
   - Shot type (exactly one: extreme wide shot / wide shot / medium shot / medium close-up / close-up / extreme close-up) — consistent with how the reference image is framed at the start.
   - Camera motion (always stated; if none, explicitly say the camera remains static). Camera movement is expected and good — match the user if they specified it, otherwise choose the treatment that best presents the requested scene starting from this frame.
   - Camera viewpoint relative to subject (front-facing / back-facing / side view / over-the-shoulder / top-down / low-angle / high-angle) — matching the reference image's viewpoint at the opening.
   Express these as flowing prose: "a medium shot frames…, captured from a front-facing angle as the camera slowly pans…". Never as "medium shot, static camera —".

5. Complete soundscape, integrated naturally: any dialogue (quote it exactly, in the original language), tone of voice, background music (type, mood, volume changes), and environmental sounds (footsteps, wind, traffic, animals). If the request implies sound, describe it plausibly.

6. Strict chronological, real-time flow using transitions like "Initially…", "A moment later…", "Simultaneously…". Keep the user's requested motion/action central and in motion throughout.

7. One single continuous paragraph. No bullet points, no section headers, no labels like "Audio:" or "Visual:". Exhaustive and lossless — include background elements, subtle movements, lighting, secondary sounds — detailed enough to reconstruct the scene. Aim for a rich, complete paragraph (roughly 150–220 words).

If the user wrote in another language, produce the English caption of the same content. Output ONLY the caption text — no JSON, no preamble.

AESTHETIC QUALITY (in addition to the above, without breaking the objective caption style or contradicting the reference image): render the described scene with strong visual production value — cinematic, film-grade color and contrast, beautiful natural lighting, crisp fine detail and texture, pleasing composition and depth. Weave these quality descriptors naturally into the same observable prose (e.g. "warm cinematic lighting", "richly saturated film-grade color", "crisp high-resolution detail") — describe how the exact requested scene, starting from this frame, LOOKS at its most visually striking, never adding new objects or actions and never contradicting the first frame. Keep everything else (first-frame grounding, framing triple, soundscape, chronological single paragraph, faithfulness) exactly as specified.
"""

    private static let promptEnhancementGemma4T2VSystemPrompt = """
You are given a user's short text-to-video request. Write a single, highly detailed audio-visual caption describing the video that best fulfills that request, in the EXACT style of the training captions used for this video model. The generated video is scored against the user's ORIGINAL request, so preserve every element the user stated; expand faithfully into the full caption style without contradicting or dropping anything they asked for.

Match this captioning style precisely:

1. Begin immediately with the action or visual detail. Do NOT use "The scene opens…", "We see…", "There is…".

2. Objective, observable description only. Do not infer emotions or intentions — describe what is visible and audible (e.g. not "he looks sad" but "his eyebrows angle downward and his lips are pressed together").

3. Full visual detail: environment (materials, textures, lighting, colors), character appearance (clothing, posture, facial details), and the spatial positioning of all elements. When a human appears, identify them specifically (gendered terms when clearly implied; differentiate multiple people consistently) and describe visible physical attributes — apparent gender presentation, skin tone, estimated age group, hair color/length/style, build, clothing and accessories. Do not infer ethnicity, nationality, religion, or culture.

4. Precise motion and cinematic description. For every shot you MUST include, woven naturally into the prose (never as tags or labels):
   - Shot type (exactly one: extreme wide shot / wide shot / medium shot / medium close-up / close-up / extreme close-up)
   - Camera motion (always stated; if none, explicitly say the camera remains static). Camera movement is expected and good — match the user if they specified it, otherwise choose the treatment that best presents the requested scene.
   - Camera viewpoint relative to subject (front-facing / back-facing / side view / over-the-shoulder / top-down / low-angle / high-angle).
   Express these as flowing prose: "a medium shot frames…, captured from a front-facing angle as the camera slowly pans…". Never as "medium shot, static camera —".

5. Complete soundscape, integrated naturally: any dialogue (quote it exactly, in the original language), tone of voice, background music (type, mood, volume changes), and environmental sounds (footsteps, wind, traffic, animals). If the request implies sound, describe it plausibly.

6. Strict chronological, real-time flow using transitions like "Initially…", "A moment later…", "Simultaneously…". Keep every stated action in motion.

7. One single continuous paragraph. No bullet points, no section headers, no labels like "Audio:" or "Visual:". Exhaustive and lossless — include background elements, subtle movements, lighting, secondary sounds — detailed enough to reconstruct the scene. Aim for a rich, complete paragraph (roughly 150–220 words).

If the user wrote in another language, produce the English caption of the same content. Output ONLY the caption text — no JSON, no preamble.

AESTHETIC QUALITY (in addition to the above, without breaking the objective caption style): render the described scene with strong visual production value — cinematic, film-grade color and contrast, beautiful natural lighting, crisp fine detail and texture, pleasing composition and depth. Weave these quality descriptors naturally into the same observable prose (e.g. "warm cinematic lighting", "richly saturated film-grade color", "crisp high-resolution detail") — describe how the exact requested scene LOOKS at its most visually striking, never adding new objects or actions. Keep everything else (framing triple, soundscape, chronological single paragraph, faithfulness) exactly as specified.
"""

    /// Enhancement decoding policy, measured on the 2CV bench (E2B bf16, same
    /// prompt/image/seed, one variable at a time):
    ///
    /// | thinking | ngram 5 | timeline produced                        |
    /// |----------|---------|------------------------------------------|
    /// | off      | on      | hover spans the launch; 3 marker formats |
    /// | **on**   | **off** | **hover bounded, launch at 07:500, one format** |
    /// | on       | on      | *no* timestamps at all                   |
    ///
    /// Reasoning is what fixes the timeline arithmetic; the n-gram ban then
    /// forbade the caption from repeating what the reasoning had just written,
    /// which erased the markers entirely — so the two could not both be on.
    /// gemma-4-swift-mlx 1.5.0 resolves that: the ban window can skip the
    /// thought channel, so reasoning and loop protection now coexist.
    /// Reasoning costs ~350 tokens before the answer, hence the raised budget.
    private let enhancerThinking = true

    /// N-gram blocking stays off, now as a measured choice rather than a
    /// workaround. gemma-4-swift-mlx 1.5.0 made it *possible* to run it
    /// alongside reasoning (its window can skip the thought channel), and that
    /// works — but a four-prompt bench against the reference service found it
    /// inert on three prompts (byte-identical captions) and harmful on the
    /// fourth, the one dense with repeated timestamps: two marker formats
    /// instead of one, and 250 characters lost. Its purpose is loop protection,
    /// and no loop has ever been observed here.
    ///
    /// Flip to `true` if a prompt ever loops; the machinery is in place.
    private var enhancerNGram: Bool {
        ProcessInfo.processInfo.environment["LTX_ENH_NGRAM"].map { $0 == "1" } ?? false
    }

    /// LTX-2.5 prompt enhancement through our own `Gemma4Swift` stack.
    ///
    /// Mirrors upstream\'s design: the bundled 12B encoder is encode-only
    /// (vestigial LM head, measured — docs/knowledge), so enhancement runs on a
    /// separate small generative Gemma 4 E2B-it (bf16, as the reference space
    /// runs it), greedy, 600 tokens, no_repeat_ngram_size 5
    /// (`GEMMA4_ENHANCE_GENERATION_KWARGS`). The checkpoint downloads through our
    /// `ModelDownloader` so `--models-dir` routes it like every other model.
    private func enhancePromptWithGemma4(
        _ prompt: String,
        imagePath: String?,
        startTime: Date
    ) async throws -> String {
        print("Prompt enhancer: Gemma 4 E2B-it bf16 via Gemma4Swift (downloading if needed, ~10GB)...")
        fflush(stdout)
        // Its own phase: enhancement is a separate model and a separate cost,
        // and folding it into text encoding would misattribute both.
        LTXVideoProfiler.shared.start("Prompt Enhancement")
        defer { LTXVideoProfiler.shared.end("Prompt Enhancement") }
        let dir = try await downloader.downloadGemma4Enhancer { p in
            if let f = p.currentFile { print("  \(f)"); fflush(stdout) }
        }

        let g4 = await Gemma4Pipeline()
        try await g4.load(from: dir.resolvingSymlinksInPath(), multimodal: true)
        LTXDebug.log("Gemma 4 E2B enhancer loaded")

        // Gemma folds the system role into the first user turn anyway, and the
        // multimodal path takes a single prompt string — prepend explicitly.
        MLXRandom.seed(42)
        let isI2V = imagePath != nil
        let stream: AsyncThrowingStream<String, Error>
        if let imagePath {
            let pixels = try Gemma4ImageProcessor.processImage(
                url: URL(fileURLWithPath: imagePath))
            stream = try await g4.chatStreamMultimodal(
                prompt: "user prompt: \(prompt)",
                pixelValues: pixels,
                // A real system turn, as the reference space sends it — the
                // Gemma 4 template renders it as a distinct turn rather than
                // folding it into the user message (gemma-4-swift-mlx 1.3.0).
                systemPrompt: Self.promptEnhancementGemma4I2VSystemPrompt,
                temperature: 0.0,
                // Reasoning costs ~350 tokens on a short answer, but it scales
                // with the prompt: a plain scene description reasoned long
                // enough to leave the caption cut mid-word at 1200. The budget
                // has to cover thought *and* answer, so it is generous.
                maxTokens: enhancerThinking ? 2400 : 600,
                noRepeatNGramSize: enhancerNGram ? 5 : nil,
                // Deliberate deviation from HF semantics (and from the reference
                // space): ban only n-grams repeated within the GENERATED text, so
                // the caption may quote the prompt's timeline verbatim. Measured:
                // with the prompt in the window, timestamps come out mangled and
                // the duration head over-predicts ~5 s (docs/knowledge pitfall).
                noRepeatNGramIncludesPrompt: false,
                // The reasoning must not count as "already written", or the
                // caption cannot restate the timeline it just worked out.
                noRepeatNGramIncludesThinking: false,
                templateVariables: enhancerThinking ? ["enable_thinking": true] : nil)
        } else {
            stream = try await g4.chatStream(
                prompt: "user prompt: \(prompt)",
                systemPrompt: Self.promptEnhancementGemma4T2VSystemPrompt,
                temperature: 0.0,
                maxTokens: 600,
                noRepeatNGramSize: 5,
                // Deliberate deviation from HF semantics (and from the reference
                // space): ban only n-grams repeated within the GENERATED text, so
                // the caption may quote the prompt's timeline verbatim. Measured:
                // with the prompt in the window, timestamps come out mangled and
                // the duration head over-predicts ~5 s (docs/knowledge pitfall).
                noRepeatNGramIncludesPrompt: false)
        }

        var generatedText = ""
        for try await chunk in stream { generatedText += chunk }
        let cleaned = cleanEnhancedPrompt(generatedText)

        await g4.unload()
        Memory.clearCache()

        let elapsed = Date().timeIntervalSince(startTime)
        LTXDebug.log("Gemma 4 enhancement (\(isI2V ? "I2V" : "T2V")) took \(String(format: "%.1f", elapsed))s")
        guard !cleaned.isEmpty else {
            print("Enhancement produced an empty result; keeping the original prompt")
            return prompt
        }
        // A caption cut mid-sentence is worse than no enhancement: it encodes a
        // truncated scene as if it were the whole one. Say so loudly rather
        // than letting it through silently.
        if let last = cleaned.last, !".!?\"".contains(last) {
            print("⚠️ Enhanced prompt looks truncated (ends \"…\(cleaned.suffix(30))\") — "
                + "the reasoning likely consumed the token budget. Using it anyway; "
                + "re-run without --enhance-prompt if the result ignores part of the scene.")
        }
        print("Enhanced prompt (Gemma 4 E2B-it):\n\(cleaned)")
        fflush(stdout)
        return cleaned
    }

    /// Clean up a Gemma-enhanced prompt: strip control tokens and trailing noise
    private func cleanEnhancedPrompt(_ raw: String) -> String {
        var text = raw
        // Thinking mode streams reasoning in a channel before the answer; the
        // template's own strip_thinking macro does the same for prior turns.
        if let close = text.range(of: "<channel|>", options: .backwards) {
            text = String(text[close.upperBound...])
        }
        text = text.replacingOccurrences(of: "<|channel>thought", with: "")
        text = text.replacingOccurrences(of: "<|think|>", with: "")
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

    // MARK: - Vocoder selection

    /// Build the vocoder this checkpoint ships, falling back to the LTX-2 one.
    ///
    /// LTX-2.3 and LTX-2.5 both bundle a BigVGAN v2 generator plus a
    /// bandwidth-extension stage (667 + 557 tensors, byte-identical between the two
    /// generations) that outputs 48 kHz. The `Lightricks/LTX-2` standalone vocoder
    /// this package used to load is a different architecture entirely — it shares
    /// no key with them and stops at 24 kHz. It decodes the same latent space, so
    /// it produces plausible audio, which is exactly why the mismatch went
    /// unnoticed; it is kept only as a fallback for checkpoints that ship nothing
    /// better.
    private func loadVocoder(bundle: URL? = nil, audioVAEPath: URL, legacyPath: URL) throws -> any LTXVocoding {
        let resolvedBundle = bundle ?? (try? resolveCheckpointSync())?.audioBundle ?? audioVAEPath
        for candidate in [resolvedBundle, audioVAEPath] {
            do {
                let vocoder = try BigVGANWeightLoader.load(from: candidate)
                LTXDebug.log("Vocoder: BigVGAN + BWE from \(candidate.lastPathComponent)")
                return vocoder
            } catch {
                LTXDebug.log("Vocoder: \(candidate.lastPathComponent) has no BigVGAN (\(error))")
            }
        }

        // Loud on purpose: this fallback is the documented top-octave-loss
        // pitfall, and a debug-only log let it regress unnoticed once already.
        print("⚠️ BigVGAN vocoder unavailable for this checkpoint — falling back to the "
            + "legacy 24 kHz vocoder (audio loses the 12-16 kHz octave)")
        let weights = try LTXWeightLoader.loadVocoderWeights(from: legacyPath.path)
        let legacy = LTX2Vocoder()
        try LTXWeightLoader.applyVocoderWeights(weights, to: legacy)
        return legacy
    }

    /// The checkpoint paths resolved earlier in this load, without re-downloading.
    private func resolveCheckpointSync() throws -> LTXCheckpointPaths? {
        guard let cached = unifiedWeightsPathCache.cachedPath(for: model) else { return nil }
        let transformer = URL(fileURLWithPath: cached)
        switch model.weightsLayout {
        case .unified:
            return LTXCheckpointPaths(transformer: transformer, videoVAE: transformer)
        case .split:
            let directory = transformer.deletingLastPathComponent()
            guard let audio = model.family.sharedComponentFiles.first(where: { $0.kind == .audioVAE })
            else { return nil }
            return LTXCheckpointPaths(
                transformer: transformer,
                videoVAE: transformer,
                audioBundle: directory.appendingPathComponent(audio.filename))
        }
    }

    // MARK: - Auto Duration (LTX-2.5)

    /// Predict a clip length from the prompt, as LTX-2.5's `--auto-duration` does.
    ///
    /// Runs the text encoder and the 3.8 MB duration head — no diffusion — and
    /// returns a frame count already snapped to the `8k + 1` grid, so the result
    /// can be handed straight to ``LTXVideoGenerationConfig``.
    ///
    /// The clamp is a safety rail: an outlier prediction would otherwise request a
    /// degenerate or OOM-sized generation. `wasClamped` reports when it bit, so a
    /// caller can tell "the model asked for 3 s" from "the model asked for 90 s".
    ///
    /// - Throws: when the checkpoint predates LTX-2.5 — no earlier generation
    ///   ships a duration head, and guessing a length would defeat the purpose.
    public func predictFrameCount(
        for prompt: String,
        frameRate: Float = 24.0,
        minSeconds: Float = 1.0,
        maxSeconds: Float = 20.0
    ) async throws -> (frames: Int, seconds: Float, wasClamped: Bool) {
        guard model.family == .ltx25 else {
            throw LTXError.invalidConfiguration(
                "Auto duration needs a duration head, which ships from LTX-2.5 onward — "
                + "\(model.displayName) has none. Pass an explicit frame count.")
        }
        if !isGemmaLoaded || textEncoder == nil {
            try await loadModels(progressCallback: nil)
        }
        guard let textEncoder else {
            throw LTXError.modelNotLoaded("Text encoder not loaded")
        }

        let headPath = try await downloader.downloadDurationHead()
        let head = try LTXDurationHead.load(from: headPath)

        let (states, attentionMask) = try encodeHiddenStates(prompt)
        let encoded = try textEncoder.encodeFromHiddenStates(
            hiddenStates: states, attentionMask: attentionMask, paddingSide: "left")

        let result = try head.predictFrameCount(
            videoTokens: encoded.videoEncoding,
            audioTokens: encoded.audioEncoding,
            frameRate: frameRate,
            minSeconds: minSeconds,
            maxSeconds: maxSeconds)
        LTXDebug.log("Duration head: \(result.rawSeconds)s → \(result.frames) frames"
            + (result.wasClamped ? " (clamped)" : ""))
        return (result.frames, result.rawSeconds, result.wasClamped)
    }

    // MARK: - Download Helpers

    /// Download the spatial upscaler matching this pipeline's checkpoint generation.
    ///
    /// The module is architecturally identical across 2.3 and 2.5 — same 24 tensor
    /// patterns, same shapes — but the weights are not: a 2.5 latent is not a 2.3
    /// latent, so the two-stage pass must use its own generation's upscaler.
    /// - Returns: Path to the upscaler safetensors file
    public func downloadUpscalerWeights() async throws -> String {
        let url = try await downloader.downloadAuxiliaryModel(model.defaultSpatialUpscaler)
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
    /// - Returns: Number of LoRA layer-pairs fused (NOT the number of tensors
    ///   saved for unfusing — quantized layers save weight+scales+biases)
    @discardableResult
    public func fuseLoRA(
        from loraPath: String,
        scale: Float = 1.0
    ) throws -> Int {
        // Fusing a generic LoRA on top of the destructively-fused LipDub IC-LoRA
        // would capture LipDub-contaminated weights as the "originals" — a later
        // unfuseLoRA() would then restore corrupted weights as if pristine.
        try ensureNoLipDubLoRAFused(wouldCorrupt: "fuseLoRA (its unfuse originals)")
        LoRALoader.warnOnGenerationMismatch(
            loraPath: loraPath, checkpointVersion: model.family.checkpointModelVersion)
        let target = try getTransformerModule()
        let (originals, result) = try target.fuseLoRA(from: loraPath, scale: scale)
        // Store state for unfusing
        self.loraOriginalWeights = originals
        self.loraFusedPath = loraPath
        self.loraFusedScale = scale
        Memory.clearCache()
        return result.modifiedLayerCount
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
        gemmaEncoder = nil
        textEncoder = nil
        transformer = nil
        vaeDecoder = nil
        vaeEncoder = nil
        ltx2Transformer = nil
        lipdubFusion = nil
        audioVAE = nil
        vocoder = nil
        loraOriginalWeights = nil
        loraFusedPath = nil
        unifiedWeightsPathCache.clear()

        // Clear GPU cache from within the actor's isolation context
        // so ARC has already released the model refs above
        Memory.clearCache()
        eval([MLXArray]())

        LTXDebug.log("All models cleared and GPU cache flushed")
    }

    /// Clear only Gemma model (to save memory after encoding)
    public func clearGemma() {
        gemmaEncoder = nil
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
        // Step 1 & 2: Tokenize and run the Gemma forward pass for all 49 hidden states
        let (states, attentionMask) = try encodeHiddenStates(prompt)
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
    /// Run the prompt through the loaded Gemma stack.
    ///
    /// The single funnel every generation path goes through: five call sites used
    /// to repeat tokenize → forward → count-check inline, which meant Gemma 4
    /// support would have had to be written five times.
    private func encodeHiddenStates(
        _ prompt: String
    ) throws -> (states: [MLXArray], attentionMask: MLXArray) {
        guard let encoder = gemmaEncoder else {
            throw LTXError.modelNotLoaded("Gemma model not loaded. Call loadModels() first.")
        }
        return try encoder.encode(prompt: prompt, maxLength: textMaxLength)
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
