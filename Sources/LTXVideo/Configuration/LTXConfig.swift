// LTXConfig.swift - LTX-2 Model Configuration
// Copyright 2025

import Foundation

// MARK: - Model Selection

/// LTX-2.3 model variants
///
/// Uses unified safetensors format from `Lightricks/LTX-2.3`:
/// - `ltx-2.3-22b-distilled.safetensors` — Distilled model (transformer + VAE + connector)
/// - `ltx-2.3-22b-dev.safetensors` — Dev model (transformer + VAE + connector)
///
/// Audio VAE and vocoder are downloaded from `Lightricks/LTX-2` (shared components).
/// VLM Gemma is shared across all variants (`mlx-community/gemma-3-12b-it-qat-4bit`).
public enum LTXModel: String, CaseIterable, Sendable {
    /// LTX-2.3 Distilled - 8 steps, no CFG, two-stage pipeline
    case distilled = "distilled"

    /// LTX-2.3 Dev - 30 steps, CFG 3.0, required for LoRA training
    case dev = "dev"

    public var displayName: String {
        switch self {
        case .distilled: return "LTX-2.3 Distilled (~46GB)"
        case .dev: return "LTX-2.3 Dev (~46GB)"
        }
    }

    /// Default number of inference steps
    public var defaultSteps: Int {
        switch self {
        case .distilled: return 8
        case .dev: return 30
        }
    }

    /// Estimated VRAM usage in GB (with 3-phase loading)
    public var estimatedVRAM: Int {
        return 46
    }

    /// HuggingFace repository for this model
    public var huggingFaceRepo: String {
        return "Lightricks/LTX-2.3"
    }

    /// Unified weights filename (single file containing transformer, VAE, connector)
    public var unifiedWeightsFilename: String {
        switch self {
        case .distilled: return "ltx-2.3-22b-distilled.safetensors"
        case .dev: return "ltx-2.3-22b-dev.safetensors"
        }
    }

    /// Get the transformer configuration for this model
    public var transformerConfig: LTXTransformerConfig {
        return .ltx23
    }

    // MARK: - Model Capabilities

    /// Whether this model can be used for inference
    public var isForInference: Bool { true }

    /// Whether this model can be used for LoRA training
    public var isForTraining: Bool {
        switch self {
        case .dev: return true
        case .distilled: return false
        }
    }

    /// Whether the model requires authentication to download
    public var isGated: Bool { false }

    /// License identifier
    public var license: String { "lightricks-ltx-2.3" }

    /// Whether commercial use is allowed
    public var isCommercialUseAllowed: Bool { true }

    /// Estimated model size on disk in GB
    public var estimatedSizeGB: Float { 46.1 }

    /// Default CFG guidance scale
    public var defaultGuidance: Float {
        switch self {
        case .dev: return 3.0
        case .distilled: return 1.0
        }
    }

    /// Default STG scale
    public var defaultSTGScale: Float {
        switch self {
        case .dev: return 1.0
        case .distilled: return 0.0
        }
    }

    /// Short description of this model variant
    public var variantDescription: String {
        switch self {
        case .distilled: return "Fast inference (8 steps), two-stage pipeline"
        case .dev: return "Full quality (30 steps), CFG 3.0, LoRA training"
        }
    }

    /// Print a summary of all available models
    public static func printModelList() {
        print("Available LTX-2.3 Models:")
        print(String(repeating: "-", count: 80))
        let header = "Variant".padding(toLength: 12, withPad: " ", startingAt: 0)
            + "Inference".padding(toLength: 10, withPad: " ", startingAt: 0)
            + "Training".padding(toLength: 10, withPad: " ", startingAt: 0)
            + "Steps".padding(toLength: 8, withPad: " ", startingAt: 0)
            + "Size".padding(toLength: 10, withPad: " ", startingAt: 0)
            + "Gated".padding(toLength: 8, withPad: " ", startingAt: 0)
            + "License"
        print(header)
        print(String(repeating: "-", count: 80))
        for model in LTXModel.allCases {
            let line = model.rawValue.padding(toLength: 12, withPad: " ", startingAt: 0)
                + (model.isForInference ? "yes" : "no").padding(toLength: 10, withPad: " ", startingAt: 0)
                + (model.isForTraining ? "yes" : "no").padding(toLength: 10, withPad: " ", startingAt: 0)
                + "\(model.defaultSteps)".padding(toLength: 8, withPad: " ", startingAt: 0)
                + "\(String(format: "%.1f", model.estimatedSizeGB))GB".padding(toLength: 10, withPad: " ", startingAt: 0)
                + (model.isGated ? "yes" : "no").padding(toLength: 8, withPad: " ", startingAt: 0)
                + model.license
            print(line)
        }
        print(String(repeating: "-", count: 80))
        print("Shared components: VLM Gemma (~7.5GB), Audio VAE (~100MB), Vocoder (~106MB)")
    }
}

// MARK: - Transformer Configuration

/// Configuration for the LTX-2 diffusion transformer
public struct LTXTransformerConfig: Codable, Sendable {
    /// Number of transformer blocks
    public var numLayers: Int

    /// Number of attention heads
    public var numAttentionHeads: Int

    /// Dimension of each attention head
    public var attentionHeadDim: Int

    /// Inner dimension (numAttentionHeads * attentionHeadDim)
    public var innerDim: Int {
        numAttentionHeads * attentionHeadDim
    }

    /// Input/output channels from VAE (128 for LTX-2)
    public var inChannels: Int

    /// Output channels (same as input)
    public var outChannels: Int

    /// Cross-attention dimension (from text encoder)
    public var crossAttentionDim: Int

    /// Caption embedding dimension (3840 from Gemma3)
    public var captionChannels: Int

    /// RoPE theta value
    public var ropeTheta: Float

    /// Maximum positions for RoPE [time, height, width]
    public var maxPos: [Int]

    /// Timestep scale multiplier
    public var timestepScaleMultiplier: Int

    /// Layer norm epsilon
    public var normEps: Float

    /// LTX-2.3: enable gated attention (to_gate_logits per attention head)
    public var gatedAttention: Bool

    /// LTX-2.3: enable cross-attention AdaLN (prompt_adaln_single + prompt_scale_shift_table)
    public var crossAttentionAdaLN: Bool

    /// LTX-2.3 22B: caption projection is done in the connector, not the transformer.
    /// When true, both `captionProjection` and `audioCaptionProjection` are skipped.
    public var captionProjBeforeConnector: Bool

    public init(
        numLayers: Int = 48,
        numAttentionHeads: Int = 32,
        attentionHeadDim: Int = 128,
        inChannels: Int = 128,
        outChannels: Int = 128,
        crossAttentionDim: Int = 4096,
        captionChannels: Int = 3840,
        ropeTheta: Float = 10000.0,
        maxPos: [Int] = [20, 2048, 2048],
        timestepScaleMultiplier: Int = 1000,
        normEps: Float = 1e-6,
        audioNumAttentionHeads: Int = 32,
        audioAttentionHeadDim: Int = 64,
        audioInChannels: Int = 128,
        audioOutChannels: Int = 128,
        audioMaxPos: [Int] = [20],
        gatedAttention: Bool = false,
        crossAttentionAdaLN: Bool = false,
        captionProjBeforeConnector: Bool = false
    ) {
        self.numLayers = numLayers
        self.numAttentionHeads = numAttentionHeads
        self.attentionHeadDim = attentionHeadDim
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.crossAttentionDim = crossAttentionDim
        self.captionChannels = captionChannels
        self.ropeTheta = ropeTheta
        self.maxPos = maxPos
        self.timestepScaleMultiplier = timestepScaleMultiplier
        self.normEps = normEps
        self.audioNumAttentionHeads = audioNumAttentionHeads
        self.audioAttentionHeadDim = audioAttentionHeadDim
        self.audioInChannels = audioInChannels
        self.audioOutChannels = audioOutChannels
        self.audioMaxPos = audioMaxPos
        self.gatedAttention = gatedAttention
        self.crossAttentionAdaLN = crossAttentionAdaLN
        self.captionProjBeforeConnector = captionProjBeforeConnector
    }

    // MARK: - Audio Configuration

    /// Audio inner dimension (32 heads * 64 dim_head = 2048)
    public var audioNumAttentionHeads: Int
    /// Audio attention head dimension
    public var audioAttentionHeadDim: Int
    /// Audio inner dimension
    public var audioInnerDim: Int { audioNumAttentionHeads * audioAttentionHeadDim }
    /// Audio input/output channels (128, same as video)
    public var audioInChannels: Int
    /// Audio output channels
    public var audioOutChannels: Int
    /// Audio cross-attention dimension
    public var audioCrossAttentionDim: Int { audioInnerDim }
    /// Audio RoPE max positions
    public var audioMaxPos: [Int]

    /// Default LTX-2 configuration (legacy, gated attention off)
    public static let `default` = LTXTransformerConfig(
        gatedAttention: false,
        crossAttentionAdaLN: false
    )

    /// LTX-2.3 configuration (gated attention + cross-attention AdaLN, no caption projection)
    public static let ltx23 = LTXTransformerConfig(
        captionChannels: 4096,
        gatedAttention: true,
        crossAttentionAdaLN: true,
        captionProjBeforeConnector: true
    )
}

extension LTXTransformerConfig: CustomStringConvertible {
    public var description: String {
        """
        LTXTransformerConfig(
            layers: \(numLayers),
            heads: \(numAttentionHeads) × \(attentionHeadDim) = \(innerDim),
            caption: \(captionChannels) → \(crossAttentionDim),
            rope: θ=\(ropeTheta), maxPos=\(maxPos)
        )
        """
    }
}

// MARK: - Video Generation Configuration

/// Parameters controlling video generation output.
///
/// Configure resolution, frame count, and features like two-stage upscaling.
///
/// ## Constraints
/// - **Width/Height**: Must be divisible by 64 (for two-stage)
/// - **Frame count**: Must be `8n + 1` (9, 17, 25, ..., 241)
///
/// ## Example
/// ```swift
/// let config = LTXVideoGenerationConfig(
///     width: 768,
///     height: 512,
///     numFrames: 121,    // 5 seconds at 24fps
///     seed: 42
/// )
/// try config.validate()
/// ```
public struct LTXVideoGenerationConfig: Sendable {
    /// Video width in pixels (must be divisible by 64)
    public var width: Int

    /// Video height in pixels (must be divisible by 64)
    public var height: Int

    /// Number of frames (must be 8n + 1)
    public var numFrames: Int

    /// Number of inference steps
    public var numSteps: Int

    /// Random seed (nil for random)
    public var seed: UInt64?

    /// Whether to enhance the prompt using Gemma before generation.
    public var enhancePrompt: Bool

    /// Path to input image for image-to-video generation.
    /// nil = text-to-video (default), non-nil = image-to-video.
    public var imagePath: String?

    /// Noise scale for image conditioning. Adds quadratically-decreasing noise to the
    /// conditioned frame to allow smoother motion transitions. 0.0 = disabled (matches Diffusers default),
    /// 0.15 = optional for natural motion.
    public var imageCondNoiseScale: Float

    /// Source video path for retake (video-to-video) mode. nil = generate from scratch.
    public var videoPath: String?

    /// Retake strength: how much of the source to change.
    /// 0.0 = keep original (no change), 1.0 = full regeneration (source ignored).
    /// Controls where the truncated sigma schedule starts. Default: 0.8.
    public var retakeStrength: Float

    /// Start time (seconds) for partial retake. nil = retake all frames.
    /// Only the region [retakeStartTime, retakeEndTime] is regenerated; outside frames are kept.
    public var retakeStartTime: Float?

    /// End time (seconds) for partial retake. nil = retake all frames.
    public var retakeEndTime: Float?

    /// Whether to regenerate audio during retake (requires audio models loaded).
    /// When false (default), source audio is preserved via passthrough.
    /// When true, audio is denoised alongside video using LTX2Transformer.
    public var regenerateAudio: Bool

    /// Optional list of keyframes for multi-frame interpolation (first / middle / last frame).
    /// When non-empty, generation is constrained to pass through each keyframe at its
    /// specified pixel frame index. Mutually exclusive with `videoPath` (retake).
    /// If both `imagePath` and `keyframes` are provided, `keyframes` takes precedence.
    public var keyframes: [KeyframeInput]

    public init(
        width: Int = 704,
        height: Int = 480,
        numFrames: Int = 121,
        numSteps: Int = 8,
        seed: UInt64? = nil,
        enhancePrompt: Bool = false,
        imagePath: String? = nil,
        imageCondNoiseScale: Float = 0.0,
        videoPath: String? = nil,
        retakeStrength: Float = 0.8,
        retakeStartTime: Float? = nil,
        retakeEndTime: Float? = nil,
        regenerateAudio: Bool = false,
        keyframes: [KeyframeInput] = []
    ) {
        self.width = width
        self.height = height
        self.numFrames = numFrames
        self.numSteps = numSteps
        self.seed = seed
        self.enhancePrompt = enhancePrompt
        self.imagePath = imagePath
        self.imageCondNoiseScale = imageCondNoiseScale
        self.videoPath = videoPath
        self.retakeStrength = retakeStrength
        self.retakeStartTime = retakeStartTime
        self.retakeEndTime = retakeEndTime
        self.regenerateAudio = regenerateAudio
        self.keyframes = keyframes
    }

    /// Convenience initializer that applies model-specific defaults.
    public init(
        model: LTXModel,
        width: Int = 704,
        height: Int = 480,
        numFrames: Int = 121,
        numSteps: Int? = nil,
        seed: UInt64? = nil,
        enhancePrompt: Bool = false,
        imagePath: String? = nil,
        imageCondNoiseScale: Float = 0.0,
        videoPath: String? = nil,
        retakeStrength: Float = 0.8,
        retakeStartTime: Float? = nil,
        retakeEndTime: Float? = nil,
        regenerateAudio: Bool = false,
        keyframes: [KeyframeInput] = []
    ) {
        self.width = width
        self.height = height
        self.numFrames = numFrames
        self.numSteps = numSteps ?? model.defaultSteps
        self.seed = seed
        self.enhancePrompt = enhancePrompt
        self.imagePath = imagePath
        self.imageCondNoiseScale = imageCondNoiseScale
        self.videoPath = videoPath
        self.retakeStrength = retakeStrength
        self.retakeStartTime = retakeStartTime
        self.retakeEndTime = retakeEndTime
        self.regenerateAudio = regenerateAudio
        self.keyframes = keyframes
    }

    /// Validate the configuration
    public func validate() throws {
        // Width must be divisible by 64
        guard width % 64 == 0 else {
            throw LTXError.invalidConfiguration("Width must be divisible by 64, got \(width)")
        }

        // Height must be divisible by 64
        guard height % 64 == 0 else {
            throw LTXError.invalidConfiguration("Height must be divisible by 64, got \(height)")
        }

        // Frames must be 8n + 1
        guard (numFrames - 1) % 8 == 0 else {
            throw LTXError.invalidConfiguration("Number of frames must be 8n + 1 (e.g., 9, 17, 25, ..., 121), got \(numFrames)")
        }

        // Reasonable bounds
        guard width >= 64 && width <= 2048 else {
            throw LTXError.invalidConfiguration("Width must be between 64 and 2048, got \(width)")
        }

        guard height >= 64 && height <= 2048 else {
            throw LTXError.invalidConfiguration("Height must be between 64 and 2048, got \(height)")
        }

        guard numFrames >= 9 && numFrames <= 257 else {
            throw LTXError.invalidConfiguration("Number of frames must be between 9 and 257, got \(numFrames)")
        }

        guard numSteps >= 1 && numSteps <= 100 else {
            throw LTXError.invalidConfiguration("Number of steps must be between 1 and 100, got \(numSteps)")
        }

        // Validate image path exists if provided
        if let imagePath = imagePath {
            guard FileManager.default.fileExists(atPath: imagePath) else {
                throw LTXError.fileNotFound("Input image not found: \(imagePath)")
            }
        }

        // Validate video path exists if provided
        if let videoPath = videoPath {
            guard FileManager.default.fileExists(atPath: videoPath) else {
                throw LTXError.fileNotFound("Input video not found: \(videoPath)")
            }
            guard retakeStrength > 0.0 && retakeStrength <= 1.0 else {
                throw LTXError.invalidConfiguration("Retake strength must be in (0.0, 1.0], got \(retakeStrength)")
            }
        }

        // Validate keyframes
        if !keyframes.isEmpty {
            guard videoPath == nil else {
                throw LTXError.invalidConfiguration("Keyframes cannot be combined with retake (videoPath)")
            }
            try validateKeyframes(keyframes, numFrames: numFrames)
        }
    }

    /// Latent dimensions (after VAE encoding)
    public var latentWidth: Int { width / 32 }
    public var latentHeight: Int { height / 32 }
    public var latentFrames: Int { (numFrames - 1) / 8 + 1 }

    /// Total number of latent tokens
    public var numLatentTokens: Int { latentFrames * latentHeight * latentWidth }
}

// MARK: - Spatio-Temporal Scale Factors
// Note: SpatioTemporalScaleFactors is defined in Pipeline/VideoLatentShape.swift
