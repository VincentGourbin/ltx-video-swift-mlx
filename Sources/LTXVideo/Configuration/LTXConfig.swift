// LTXConfig.swift - LTX-2 Model Configuration
// Copyright 2025

import Foundation

// MARK: - Model Selection

/// LTX-2 model variants
///
/// Uses HuggingFace Diffusers format with per-component weight files:
/// - `transformer/` — sharded safetensors for the diffusion transformer
/// - `vae/` — standalone VAE decoder weights
/// - `connectors/` — text encoder connector weights
/// - `text_encoder/` — Gemma 3 12B sharded weights
/// - `tokenizer/` — Gemma tokenizer files
public enum LTXModel: String, CaseIterable, Sendable {
    /// LTX-2 Dev - Full model, 40 steps, CFG guidance, highest quality
    case dev = "dev"

    /// LTX-2 Distilled - Faster model, 8 steps, no CFG
    case distilled = "distilled"

    public var displayName: String {
        switch self {
        case .dev: return "LTX-2 Dev (~25GB)"
        case .distilled: return "LTX-2 Distilled (~16GB)"
        }
    }

    /// Whether this model uses the distilled sigma schedule
    public var isDistilled: Bool {
        self == .distilled
    }

    /// Default number of inference steps
    public var defaultSteps: Int {
        switch self {
        case .dev: return 40
        case .distilled: return 8
        }
    }

    /// Default guidance scale
    public var defaultGuidance: Float {
        switch self {
        case .dev: return 4.0
        case .distilled: return 1.0
        }
    }

    /// Estimated VRAM usage in GB (with 3-phase loading)
    public var estimatedVRAM: Int {
        switch self {
        case .dev: return 25
        case .distilled: return 16
        }
    }

    /// HuggingFace repository for this model
    /// Both variants use the official Lightricks repo — only the unified weights filename differs.
    public var huggingFaceRepo: String {
        return "Lightricks/LTX-2"
    }

    /// Unified weights filename (single file containing transformer, VAE, connector)
    /// Used as a fallback when per-component files aren't available.
    public var unifiedWeightsFilename: String {
        switch self {
        case .dev: return "ltx-2-19b-dev.safetensors"
        case .distilled: return "ltx-2-19b-distilled.safetensors"
        }
    }

    /// Get the transformer configuration for this model
    public var transformerConfig: LTXTransformerConfig {
        return .default
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
        normEps: Float = 1e-6
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
    }

    /// Default LTX-2 configuration
    public static let `default` = LTXTransformerConfig()
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
/// Configure resolution, frame count, inference steps, guidance, and advanced
/// features like STG (Spatio-Temporal Guidance) and two-stage upscaling.
///
/// ## Constraints
/// - **Width/Height**: Must be divisible by 32 (or 64 for two-stage)
/// - **Frame count**: Must be `8n + 1` (9, 17, 25, ..., 241)
/// - **CFG scale**: 1.0 disables classifier-free guidance (required for distilled)
///
/// ## Example
/// ```swift
/// let config = LTXVideoGenerationConfig(
///     width: 768,
///     height: 512,
///     numFrames: 121,    // 5 seconds at 24fps
///     numSteps: 8,
///     cfgScale: 1.0,
///     seed: 42
/// )
/// try config.validate()
/// ```
public struct LTXVideoGenerationConfig: Sendable {
    /// Video width in pixels (must be divisible by 32)
    public var width: Int

    /// Video height in pixels (must be divisible by 32)
    public var height: Int

    /// Number of frames (must be 8n + 1)
    public var numFrames: Int

    /// Number of inference steps
    public var numSteps: Int

    /// Classifier-free guidance scale
    public var cfgScale: Float

    /// Random seed (nil for random)
    public var seed: UInt64?

    /// Negative prompt for CFG
    public var negativePrompt: String?

    /// Guidance rescale factor (phi). 0.0 = disabled, 0.7 = recommended with CFG.
    /// Rescales CFG output to match the standard deviation of the conditional output,
    /// reducing overexposure artifacts.
    public var guidanceRescale: Float

    /// Cross-attention scaling factor. 1.0 = no change, >1.0 = stronger prompt adherence.
    /// Applied to the output of cross-attention layers in the transformer.
    public var crossAttentionScale: Float

    /// GE velocity correction gamma. 0.0 = disabled.
    /// Applies momentum on the velocity prediction: v' = gamma * (v - v_prev) + v_prev
    public var geGamma: Float

    /// STG (Spatio-Temporal Guidance) scale. 0.0 = disabled.
    /// Higher values improve spatial/temporal coherence at the cost of ~2x denoise time.
    public var stgScale: Float

    /// Transformer block indices where STG skips self-attention.
    public var stgBlocks: [Int]

    /// Whether to use two-stage generation (half-res + upscale + refine).
    public var twoStage: Bool

    /// Whether to enhance the prompt using Gemma before generation.
    public var enhancePrompt: Bool

    public init(
        width: Int = 704,
        height: Int = 480,
        numFrames: Int = 121,
        numSteps: Int = 8,
        cfgScale: Float = 1.0,
        seed: UInt64? = nil,
        negativePrompt: String? = nil,
        guidanceRescale: Float = 0.0,
        crossAttentionScale: Float = 1.0,
        geGamma: Float = 0.0,
        stgScale: Float = 0.0,
        stgBlocks: [Int] = [29],
        twoStage: Bool = false,
        enhancePrompt: Bool = false
    ) {
        self.width = width
        self.height = height
        self.numFrames = numFrames
        self.numSteps = numSteps
        self.cfgScale = cfgScale
        self.seed = seed
        self.negativePrompt = negativePrompt
        self.guidanceRescale = guidanceRescale
        self.crossAttentionScale = crossAttentionScale
        self.geGamma = geGamma
        self.stgScale = stgScale
        self.stgBlocks = stgBlocks
        self.twoStage = twoStage
        self.enhancePrompt = enhancePrompt
    }

    /// Validate the configuration
    public func validate() throws {
        // Width must be divisible by 32
        guard width % 32 == 0 else {
            throw LTXError.invalidConfiguration("Width must be divisible by 32, got \(width)")
        }

        // Height must be divisible by 32
        guard height % 32 == 0 else {
            throw LTXError.invalidConfiguration("Height must be divisible by 32, got \(height)")
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

        guard cfgScale >= 1.0 && cfgScale <= 20.0 else {
            throw LTXError.invalidConfiguration("CFG scale must be between 1.0 and 20.0, got \(cfgScale)")
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
