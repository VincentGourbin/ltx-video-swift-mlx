// LTXConfig.swift - LTX-2 Model Configuration
// Copyright 2025

import Foundation

// MARK: - Model Selection

/// LTX-2 model variants
public enum LTXModel: String, CaseIterable, Sendable {
    /// LTX-2 Dev - Full model, 50 steps, highest quality
    case dev = "dev"

    /// LTX-2 Distilled - Faster model, 8 steps
    case distilled = "distilled"

    /// LTX-2 Distilled FP8 - Smallest model, 8 steps, FP8 quantized
    case distilledFP8 = "distilledFP8"

    public var displayName: String {
        switch self {
        case .dev: return "LTX-2 Dev (~25GB)"
        case .distilled: return "LTX-2 Distilled (~16GB)"
        case .distilledFP8: return "LTX-2 Distilled FP8 (~12GB)"
        }
    }

    /// Whether this model uses the distilled sigma schedule
    public var isDistilled: Bool {
        switch self {
        case .dev: return false
        case .distilled, .distilledFP8: return true
        }
    }

    /// Default number of inference steps
    public var defaultSteps: Int {
        switch self {
        case .dev: return 50
        case .distilled, .distilledFP8: return 8
        }
    }

    /// Default guidance scale
    public var defaultGuidance: Float {
        switch self {
        case .dev: return 7.5
        case .distilled, .distilledFP8: return 1.0
        }
    }

    /// Estimated VRAM usage in GB
    public var estimatedVRAM: Int {
        switch self {
        case .dev: return 25
        case .distilled: return 16
        case .distilledFP8: return 12
        }
    }

    /// HuggingFace repository for this model
    public var huggingFaceRepo: String {
        return "Lightricks/LTX-2"
    }

    /// Safetensors file name for this model variant
    public var transformerFilename: String {
        switch self {
        case .dev: return "ltx-2-19b-dev.safetensors"
        case .distilled: return "ltx-2-19b-distilled.safetensors"
        case .distilledFP8: return "ltx-2-19b-distilled-fp8.safetensors"
        }
    }

    /// Unified weights filename (contains all components: transformer, VAE, text encoder)
    /// Same file as transformerFilename — it contains everything.
    public var weightsFilename: String { transformerFilename }

    /// Whether this model uses FP8 quantized weights
    public var isFP8: Bool { self == .distilledFP8 }

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

/// Configuration for video generation
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

    public init(
        width: Int = 704,
        height: Int = 480,
        numFrames: Int = 121,
        numSteps: Int = 8,
        cfgScale: Float = 1.0,
        seed: UInt64? = nil,
        negativePrompt: String? = nil
    ) {
        self.width = width
        self.height = height
        self.numFrames = numFrames
        self.numSteps = numSteps
        self.cfgScale = cfgScale
        self.seed = seed
        self.negativePrompt = negativePrompt
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
