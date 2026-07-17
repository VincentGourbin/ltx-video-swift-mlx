// LoRATrainingConfig.swift - Training Configuration
// Copyright 2026

import Foundation

/// Configuration for LoRA training
public struct LoRATrainingConfig: Sendable {
    /// LoRA rank (low-rank dimension)
    public var rank: Int

    /// LoRA alpha (scaling factor). Default: same as rank
    public var alpha: Float

    /// Learning rate
    public var learningRate: Float

    /// AdamW weight decay
    public var weightDecay: Float

    /// Maximum training steps
    public var maxSteps: Int

    /// Save checkpoint every N steps
    public var saveEvery: Int

    /// Video width (must be divisible by 32)
    public var width: Int

    /// Video height (must be divisible by 32)
    public var height: Int

    /// Number of frames (must be 8n+1)
    public var numFrames: Int

    /// Model variant to train on
    public var model: String

    /// Whether to include audio LoRA targets (not yet supported)
    public var includeAudio: Bool

    /// Audio loss weight relative to video loss
    public var audioLossWeight: Float

    /// Whether to include FFN layers in LoRA targets
    public var includeFFN: Bool

    /// Transformer quantization for QLoRA (bf16, qint8, int4)
    public var transformerQuant: String

    /// Number of last transformer blocks to apply LoRA to (0 = all blocks)
    public var loraBlocks: Int

    /// Gradient accumulation steps (effective batch size = gradientAccumulationSteps)
    public var gradientAccumulationSteps: Int

    /// Maximum gradient norm for clipping (0 = disabled)
    public var maxGradNorm: Float

    /// LR warmup steps
    public var warmupSteps: Int

    /// LR schedule applied after warmup. Uses the official MLXOptimizers
    /// schedulers. Single source of truth for the valid values — the training
    /// loop switches over this exhaustively, so adding a case cannot silently
    /// fall through to the wrong schedule.
    public enum LRSchedule: String, Sendable, CaseIterable {
        /// Cosine decay from peak LR to 10% of peak over the remaining steps —
        /// the LoRA fine-tuning standard, and the default since July 2026.
        case cosine
        /// Flat peak LR after warmup — the pre-July-2026 behavior. Note that
        /// `constant` does NOT reproduce pre-July-2026 runs bit-for-bit:
        /// AdamW bias correction is now always on (see TrainingLoop).
        case constant
    }
    public var lrSchedule: LRSchedule

    /// Trigger word to prepend to all captions (e.g., "CAKEIFY")
    public var triggerWord: String?

    /// Seed for reproducibility
    public var seed: UInt64?

    /// HuggingFace token
    public var hfToken: String?

    /// Custom models directory
    public var modelsDir: String?

    /// Gemma model path
    public var gemmaPath: String?

    /// LTX weights path
    public var ltxWeightsPath: String?

    /// Text prompt used to generate a preview video after each checkpoint.
    /// Set to `nil` to disable preview generation (default).
    public var previewPrompt: String?

    /// Optional image path for I2V preview generation at checkpoints.
    public var previewImage: String?

    /// Whether to generate a preview video at each checkpoint.
    /// Automatically `true` when `previewPrompt` is non-nil.
    public var generatePreview: Bool { previewPrompt != nil }

    // MARK: - Memory Presets (for 22B model)

    /// Preset for 32GB systems (int4 quantization, minimal resolution)
    public static let compact = LoRATrainingConfig(
        rank: 16, alpha: 16, learningRate: 2e-4,
        width: 256, height: 256, numFrames: 9,
        includeFFN: false, transformerQuant: "int4"
    )

    /// Preset for 64GB systems (int4, moderate resolution)
    public static let balanced = LoRATrainingConfig(
        rank: 32, alpha: 32, learningRate: 2e-4,
        width: 384, height: 384, numFrames: 9,
        includeFFN: false, transformerQuant: "int4"
    )

    /// Preset for 96GB systems (int4, higher resolution)
    public static let quality = LoRATrainingConfig(
        rank: 64, alpha: 64, learningRate: 2e-4,
        width: 512, height: 512, numFrames: 9,
        includeFFN: false, transformerQuant: "int4"
    )

    /// Preset for 192GB+ systems (bf16 full precision)
    public static let max = LoRATrainingConfig(
        rank: 128, alpha: 128, learningRate: 2e-4,
        width: 512, height: 512, numFrames: 9,
        includeFFN: false, transformerQuant: "bf16"
    )

    public init(
        rank: Int = 16,
        alpha: Float? = nil,
        learningRate: Float = 2e-4,
        weightDecay: Float = 0.01,
        maxSteps: Int = 2000,
        saveEvery: Int = 250,
        width: Int = 256,
        height: Int = 256,
        numFrames: Int = 9,
        model: String = "dev",
        includeAudio: Bool = false,
        audioLossWeight: Float = 0.5,
        includeFFN: Bool = false,
        // qint8 by default: bf16 training peaks at 84+ GB and swaps on 96 GB
        // machines, while qint8 halves the peak with near-exact loss parity
        // (see docs/knowledge/decisions/qlora-training-default.md).
        transformerQuant: String = "qint8",
        loraBlocks: Int = 0,
        gradientAccumulationSteps: Int = 1,
        maxGradNorm: Float = 1.0,
        warmupSteps: Int = 100,
        lrSchedule: LRSchedule = .cosine,
        triggerWord: String? = nil,
        seed: UInt64? = nil,
        hfToken: String? = nil,
        modelsDir: String? = nil,
        gemmaPath: String? = nil,
        ltxWeightsPath: String? = nil,
        previewPrompt: String? = nil,
        previewImage: String? = nil
    ) {
        self.rank = rank
        self.alpha = alpha ?? Float(rank)
        self.learningRate = learningRate
        self.weightDecay = weightDecay
        self.maxSteps = maxSteps
        self.saveEvery = saveEvery
        self.width = width
        self.height = height
        self.numFrames = numFrames
        self.model = model
        self.includeAudio = includeAudio
        self.audioLossWeight = audioLossWeight
        self.includeFFN = includeFFN
        self.transformerQuant = transformerQuant
        self.loraBlocks = loraBlocks
        self.gradientAccumulationSteps = gradientAccumulationSteps
        self.maxGradNorm = maxGradNorm
        self.warmupSteps = warmupSteps
        self.lrSchedule = lrSchedule
        self.triggerWord = triggerWord
        self.seed = seed
        self.hfToken = hfToken
        self.modelsDir = modelsDir
        self.gemmaPath = gemmaPath
        self.ltxWeightsPath = ltxWeightsPath
        self.previewPrompt = previewPrompt
        self.previewImage = previewImage
    }

    /// Validate configuration
    public func validate() throws {
        guard (numFrames - 1) % 8 == 0 else {
            throw TrainingError.invalidConfig("Frame count must be 8n+1, got \(numFrames)")
        }
        guard width % 32 == 0 && height % 32 == 0 else {
            throw TrainingError.invalidConfig("Width and height must be divisible by 32, got \(width)x\(height)")
        }
        guard rank > 0 && rank <= 512 else {
            throw TrainingError.invalidConfig("Rank must be between 1 and 512, got \(rank)")
        }
        guard learningRate > 0 && learningRate < 1 else {
            throw TrainingError.invalidConfig("Learning rate must be between 0 and 1, got \(learningRate)")
        }
        guard gradientAccumulationSteps >= 1 else {
            throw TrainingError.invalidConfig("Gradient accumulation steps must be >= 1, got \(gradientAccumulationSteps)")
        }
        guard warmupSteps >= 0 && warmupSteps < maxSteps else {
            throw TrainingError.invalidConfig(
                "Warmup steps must be in 0..<maxSteps (\(maxSteps)), got \(warmupSteps) — " +
                "warmup >= maxSteps would spend the whole run warming up and never reach the schedule")
        }
        // Training always quantizes the bf16 unified weights on the fly, which
        // upstream mlx-swift does not support for nvfp4/mxfp8 (mlx-swift #285).
        guard !["nvfp4", "mxfp8"].contains(transformerQuant) else {
            throw TrainingError.invalidConfig(
                "Transformer quantization '\(transformerQuant)' is not supported for training " +
                "(on-the-fly quantization unavailable upstream, mlx-swift #285). Use bf16, qint8 or int4.")
        }
    }
}

/// Training-specific errors
public enum TrainingError: Error, LocalizedError {
    case invalidConfig(String)
    case datasetError(String)
    case modelError(String)
    case checkpointError(String)

    public var errorDescription: String? {
        switch self {
        case .invalidConfig(let msg): return "Invalid training config: \(msg)"
        case .datasetError(let msg): return "Dataset error: \(msg)"
        case .modelError(let msg): return "Model error: \(msg)"
        case .checkpointError(let msg): return "Checkpoint error: \(msg)"
        }
    }
}
