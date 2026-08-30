// LTXConfig.swift - LTX-2 Model Configuration
// Copyright 2025

import Foundation

// MARK: - Model Selection

/// LTX model variants across the 2.3 and 2.5 generations.
///
/// **LTX-2.3** (`Lightricks/LTX-2.3`, open repo) uses a unified safetensors file:
/// - `ltx-2.3-22b-distilled.safetensors` — Distilled model (transformer + VAE + connector)
/// - `ltx-2.3-22b-dev.safetensors` — Dev model (transformer + VAE + connector)
///
/// Audio VAE and vocoder are downloaded from `Lightricks/LTX-2` (shared components).
/// The text encoder is stock Gemma 3 12B (`mlx-community/gemma-3-12b-it-qat-4bit`).
///
/// **LTX-2.5** (`Lightricks/LTX-2.5`, gated) ships one file per component and swaps
/// the text encoder for an LTX-specific Gemma 4 12B derivative bundled with the
/// checkpoint. Text- and image-to-video run; the diffusion decoder, the temporal
/// upscaler and the pixel upscaler IC-LoRA are not implemented — see
/// ``LTXModel/support``.
///
/// Licensing, gating and packaging live in `LTXModelCatalog.swift`.
public enum LTXModel: String, CaseIterable, Sendable {
    /// LTX-2.3 Distilled - 8 steps, no CFG, two-stage pipeline
    case distilled = "distilled"

    /// LTX-2.3 Dev - 30 steps, CFG 3.0, required for LoRA training
    case dev = "dev"

    /// LTX-2.5 Distilled - 8 steps, no CFG, gated repo, Gemma 4 text encoder
    case v25Distilled = "2.5-distilled"

    /// LTX-2.5 Dev - 30 steps, CFG 3.0, gated repo, Gemma 4 text encoder
    case v25Dev = "2.5-dev"

    public var displayName: String {
        switch self {
        case .distilled: return "LTX-2.3 Distilled (~46GB)"
        case .dev: return "LTX-2.3 Dev (~46GB)"
        case .v25Distilled: return "LTX-2.5 Distilled (~70GB)"
        case .v25Dev: return "LTX-2.5 Dev (~70GB)"
        }
    }

    /// Default number of inference steps
    public var defaultSteps: Int {
        switch self {
        case .distilled, .v25Distilled: return 8
        case .dev, .v25Dev: return 30
        }
    }

    /// Estimated VRAM usage in GB (with 3-phase loading)
    public var estimatedVRAM: Int {
        switch self {
        case .distilled, .dev: return 46
        // 2.5 is split: 42 GB transformer + 26 GB bf16 Gemma 4 encoder, and unlike
        // 2.3 there is no community 4-bit encoder to fall back on.
        case .v25Distilled, .v25Dev: return 70
        }
    }

    /// HuggingFace repository for this model
    public var huggingFaceRepo: String {
        family.huggingFaceRepo
    }

    /// Main weights filename.
    ///
    /// For LTX-2.3 (``LTXWeightsLayout/unified``) this single file holds transformer,
    /// VAE and connector. For LTX-2.5 (``LTXWeightsLayout/split``) it is the transformer
    /// shard only — the other components are listed in ``componentFiles``.
    public var unifiedWeightsFilename: String {
        switch self {
        case .distilled: return "ltx-2.3-22b-distilled.safetensors"
        case .dev: return "ltx-2.3-22b-dev.safetensors"
        case .v25Distilled:
            return "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
        case .v25Dev:
            return "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors"
        }
    }

    /// Get the transformer configuration for this model
    public var transformerConfig: LTXTransformerConfig {
        switch family {
        case .ltx23: return .ltx23
        case .ltx25: return .ltx25
        }
    }

    // MARK: - Model Capabilities

    /// Whether this model can be used for inference
    public var isForInference: Bool { true }

    /// Whether this model can be used for LoRA training
    public var isForTraining: Bool {
        switch self {
        case .dev, .v25Dev: return true
        case .distilled, .v25Distilled: return false
        }
    }

    /// Whether the model requires accepting a licence on HuggingFace (and a token) to download
    public var isGated: Bool { gating.requiresToken }

    /// License identifier
    public var license: String { licenseInfo.id }

    /// Whether commercial use is allowed — under the conditions in ``commercialUseSummary``
    public var isCommercialUseAllowed: Bool { licenseInfo.allowsCommercialUse }

    /// Estimated model size on disk in GB
    public var estimatedSizeGB: Float {
        switch self {
        case .distilled, .dev: return 46.1
        // Transformer 42.0 + Gemma 4 encoder 26.3 + video VAE 1.45 + audio VAE 0.36.
        case .v25Distilled, .v25Dev: return 70.1
        }
    }

    /// Default CFG guidance scale
    public var defaultGuidance: Float {
        switch self {
        case .dev, .v25Dev: return 3.0
        case .distilled, .v25Distilled: return 1.0
        }
    }

    /// Default STG scale
    public var defaultSTGScale: Float {
        switch self {
        case .dev, .v25Dev: return 1.0
        case .distilled, .v25Distilled: return 0.0
        }
    }

    /// Short description of this model variant
    public var variantDescription: String {
        switch self {
        case .distilled: return "Fast inference (8 steps), two-stage pipeline"
        case .dev: return "Full quality (30 steps), CFG 3.0, LoRA training"
        case .v25Distilled: return "LTX-2.5 fast inference (8 steps), multishot, Gemma 4"
        case .v25Dev: return "LTX-2.5 full quality (30 steps), CFG 3.0, LoRA training"
        }
    }

    /// Print a summary of all available models, their licensing and their status.
    public static func printModelList() {
        print("Available LTX models:")
        print(String(repeating: "-", count: 104))
        let header = "Variant".padding(toLength: 16, withPad: " ", startingAt: 0)
            + "Status".padding(toLength: 9, withPad: " ", startingAt: 0)
            + "Infer".padding(toLength: 7, withPad: " ", startingAt: 0)
            + "Train".padding(toLength: 7, withPad: " ", startingAt: 0)
            + "Steps".padding(toLength: 7, withPad: " ", startingAt: 0)
            + "Size".padding(toLength: 9, withPad: " ", startingAt: 0)
            + "Gated".padding(toLength: 7, withPad: " ", startingAt: 0)
            + "Text encoder".padding(toLength: 20, withPad: " ", startingAt: 0)
            + "License"
        print(header)
        print(String(repeating: "-", count: 104))
        for model in LTXModel.allCases {
            let line = model.rawValue.padding(toLength: 16, withPad: " ", startingAt: 0)
                + model.support.label.padding(toLength: 9, withPad: " ", startingAt: 0)
                + (model.isForInference ? "yes" : "no").padding(toLength: 7, withPad: " ", startingAt: 0)
                + (model.isForTraining ? "yes" : "no").padding(toLength: 7, withPad: " ", startingAt: 0)
                + "\(model.defaultSteps)".padding(toLength: 7, withPad: " ", startingAt: 0)
                + "\(String(format: "%.0f", model.estimatedSizeGB))GB".padding(toLength: 9, withPad: " ", startingAt: 0)
                + (model.isGated ? "yes" : "no").padding(toLength: 7, withPad: " ", startingAt: 0)
                + model.textEncoder.rawValue.padding(toLength: 20, withPad: " ", startingAt: 0)
                + model.licenseName
            print(line)
        }
        print(String(repeating: "-", count: 104))
        print("Status: ready = runnable today; catalog = published weights, not implemented here yet.")
        for model in LTXModel.allCases {
            if case .notImplemented(let reason) = model.support {
                print("  \(model.rawValue): \(reason)")
            }
        }
        print()

        print("Auxiliary models (upscalers, LoRAs):")
        print(String(repeating: "-", count: 104))
        let auxHeader = "Name".padding(toLength: 34, withPad: " ", startingAt: 0)
            + "Status".padding(toLength: 9, withPad: " ", startingAt: 0)
            + "Size".padding(toLength: 9, withPad: " ", startingAt: 0)
            + "Gated".padding(toLength: 7, withPad: " ", startingAt: 0)
            + "Repository"
        print(auxHeader)
        print(String(repeating: "-", count: 104))
        for aux in LTXAuxiliaryModel.allCases {
            let line = aux.rawValue.padding(toLength: 34, withPad: " ", startingAt: 0)
                + aux.support.label.padding(toLength: 9, withPad: " ", startingAt: 0)
                + "\(String(format: "%.1f", aux.approximateSizeGB))GB".padding(toLength: 9, withPad: " ", startingAt: 0)
                + (aux.gating.requiresToken ? "yes" : "no").padding(toLength: 7, withPad: " ", startingAt: 0)
                + aux.huggingFaceRepo
            print(line)
        }
        print(String(repeating: "-", count: 104))
        print()
        print("Gated repositories require accepting the licence on the model page, then a")
        print("HuggingFace token (--hf-token, $HF_TOKEN, or ~/.cache/huggingface/token).")
        print("License: \(LTXLicense.ltx2Community.name) — \(LTXLicense.ltx2Community.url)")
        print("  \(LTXLicense.ltx2Community.summary)")
        print("Shared components (LTX-2.3): VLM Gemma (~7.5GB), Audio VAE (~100MB), Vocoder (~106MB)")
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

    /// Whether the **video** block feed-forward layers carry biases.
    ///
    /// LTX-2.3 checkpoints ship `ff.net.0.proj.bias` and `ff.net.2.bias` for every
    /// block; LTX-2.5 sets `ff_bias: false` and ships neither. Building bias-carrying
    /// Linears for a 2.5 checkpoint would leave 96 randomly-initialised bias vectors
    /// in the forward pass, so this must track the checkpoint.
    /// (Connector blocks keep their FFN biases in both generations.)
    public var ffBias: Bool

    /// Whether the **audio** block feed-forward layers carry biases.
    ///
    /// Separate from ``ffBias`` because LTX-2.5 diverges between the two streams:
    /// it sets `ff_bias: false` but leaves `audio_ff_bias` unset, and the audio
    /// blocks do ship `audio_ff.net.{0.proj,2}.bias`. Tying the two together drops
    /// 96 trained audio biases — which the video path never notices.
    public var audioFfBias: Bool

    /// LTX-2.5 `use_keyframes_abs_pos_embedding`: the checkpoint carries a learned
    /// `keyframes_abs_pos_embedding` marker added to *generated* keyframe slots
    /// (the DFR pipeline's interior keyframes). Ordinary image / first-and-last-frame
    /// conditioning is never marked, so this stays unused by the standard paths.
    public var keyframesAbsPosEmbedding: Bool

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
        captionProjBeforeConnector: Bool = false,
        ffBias: Bool = true,
        audioFfBias: Bool = true,
        keyframesAbsPosEmbedding: Bool = false
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
        self.ffBias = ffBias
        self.audioFfBias = audioFfBias
        self.keyframesAbsPosEmbedding = keyframesAbsPosEmbedding
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

    /// LTX-2.5 configuration.
    ///
    /// Measured against `ltx-2.5-22b-distilled-transformer-bf16.safetensors`: the
    /// transformer config embedded in the checkpoint metadata is byte-identical to
    /// LTX-2.3's apart from two added keys, `ff_bias: false` and
    /// `use_keyframes_abs_pos_embedding: true`. Layer count, head geometry, RoPE
    /// parameters, connector shape and every tensor shape are unchanged; the only
    /// tensor-level differences are the 96 dropped **video** FFN biases and the new
    /// `keyframes_abs_pos_embedding` marker. The audio blocks keep their FFN biases:
    /// the checkpoint sets `ff_bias: false` and leaves `audio_ff_bias` unset, and
    /// `audio_ff.net.{0.proj,2}.bias` is present for all 48 blocks.
    public static let ltx25 = LTXTransformerConfig(
        captionChannels: 4096,
        gatedAttention: true,
        crossAttentionAdaLN: true,
        captionProjBeforeConnector: true,
        ffBias: false,
        audioFfBias: true,
        keyframesAbsPosEmbedding: true
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
/// - **Frame count**: Must be `8n + 1` (9, 17, 25, ..., 481)
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
/// Which stream a retake regenerates.
///
/// The dual-stream transformer denoises video and audio separately, and freezing
/// one is done by holding it at σ = 0 for the whole schedule — the frozen stream
/// stays live as cross-modal context rather than being dropped, which is what
/// keeps the regenerated stream consistent with it.
public enum RetakeModality: String, Sendable, CaseIterable, Codable {
    /// Regenerate the picture, keep the source audio (passthrough). The default,
    /// and the historical behaviour of `regenerateAudio == false`.
    case videoOnly

    /// Regenerate both streams. Historical `regenerateAudio == true`.
    case both

    /// Regenerate the sound, keep the picture untouched — "same shots, new
    /// audio". The frames are re-muxed from the source rather than decoded, so
    /// the picture is bit-identical and no VAE decode is paid.
    ///
    /// Requires the audio models (``LTXPipeline/loadAudioModels(includeEncoder:progressCallback:)``
    /// with `includeEncoder: true`) and a source video that carries an audio track.
    case audioOnly

    /// Whether the video latents are denoised.
    public var regeneratesVideo: Bool { self != .audioOnly }

    /// Whether the audio latents are denoised.
    public var regeneratesAudio: Bool { self != .videoOnly }
}

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

    /// **Deprecated, no-op.** Used by the legacy hard-injection keyframe path which
    /// re-injected the keyframe slot pre-step with σ-scaled noise. The current append-based
    /// keyframe path keeps guide tokens at σ=0 throughout, so this value is ignored.
    /// Field kept to preserve source-compat for SPM consumers.
    @available(*, deprecated, message: "No-op since the keyframe-append fix (issue #21). Will be removed in a future major release.")
    public var imageCondNoiseScale: Float {
        get { _imageCondNoiseScale }
        set { _imageCondNoiseScale = newValue }
    }
    // Backing storage so internal init assignments don't trip the deprecation warning.
    private var _imageCondNoiseScale: Float

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

    /// Which stream a retake regenerates. Default: ``RetakeModality/videoOnly``.
    ///
    /// `.both` and `.audioOnly` need the audio models loaded with their encoder.
    public var retakeModality: RetakeModality

    /// How far to renoise the source audio in an audio-only retake.
    ///
    /// `1.0` (default) starts from pure noise: a new soundtrack, unrelated to
    /// the source track. Lower values keep a parenté with it — the schedule
    /// starts at the highest trained sigma `<= this value`, and the audio latent
    /// enters it as `σ·noise + (1 − σ)·source`, so rhythm and ambience survive
    /// in proportion. Fewer steps run, so it is also faster.
    ///
    /// Only meaningful for ``RetakeModality/audioOnly``: the two streams share
    /// one sigma schedule, so truncating it for `.both` would truncate the
    /// picture's schedule too. Setting it below `1.0` in any other modality is a
    /// configuration error rather than a silent no-op.
    ///
    /// The distilled schedule is 9 trained values (`1.0 … 0.421875, 0`), so on
    /// the distilled model this snaps: `0.9` starts at `0.909375` — the same
    /// level stage 2 renoises to — `0.8` at `0.725`, `0.5` at `0.421875`, and
    /// anything below `0.421875` leaves no step to run and throws.
    public var audioRetakeStrength: Float

    /// Whether to regenerate audio during retake.
    ///
    /// Kept as a two-value view of ``retakeModality``: reading it is
    /// `retakeModality.regeneratesAudio`, writing it selects `.both` or
    /// `.videoOnly`. It cannot express `.audioOnly` — set ``retakeModality``
    /// directly for that.
    @available(*, deprecated, renamed: "retakeModality",
               message: "Use retakeModality (.videoOnly / .both / .audioOnly).")
    public var regenerateAudio: Bool {
        get { retakeModality.regeneratesAudio }
        set { retakeModality = newValue ? .both : .videoOnly }
    }

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
        keyframes: [KeyframeInput] = [],
        retakeModality: RetakeModality? = nil,
        audioRetakeStrength: Float = 1.0
    ) {
        self.width = width
        self.height = height
        self.numFrames = numFrames
        self.numSteps = numSteps
        self.seed = seed
        self.enhancePrompt = enhancePrompt
        self.imagePath = imagePath
        self._imageCondNoiseScale = imageCondNoiseScale
        self.videoPath = videoPath
        self.retakeStrength = retakeStrength
        self.retakeStartTime = retakeStartTime
        self.retakeEndTime = retakeEndTime
        // `regenerateAudio` is the legacy two-value spelling; an explicit
        // modality wins over it.
        self.retakeModality = retakeModality ?? (regenerateAudio ? .both : .videoOnly)
        self.audioRetakeStrength = audioRetakeStrength
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
        keyframes: [KeyframeInput] = [],
        retakeModality: RetakeModality? = nil,
        audioRetakeStrength: Float = 1.0
    ) {
        self.width = width
        self.height = height
        self.numFrames = numFrames
        self.numSteps = numSteps ?? model.defaultSteps
        self.seed = seed
        self.enhancePrompt = enhancePrompt
        self.imagePath = imagePath
        self._imageCondNoiseScale = imageCondNoiseScale
        self.videoPath = videoPath
        self.retakeStrength = retakeStrength
        self.retakeStartTime = retakeStartTime
        self.retakeEndTime = retakeEndTime
        // `regenerateAudio` is the legacy two-value spelling; an explicit
        // modality wins over it.
        self.retakeModality = retakeModality ?? (regenerateAudio ? .both : .videoOnly)
        self.audioRetakeStrength = audioRetakeStrength
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
            throw LTXError.invalidConfiguration("Number of frames must be 8n + 1 (e.g., 9, 17, 25, ..., 481), got \(numFrames)")
        }

        // Reasonable bounds
        guard width >= 64 && width <= 2048 else {
            throw LTXError.invalidConfiguration("Width must be between 64 and 2048, got \(width)")
        }

        guard height >= 64 && height <= 2048 else {
            throw LTXError.invalidConfiguration("Height must be between 64 and 2048, got \(height)")
        }

        // The upper bound tracks the transformer's RoPE positional design, not an
        // arbitrary limit: temporal coordinates are seconds (pixel frame / fps),
        // normalized by LTXTransformerConfig.maxPos[0] = 20 s — i.e. 481 frames at
        // 24 fps. Beyond that, fractional positions exceed 1.0, outside the range
        // the embedding was designed (and trained) for. Note that typical training
        // clips are shorter (~10 s), so quality may soften on very long videos even
        // within this bound.
        guard numFrames >= 9 && numFrames <= 481 else {
            throw LTXError.invalidConfiguration(
                "Number of frames must be between 9 and 481 (20 s at 24 fps — the RoPE " +
                "positional range of the model), got \(numFrames)")
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
            guard audioRetakeStrength > 0.0 && audioRetakeStrength <= 1.0 else {
                throw LTXError.invalidConfiguration(
                    "Audio retake strength must be in (0.0, 1.0], got \(audioRetakeStrength)")
            }
            // Truncating the schedule truncates it for both streams, so a
            // partial audio renoise only means something when the picture is
            // frozen. Silently ignoring it would look like it worked.
            if audioRetakeStrength < 1.0 && retakeModality != .audioOnly {
                throw LTXError.invalidConfiguration(
                    "audioRetakeStrength < 1 renoises the source audio partially, which needs "
                    + "retakeModality == .audioOnly (video and audio share one sigma schedule). "
                    + "Got \(retakeModality).")
            }
            // A partial window masks *video* latent frames. With the picture
            // frozen there is nothing for it to select, and silently ignoring it
            // would look like a working audio-window feature.
            if retakeModality == .audioOnly && (retakeStartTime != nil || retakeEndTime != nil) {
                throw LTXError.invalidConfiguration(
                    "A partial retake window (retakeStartTime / retakeEndTime) selects video "
                    + "frames to regenerate, and .audioOnly regenerates none. Drop the window, "
                    + "or use .both to retake picture and sound over it.")
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
