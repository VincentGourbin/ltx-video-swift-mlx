// Gemma3Config.swift - Codable configuration for Gemma 3 text model
// Matches MLXLLM's Gemma3TextConfiguration for loading HuggingFace config.json

import Foundation
import MLXLMCommon

/// Configuration for the Gemma 3 text model.
/// Codable to parse config.json from HuggingFace models (e.g. mlx-community/gemma-3-12b-it-4bit).
/// Architecture matches MLXLLM's Gemma3TextConfiguration.
public struct Gemma3Config: Codable, Sendable {
    public let modelType: String
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let headDim: Int
    public let rmsNormEps: Float
    public let vocabularySize: Int
    public let kvHeads: Int
    public let ropeTheta: Float
    public let ropeLocalBaseFreq: Float
    public let ropeTraditional: Bool
    public let queryPreAttnScalar: Float
    public let slidingWindow: Int
    public let slidingWindowPattern: Int
    public let maxPositionEmbeddings: Int
    public let ropeScaling: [String: StringOrNumber]?

    /// Quantization config (present in quantized models)
    public let quantization: QuantizationConfig?

    public struct QuantizationConfig: Codable, Sendable {
        public let groupSize: Int
        public let bits: Int

        enum CodingKeys: String, CodingKey {
            case groupSize = "group_size"
            case bits
        }
    }

    public init(
        modelType: String = "gemma3_text",
        hiddenSize: Int = 3840,
        hiddenLayers: Int = 48,
        intermediateSize: Int = 15360,
        attentionHeads: Int = 16,
        headDim: Int = 256,
        rmsNormEps: Float = 1e-6,
        vocabularySize: Int = 262208,
        kvHeads: Int = 8,
        ropeTheta: Float = 1_000_000.0,
        ropeLocalBaseFreq: Float = 10_000.0,
        ropeTraditional: Bool = false,
        queryPreAttnScalar: Float = 256,
        slidingWindow: Int = 1024,
        slidingWindowPattern: Int = 6,
        maxPositionEmbeddings: Int = 131072,
        ropeScaling: [String: StringOrNumber]? = nil,
        quantization: QuantizationConfig? = nil
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.headDim = headDim
        self.rmsNormEps = rmsNormEps
        self.vocabularySize = vocabularySize
        self.kvHeads = kvHeads
        self.ropeTheta = ropeTheta
        self.ropeLocalBaseFreq = ropeLocalBaseFreq
        self.ropeTraditional = ropeTraditional
        self.queryPreAttnScalar = queryPreAttnScalar
        self.slidingWindow = slidingWindow
        self.slidingWindowPattern = slidingWindowPattern
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.ropeScaling = ropeScaling
        self.quantization = quantization
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case ropeTheta = "rope_theta"
        case ropeLocalBaseFreq = "rope_local_base_freq"
        case ropeTraditional = "rope_traditional"
        case queryPreAttnScalar = "query_pre_attn_scalar"
        case slidingWindow = "sliding_window"
        case slidingWindowPattern = "sliding_window_pattern"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeScaling = "rope_scaling"
        case quantization
    }

    // Support nested text_config for VLM models converted via mlx_lm.convert
    enum VLMCodingKeys: String, CodingKey {
        case textConfig = "text_config"
    }

    public init(from decoder: Decoder) throws {
        let nestedContainer = try decoder.container(keyedBy: VLMCodingKeys.self)

        let container =
            if nestedContainer.contains(.textConfig) {
                try nestedContainer.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig)
            } else {
                try decoder.container(keyedBy: CodingKeys.self)
            }

        modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "gemma3_text"
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 3840
        hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 48
        intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 15360
        attentionHeads = try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 256
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        vocabularySize = try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 262208
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000.0
        ropeLocalBaseFreq = try container.decodeIfPresent(Float.self, forKey: .ropeLocalBaseFreq) ?? 10_000.0
        ropeTraditional = try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
        queryPreAttnScalar = try container.decodeIfPresent(Float.self, forKey: .queryPreAttnScalar) ?? 256
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 1024
        slidingWindowPattern = try container.decodeIfPresent(Int.self, forKey: .slidingWindowPattern) ?? 6
        maxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131072
        ropeScaling = try container.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)

        // Quantization may be at top level even when text_config is nested
        let topContainer = try decoder.container(keyedBy: CodingKeys.self)
        quantization = try topContainer.decodeIfPresent(QuantizationConfig.self, forKey: .quantization)
    }

    /// Load config from a directory containing config.json
    public static func load(from directory: URL) throws -> Self {
        let configURL = directory.appendingPathComponent("config.json")
        let data = try Data(contentsOf: configURL)
        return try JSONDecoder().decode(Self.self, from: data)
    }
}
