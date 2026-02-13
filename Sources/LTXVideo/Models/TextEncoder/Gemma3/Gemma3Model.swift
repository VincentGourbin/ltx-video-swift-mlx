// Gemma3Model.swift - Gemma 3 text model for LTX-Video hidden state extraction
// Architecture faithful to MLXLLM/Gemma3Text.swift, using MLXLMCommon utilities.
// Key differences from MLXLLM: returns all hidden states instead of logits.

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

// MARK: - Attention

class Gemma3Attention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let layerIdx: Int
    let scale: Float
    let isSliding: Bool
    let slidingWindow: Int

    @ModuleInfo(key: "q_proj") var queryProj: Linear
    @ModuleInfo(key: "k_proj") var keyProj: Linear
    @ModuleInfo(key: "v_proj") var valueProj: Linear
    @ModuleInfo(key: "o_proj") var outputProj: Linear

    @ModuleInfo(key: "q_norm") var queryNorm: Gemma.RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: Gemma.RMSNorm

    @ModuleInfo var rope: OffsetLayer

    init(_ config: Gemma3Config, layerIdx: Int) {
        let dim = config.hiddenSize
        self.nHeads = config.attentionHeads
        self.nKVHeads = config.kvHeads
        self.headDim = config.headDim
        self.layerIdx = layerIdx
        self.slidingWindow = config.slidingWindow

        // Gemma3 uses queryPreAttnScalar (=256 for 12B), NOT 1/sqrt(headDim)
        self.scale = pow(config.queryPreAttnScalar, -0.5)

        self._queryProj.wrappedValue = Linear(dim, nHeads * headDim, bias: false)
        self._keyProj.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
        self._valueProj.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
        self._outputProj.wrappedValue = Linear(nHeads * headDim, dim, bias: false)

        self._queryNorm.wrappedValue = Gemma.RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self._keyNorm.wrappedValue = Gemma.RMSNorm(dimensions: headDim, eps: config.rmsNormEps)

        // Sliding window pattern: every Nth layer is global, rest are local (sliding)
        self.isSliding = (layerIdx + 1) % config.slidingWindowPattern != 0

        // Local layers use ropeLocalBaseFreq (10000), global layers use ropeTheta (1000000)
        if isSliding {
            self.rope = initializeRope(
                dims: headDim, base: config.ropeLocalBaseFreq, traditional: false,
                scalingConfig: nil, maxPositionEmbeddings: nil)
        } else {
            self.rope = initializeRope(
                dims: headDim, base: config.ropeTheta, traditional: false,
                scalingConfig: config.ropeScaling,
                maxPositionEmbeddings: config.maxPositionEmbeddings)
        }

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache? = nil
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var queries = queryProj(x)
        var keys = keyProj(x)
        var values = valueProj(x)

        queries = queries.reshaped(B, L, nHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)

        queries = queryNorm(queries)
        keys = keyNorm(keys)

        if let cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
        } else {
            queries = rope(queries, offset: 0)
            keys = rope(keys, offset: 0)
        }

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)
        return outputProj(output)
    }
}

// MARK: - MLP

class Gemma3MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        self._gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        self._upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

// MARK: - Decoder Layer

class Gemma3DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: Gemma3Attention
    @ModuleInfo var mlp: Gemma3MLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Gemma.RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: Gemma.RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: Gemma.RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: Gemma.RMSNorm

    init(_ config: Gemma3Config, layerIdx: Int) {
        self._selfAttention.wrappedValue = Gemma3Attention(config, layerIdx: layerIdx)
        self.mlp = Gemma3MLP(dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)

        self._inputLayerNorm.wrappedValue = Gemma.RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = Gemma.RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._preFeedforwardLayerNorm.wrappedValue = Gemma.RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayerNorm.wrappedValue = Gemma.RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache? = nil
    ) -> MLXArray {
        let inputNorm = inputLayerNorm(x)
        let r = selfAttention(inputNorm, mask: mask, cache: cache)
        let attnNorm = postAttentionLayerNorm(r)
        let h = Gemma.clipResidual(x, attnNorm)
        let preMLPNorm = preFeedforwardLayerNorm(h)
        let r2 = mlp(preMLPNorm)
        let postMLPNorm = postFeedforwardLayerNorm(r2)
        return Gemma.clipResidual(h, postMLPNorm)
    }
}

// MARK: - Gemma3 Inner Model

/// Inner model (embed_tokens + layers + norm) matching MLXLLM's Gemma3Model.
/// Weight keys: model.embed_tokens, model.layers, model.norm
public class Gemma3InnerModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [Gemma3DecoderLayer]
    @ModuleInfo var norm: Gemma.RMSNorm

    let config: Gemma3Config

    init(_ config: Gemma3Config) {
        self.config = config

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )

        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map { layerIdx in
            Gemma3DecoderLayer(config, layerIdx: layerIdx)
        }

        self.norm = Gemma.RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        super.init()
    }

    /// Forward pass returning (lastHiddenState, allHiddenStates).
    ///
    /// allHiddenStates contains: embedding (1) + output of each layer (hiddenLayers).
    /// For 48-layer model: 1 embedding + 48 layer outputs = 49 total.
    /// The last element is the post-norm output (same as lastHiddenState).
    func callAsFunction(
        _ inputs: MLXArray,
        attentionMask: MLXArray? = nil,
        outputHiddenStates: Bool = true
    ) -> (lastHiddenState: MLXArray, allHiddenStates: [MLXArray]?) {
        // Embed and scale by sqrt(hidden_size)
        var h = embedTokens(inputs)
        let scale = MLXArray(sqrt(Float(config.hiddenSize)), dtype: .bfloat16)
        h = h * scale.asType(h.dtype)

        // Collect hidden states
        var allHiddenStates: [MLXArray]? = outputHiddenStates ? [] : nil

        // Add embedding as first hidden state
        if outputHiddenStates {
            allHiddenStates?.append(h)
        }

        // Create attention masks combining causal + padding
        let globalMask: MLXFast.ScaledDotProductAttentionMaskMode
        let slidingWindowMask: MLXFast.ScaledDotProductAttentionMaskMode

        if let attentionMask = attentionMask {
            // Combine causal mask with padding mask
            // Causal: lower triangle + diagonal = true (attend), upper triangle = false (mask)
            let seqLen = h.dim(1)
            let causal = MLXArray.tri(seqLen, m: seqLen, dtype: .bool)
            // Padding: 0 = padding (mask out), 1 = real token
            // Expand to [B, 1, 1, T] for broadcasting with [1, 1, T, T] causal
            let padMask = attentionMask.reshaped(attentionMask.dim(0), 1, 1, seqLen)
                .asType(.bool)
            // Combined: attend only where causal=true AND padding=true
            let combinedBool = causal[.newAxis, .newAxis] .&& padMask
            globalMask = .array(combinedBool)

            // Sliding window: causal + padding + window constraint
            if config.slidingWindowPattern > 1 {
                let windowSize = config.slidingWindow
                // Sliding window: only attend to last windowSize tokens
                let rowIndices = MLXArray(Array(Int32(0)..<Int32(seqLen))).reshaped(1, 1, seqLen, 1)
                let colIndices = MLXArray(Array(Int32(0)..<Int32(seqLen))).reshaped(1, 1, 1, seqLen)
                let windowMask = (rowIndices - colIndices) .< MLXArray(Int32(windowSize))
                let slidingBool = combinedBool .&& windowMask
                slidingWindowMask = .array(slidingBool)
            } else {
                slidingWindowMask = .none
            }
        } else {
            // No padding mask — causal only (original behavior)
            globalMask = createAttentionMask(h: h, cache: nil as KVCache?)
            slidingWindowMask =
                if config.slidingWindowPattern > 1 {
                    createAttentionMask(h: h, cache: nil as KVCache?, windowSize: config.slidingWindow)
                } else {
                    .none
                }
        }

        // Process through layers
        for (i, layer) in layers.enumerated() {
            let isGlobal = (i % config.slidingWindowPattern == config.slidingWindowPattern - 1)
            let mask = isGlobal ? globalMask : slidingWindowMask
            h = layer(h, mask: mask)

            // Add layer output as hidden state (raw, without final norm — matches Python)
            if outputHiddenStates {
                allHiddenStates?.append(h)
            }

            // Periodic eval for memory management (every 8 layers)
            if (i + 1) % 8 == 0 {
                MLX.eval(h)
            }
        }

        // Apply final norm for the lastHiddenState return value
        h = norm(h)

        return (h, allHiddenStates)
    }
}

// MARK: - Gemma3 Text Encoder (Public API)

/// Gemma 3 text encoder for LTX-Video.
///
/// Wraps the inner model with a `model.` prefix in the weight key path,
/// matching the MLXLLM weight structure (model.embed_tokens, model.layers, model.norm).
///
/// Usage:
/// ```swift
/// let config = try Gemma3Config.load(from: modelDirectory)
/// let encoder = Gemma3TextModel(config)
/// try Gemma3WeightLoader.loadWeights(into: encoder, from: modelDirectory)
/// let (lastHidden, allHidden) = encoder(tokenIds, outputHiddenStates: true)
/// ```
public class Gemma3TextModel: Module {
    @ModuleInfo public var model: Gemma3InnerModel
    public let config: Gemma3Config

    public init(_ config: Gemma3Config) {
        self.config = config
        self.model = Gemma3InnerModel(config)
        super.init()
    }

    /// Forward pass for text encoding with hidden state extraction.
    ///
    /// - Parameters:
    ///   - inputs: Token IDs [B, T]
    ///   - attentionMask: Optional padding mask [B, T] where 1=real token, 0=padding
    ///   - outputHiddenStates: Whether to collect all intermediate hidden states (default: true)
    /// - Returns: (lastHiddenState [B, T, D], allHiddenStates [49 x [B, T, D]])
    public func callAsFunction(
        _ inputs: MLXArray,
        attentionMask: MLXArray? = nil,
        outputHiddenStates: Bool = true
    ) -> (lastHiddenState: MLXArray, allHiddenStates: [MLXArray]?) {
        return model(inputs, attentionMask: attentionMask, outputHiddenStates: outputHiddenStates)
    }

    /// Sanitize weight keys from safetensors format.
    /// Handles `language_model.` prefix from VLM conversions.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var processedWeights = weights

        // Handle VLM models with language_model prefix
        let unflattened = ModuleParameters.unflattened(weights)
        if let lm = unflattened["language_model"] {
            processedWeights = Dictionary(uniqueKeysWithValues: lm.flattened())
        }

        // Remove lm_head keys — we don't generate tokens, only extract hidden states
        processedWeights = processedWeights.filter { !$0.key.hasPrefix("lm_head") }

        return processedWeights
    }
}

// MARK: - Backward Compatibility

/// Type alias for backward compatibility with existing code that uses `Gemma3TextEncoder`
public typealias Gemma3TextEncoder = Gemma3TextModel
