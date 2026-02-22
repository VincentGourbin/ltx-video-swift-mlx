// Gemma3Model.swift - Gemma 3 text model for LTX-Video hidden state extraction
// Architecture faithful to MLXLLM/Gemma3Text.swift, using MLXLMCommon utilities.
// Key differences from MLXLLM: returns all hidden states instead of logits.

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN
import MLXRandom

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
        // CRITICAL: Python mlx-vlm Gemma3 does NOT apply rope_scaling to attention RoPE.
        // Python: nn.RoPE(head_dim, traditional=False, base=base)  — no scaling factor
        // Passing scalingConfig here would set scale=1/factor=0.125, causing hidden state divergence
        // starting at the first global layer (index 5) and accumulating through all 48 layers.
        if isSliding {
            self.rope = initializeRope(
                dims: headDim, base: config.ropeLocalBaseFreq, traditional: false,
                scalingConfig: nil, maxPositionEmbeddings: nil)
        } else {
            self.rope = initializeRope(
                dims: headDim, base: config.ropeTheta, traditional: false,
                scalingConfig: nil, maxPositionEmbeddings: nil)
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
class Gemma3InnerModel: Module {
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
        // Embed and scale by sqrt(hidden_size) — use embedding dtype
        var h = embedTokens(inputs)
        let scale = MLXArray(sqrt(Float(config.hiddenSize))).asType(h.dtype)
        h = h * scale

        // Collect hidden states
        var allHiddenStates: [MLXArray]? = outputHiddenStates ? [] : nil

        // Add embedding as first hidden state
        if outputHiddenStates {
            allHiddenStates?.append(h)
        }

        // Create attention masks combining causal + padding
        // Use ADDITIVE masks matching Python mlx-video text_encoder exactly:
        //   Python creates: 0 for attend, min_val for don't attend
        //   This avoids potential code path differences in SDPA between boolean and additive masks
        let globalMask: MLXFast.ScaledDotProductAttentionMaskMode
        let slidingWindowMask: MLXFast.ScaledDotProductAttentionMaskMode

        if let attentionMask = attentionMask {
            let seqLen = h.dim(1)
            let dtype = h.dtype

            // Build causal mask: lower triangle = true
            let causal = MLXArray.tri(seqLen, m: seqLen, dtype: .bool)

            // Padding mask: 1 = valid, 0 = padding → bool
            let padMask = attentionMask.reshaped(attentionMask.dim(0), 1, 1, seqLen).asType(.bool)

            // Combined: attend where causal AND valid
            let combinedBool = causal[.newAxis, .newAxis] .&& padMask

            // Convert to additive mask: 0 for attend, min_val for don't attend
            // Match Python: min_val = mx.finfo(dtype).min for bf16/fp16, -1e9 otherwise
            let minVal: Float
            switch dtype {
            case .bfloat16:
                minVal = -3.3895314e38  // mx.finfo(mx.bfloat16).min
            case .float16:
                minVal = -65504.0
            default:
                minVal = -1e9
            }
            let additiveMask = MLX.where(
                combinedBool,
                MLXArray.zeros(like: combinedBool).asType(dtype),
                MLXArray(minVal).asType(dtype)
            )
            globalMask = .array(additiveMask)

            // Python text encoder: sliding_mask = full_causal_mask (no window constraint)
            slidingWindowMask = globalMask

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
        let numLayers = layers.count
        for (i, layer) in layers.enumerated() {
            let isGlobal = (i % config.slidingWindowPattern == config.slidingWindowPattern - 1)
            let mask = isGlobal ? globalMask : slidingWindowMask
            h = layer(h, mask: mask)

            // Eval after EVERY layer to match Python mlx-video behavior
            // Python does: h = layer(h, local_mask, cache[i]); mx.eval(h)
            // This is critical: lazy eval of multiple layers produces different
            // bfloat16 rounding than per-layer eval, causing divergence over 48 layers
            MLX.eval(h)

            // Add layer output as hidden state for layers 0..46 (raw, without final norm)
            // Layer 47 (last) is NOT added here — its normed version is added after the loop
            // This matches Python: `if output_hidden_states and i < num_layers - 1`
            if outputHiddenStates && i < numLayers - 1 {
                allHiddenStates?.append(h)
            }
        }

        // Apply final norm
        h = norm(h)

        // Add norm(last_layer_output) as the 49th hidden state (matches Python)
        if outputHiddenStates {
            allHiddenStates?.append(h)
        }

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
class Gemma3TextModel: Module {
    @ModuleInfo var model: Gemma3InnerModel
    let config: Gemma3Config

    init(_ config: Gemma3Config) {
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
    func callAsFunction(
        _ inputs: MLXArray,
        attentionMask: MLXArray? = nil,
        outputHiddenStates: Bool = true
    ) -> (lastHiddenState: MLXArray, allHiddenStates: [MLXArray]?) {
        return model(inputs, attentionMask: attentionMask, outputHiddenStates: outputHiddenStates)
    }

    /// Sanitize weight keys from safetensors format.
    /// Handles `language_model.` prefix from VLM conversions.
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var processedWeights = weights

        // Handle VLM models with language_model prefix
        let unflattened = ModuleParameters.unflattened(weights)
        if let lm = unflattened["language_model"] {
            processedWeights = Dictionary(uniqueKeysWithValues: lm.flattened())
        }

        // Remove lm_head keys — we don't generate tokens, only extract hidden states
        processedWeights = processedWeights.filter { !$0.key.hasPrefix("lm_head") }

        // Convert float32 weights to bfloat16 (matching Python mlx-video behavior)
        // The Python LTX2TextEncoder.sanitize() does: value.astype(mx.bfloat16) for float32 weights
        // CRITICAL: Must match Python's bf16 rounding — running in float32 produces MORE divergence
        // because the model expects bf16-precision hidden states at each layer.
        for (key, value) in processedWeights {
            if value.dtype == .float32 {
                processedWeights[key] = value.asType(.bfloat16)
            }
        }

        return processedWeights
    }
}

// MARK: - Text Generation (for Prompt Enhancement)

extension Gemma3TextModel {
    /// Generate text autoregressively using KV cache for efficiency.
    ///
    /// Uses tied embeddings (embed_tokens.weight) as lm_head for logit projection.
    /// Gemma3 models use tied weights by default.
    ///
    /// - Parameters:
    ///   - inputIds: Prompt tokens [1, T]
    ///   - maxNewTokens: Maximum number of tokens to generate
    ///   - temperature: Sampling temperature (0.0 = greedy, >0 = stochastic)
    ///   - topP: Top-p (nucleus) sampling threshold (0.0-1.0, default 0.95)
    ///   - repetitionPenalty: Penalize repeated tokens (1.0 = no penalty, >1.0 = penalize)
    ///   - repetitionContextSize: How many recent tokens to apply repetition penalty to
    ///   - eosTokenId: Token ID that signals end of generation
    /// - Returns: Array of generated token IDs (not including input)
    func generateTokens(
        inputIds: MLXArray,
        maxNewTokens: Int = 512,
        temperature: Float = 0.7,
        topP: Float = 0.95,
        repetitionPenalty: Float = 1.1,
        repetitionContextSize: Int = 64,
        eosTokenId: Int32 = 1,
        eosTokenIds: Set<Int32>? = nil
    ) -> [Int32] {
        let stopTokenIds = eosTokenIds ?? [eosTokenId]
        let innerModel = model
        let numLayers = innerModel.layers.count

        // Create KV caches for each layer
        let caches: [KVCache] = (0..<numLayers).map { _ in KVCacheSimple() }

        // Prefill: process the full prompt
        var h = innerModel.embedTokens(inputIds)
        let scale = MLXArray(sqrt(Float(config.hiddenSize))).asType(h.dtype)
        h = h * scale

        // Build causal mask for prefill
        let seqLen = inputIds.dim(1)
        let causalMask: MLXFast.ScaledDotProductAttentionMaskMode = createAttentionMask(h: h, cache: nil as KVCache?)

        // Build sliding window mask
        let slidingMask: MLXFast.ScaledDotProductAttentionMaskMode
        if config.slidingWindowPattern > 1 {
            slidingMask = createAttentionMask(h: h, cache: nil as KVCache?, windowSize: config.slidingWindow)
        } else {
            slidingMask = .none
        }

        for (i, layer) in innerModel.layers.enumerated() {
            let isGlobal = (i % config.slidingWindowPattern == config.slidingWindowPattern - 1)
            let mask = isGlobal ? causalMask : slidingMask
            h = layer(h, mask: mask, cache: caches[i])
        }
        h = innerModel.norm(h)
        MLX.eval(h)

        // Get logits for the last token — asLinear handles quantized embeddings
        var logits = innerModel.embedTokens.asLinear(h[0..., (seqLen - 1)..., 0...])

        var generatedTokens: [Int32] = []

        for _ in 0..<maxNewTokens {
            // Apply repetition penalty to recent tokens
            var processedLogits = logits
            if repetitionPenalty != 1.0 && !generatedTokens.isEmpty {
                let contextSize = min(repetitionContextSize, generatedTokens.count)
                let recentTokens = Array(generatedTokens.suffix(contextSize))
                var logitsArray = logits.reshaped(-1).asArray(Float.self)
                let tokenSet = Set(recentTokens)
                for tokenId in tokenSet {
                    let idx = Int(tokenId)
                    if idx >= 0 && idx < logitsArray.count {
                        if logitsArray[idx] > 0 {
                            logitsArray[idx] /= repetitionPenalty
                        } else {
                            logitsArray[idx] *= repetitionPenalty
                        }
                    }
                }
                processedLogits = MLXArray(logitsArray).reshaped(logits.shape)
            }

            // Sample next token
            let nextToken: MLXArray
            if temperature <= 0 {
                nextToken = MLX.argMax(processedLogits, axis: -1)
            } else {
                nextToken = sampleTopP(processedLogits, temperature: temperature, topP: topP)
            }
            MLX.eval(nextToken)

            let tokenId = nextToken.item(Int32.self)

            // Check for EOS
            if stopTokenIds.contains(tokenId) {
                break
            }

            generatedTokens.append(tokenId)

            // Forward pass for single token with cache
            let tokenInput = nextToken.reshaped(1, 1)
            h = innerModel.embedTokens(tokenInput)
            h = h * scale.asType(h.dtype)

            for (i, layer) in innerModel.layers.enumerated() {
                // With cache populated, mask is .none (causal handled by cache)
                h = layer(h, mask: .none, cache: caches[i])
            }
            h = innerModel.norm(h)

            logits = innerModel.embedTokens.asLinear(h)
            MLX.eval(logits)
        }

        return generatedTokens
    }

    /// Top-p (nucleus) sampling using MLX
    private func sampleTopP(_ logits: MLXArray, temperature: Float, topP: Float) -> MLXArray {
        let flat = logits.reshaped(-1)
        let probs = MLX.softmax(flat / MLXArray(temperature), axis: -1)

        // Sort descending
        let sortedIndices = MLX.argSort(-probs, axis: -1)
        let sortedProbs = probs[sortedIndices]

        // Cumulative sum
        let cumulativeProbs = MLX.cumsum(sortedProbs, axis: -1)

        // Mask tokens beyond top-p threshold
        let mask = cumulativeProbs .> MLXArray(1.0 - topP)
        let filteredProbs = MLX.where(mask, sortedProbs, MLXArray.zeros(like: sortedProbs))

        // Sample from filtered distribution
        let sortedToken = MLXRandom.categorical(MLX.log(filteredProbs + 1e-10))
        return sortedIndices[sortedToken]
    }
}

// MARK: - Backward Compatibility

/// Type alias for backward compatibility with existing code that uses `Gemma3TextEncoder`
typealias Gemma3TextEncoder = Gemma3TextModel
