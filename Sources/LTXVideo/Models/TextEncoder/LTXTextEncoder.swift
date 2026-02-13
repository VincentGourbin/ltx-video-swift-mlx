// LTXTextEncoder.swift - Video Gemma Text Encoder for LTX-2
// Copyright 2025

import Foundation
@preconcurrency import MLX
import MLXFast
import MLXNN

// MARK: - Configuration
// Note: Gemma3Config is defined in Gemma3/Gemma3Config.swift

/// Configuration for the text encoder (projection layers)
public struct TextEncoderConfig: Sendable {
    public let hiddenDim: Int
    public let numGemmaLayers: Int
    public let connectorHeads: Int
    public let connectorHeadDim: Int
    public let connectorLayers: Int
    public let numRegisters: Int

    public static let `default` = TextEncoderConfig(
        hiddenDim: 3840,
        numGemmaLayers: 49,  // 48 layers + 1 embedding
        connectorHeads: 30,
        connectorHeadDim: 128,
        connectorLayers: 2,
        numRegisters: 128
    )

    public init(
        hiddenDim: Int = 3840,
        numGemmaLayers: Int = 49,
        connectorHeads: Int = 30,
        connectorHeadDim: Int = 128,
        connectorLayers: Int = 2,
        numRegisters: Int = 128
    ) {
        self.hiddenDim = hiddenDim
        self.numGemmaLayers = numGemmaLayers
        self.connectorHeads = connectorHeads
        self.connectorHeadDim = connectorHeadDim
        self.connectorLayers = connectorLayers
        self.numRegisters = numRegisters
    }
}

// MARK: - Encoder Output

/// Output from the video Gemma encoder
public struct VideoGemmaEncoderOutput {
    /// Encoded video features (B, T, 3840)
    public let videoEncoding: MLXArray
    /// Attention mask (B, T)
    public let attentionMask: MLXArray
}

// MARK: - Feature Extractor

/// Normalize and concatenate multi-layer hidden states
private func normAndConcatPaddedBatch(
    encodedText: MLXArray,
    sequenceLengths: MLXArray,
    paddingSide: String = "left"
) -> MLXArray {
    let shape = encodedText.shape
    let b = shape[0]
    let t = shape[1]
    let d = shape[2]
    let numLayers = shape[3]
    let eps: Float = 1e-6

    // Build mask: [B, T]
    let tokenIndices = MLXArray(0..<t).reshaped([1, t])

    let mask: MLXArray
    if paddingSide == "right" {
        // For right padding, valid tokens are from 0 to sequence_length-1
        mask = tokenIndices .< sequenceLengths.reshaped([b, 1])
    } else {
        // For left padding, valid tokens are from (T - sequence_length) to T-1
        let startIndices = t - sequenceLengths.reshaped([b, 1])
        mask = tokenIndices .>= startIndices
    }

    // Expand mask for broadcasting: [B, T, 1, 1]
    let maskExpanded = mask.reshaped([b, t, 1, 1])

    // Zero out padded positions for mean calculation
    let masked = MLX.where(maskExpanded, encodedText, MLXArray.zeros(like: encodedText))

    // Compute masked mean per batch per layer
    let denom = (sequenceLengths * d).reshaped([b, 1, 1, 1]).asType(.float32) + eps
    let sum = MLX.sum(masked, axes: [1, 2], keepDims: true)
    let mean = sum / denom

    // Compute masked min/max per batch per layer
    let largeVal: Float = 1e9
    let xForMin = MLX.where(maskExpanded, encodedText, MLXArray(largeVal))
    let xForMax = MLX.where(maskExpanded, encodedText, MLXArray(-largeVal))

    let xMin = MLX.min(xForMin, axes: [1, 2], keepDims: true)
    let xMax = MLX.max(xForMax, axes: [1, 2], keepDims: true)
    let range = xMax - xMin

    // Normalize: scale to [-4, 4] range
    var normed = 8.0 * (encodedText - mean) / (range + eps)

    // Concatenate layers: [B, T, D, L] -> [B, T, D*L]
    normed = normed.reshaped([b, t, d * numLayers])

    // Zero out padded positions in final output
    let maskFlat = mask.reshaped([b, t, 1])
    normed = MLX.where(maskFlat, normed, MLXArray.zeros(like: normed))

    return normed
}

/// Feature extractor for Gemma hidden states
/// Projects concatenated hidden states from all layers to fixed dimension
public class GemmaFeaturesExtractor: Module {
    let hiddenDim: Int
    let numLayers: Int

    @ModuleInfo(key: "aggregate_embed") var aggregateEmbed: Linear

    public init(hiddenDim: Int = 3840, numLayers: Int = 49) {
        self.hiddenDim = hiddenDim
        self.numLayers = numLayers

        // Linear projection: hidden_dim * num_layers -> hidden_dim
        self._aggregateEmbed.wrappedValue = Linear(
            hiddenDim * numLayers,
            hiddenDim,
            bias: false
        )
    }

    /// Project concatenated hidden states
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return aggregateEmbed(x)
    }

    /// Extract features from Gemma hidden states
    public func extractFromHiddenStates(
        hiddenStates: [MLXArray],
        attentionMask: MLXArray,
        paddingSide: String = "left"
    ) -> MLXArray {
        // Stack all hidden states: list of [B, T, 3840] -> [B, T, 3840, 49]
        let stacked = MLX.stacked(hiddenStates, axis: -1)

        // Get sequence lengths from attention mask
        let sequenceLengths = MLX.sum(attentionMask, axis: -1).asType(.int32)

        // Apply per-layer normalization and concatenation
        let normedConcat = normAndConcatPaddedBatch(
            encodedText: stacked,
            sequenceLengths: sequenceLengths,
            paddingSide: paddingSide
        )

        // Project through linear layer
        return aggregateEmbed(normedConcat)
    }
}

// MARK: - Attention for Connector

/// Attention module for 1D connector (simplified from main transformer attention)
///
/// Key difference from main transformer attention:
/// - RMSNorm is applied on full inner_dim (3840) BEFORE head reshape
/// - RoPE is applied on flat [B, T, 3840] BEFORE head reshape
/// - This matches Python LTX-2-MLX connector.Attention behavior
public class ConnectorAttention: Module {
    let heads: Int
    let dimHead: Int
    let innerDim: Int
    let scale: Float

    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    public init(
        dim: Int,
        heads: Int = 30,
        dimHead: Int = 128,
        normEps: Float = 1e-6
    ) {
        self.heads = heads
        self.dimHead = dimHead
        self.innerDim = heads * dimHead
        self.scale = pow(Float(dimHead), -0.5)

        self._toQ.wrappedValue = Linear(dim, innerDim, bias: true)
        self._toK.wrappedValue = Linear(dim, innerDim, bias: true)
        self._toV.wrappedValue = Linear(dim, innerDim, bias: true)
        self._toOut.wrappedValue = Linear(innerDim, dim, bias: true)
        // Full inner_dim norm (3840), NOT per-head (128) — matches Python connector
        self._qNorm.wrappedValue = RMSNorm(dims: innerDim, eps: normEps)
        self._kNorm.wrappedValue = RMSNorm(dims: innerDim, eps: normEps)
    }

    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray? = nil,
        pe: (cos: MLXArray, sin: MLXArray)? = nil
    ) -> MLXArray {
        let b = x.dim(0)
        let t = x.dim(1)

        // Project to Q, K, V — kept flat [B, T, innerDim]
        var q = toQ(x)
        var k = toK(x)
        let vFlat = toV(x)

        // Apply RMSNorm on flat [B, T, 3840] BEFORE head reshape
        q = qNorm(q)
        k = kNorm(k)

        // Apply RoPE on flat [B, T, 3840] BEFORE head reshape
        if let pe = pe {
            q = applyRotaryEmb(q, freqsCis: pe, ropeType: .interleaved)
            k = applyRotaryEmb(k, freqsCis: pe, ropeType: .interleaved)
        }

        // NOW reshape to heads: (B, T, 3840) -> (B, T, H, D) -> (B, H, T, D)
        let qT = q.reshaped([b, t, heads, dimHead]).transposed(0, 2, 1, 3)
        let kT = k.reshaped([b, t, heads, dimHead]).transposed(0, 2, 1, 3)
        let vT = vFlat.reshaped([b, t, heads, dimHead]).transposed(0, 2, 1, 3)

        // Scaled dot-product attention
        var output = MLXFast.scaledDotProductAttention(
            queries: qT, keys: kT, values: vT,
            scale: scale, mask: mask
        )

        // Transpose back and project
        output = output.transposed(0, 2, 1, 3).reshaped([b, t, innerDim])
        return toOut(output)
    }
}

// MARK: - Feed-Forward for Connector

/// GELU feed-forward for connector
public class ConnectorFeedForward: Module {
    @ModuleInfo(key: "project_in") var projectIn: GELUProjection
    @ModuleInfo(key: "project_out") var projectOut: Linear

    public init(dim: Int, dimOut: Int? = nil) {
        let outDim = dimOut ?? dim
        let innerDim = dim * 4

        self._projectIn.wrappedValue = GELUProjection(dim: dim, innerDim: innerDim)
        self._projectOut.wrappedValue = Linear(innerDim, outDim, bias: true)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = projectIn(x)
        out = projectOut(out)
        return out
    }
}

/// GELU projection layer
public class GELUProjection: Module {
    @ModuleInfo(key: "proj") var proj: Linear

    public init(dim: Int, innerDim: Int) {
        self._proj.wrappedValue = Linear(dim, innerDim, bias: true)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return geluApproximate(proj(x))
    }
}

// MARK: - Transformer Block 1D

/// Simple 1D transformer block for sequence processing
public class BasicTransformerBlock1D: Module {
    let normEps: Float

    @ModuleInfo(key: "attn1") var attn1: ConnectorAttention
    @ModuleInfo(key: "ff") var ff: ConnectorFeedForward

    public init(
        dim: Int,
        heads: Int = 30,
        dimHead: Int = 128,
        normEps: Float = 1e-6
    ) {
        self.normEps = normEps

        self._attn1.wrappedValue = ConnectorAttention(
            dim: dim,
            heads: heads,
            dimHead: dimHead,
            normEps: normEps
        )

        self._ff.wrappedValue = ConnectorFeedForward(dim: dim, dimOut: dim)
    }

    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray? = nil,
        pe: (cos: MLXArray, sin: MLXArray)? = nil
    ) -> MLXArray {
        var hiddenStates = x

        // Handle potential extra dimensions
        if hiddenStates.ndim == 4 {
            hiddenStates = hiddenStates.squeezed(axis: 1)
        }

        // Self-attention with residual
        let normWeight1 = MLXArray.ones([hiddenStates.dim(-1)])
        let normHidden = MLXFast.rmsNorm(hiddenStates, weight: normWeight1, eps: normEps)
        let attnOutput = attn1(normHidden, mask: mask, pe: pe)
        hiddenStates = hiddenStates + attnOutput

        if hiddenStates.ndim == 4 {
            hiddenStates = hiddenStates.squeezed(axis: 1)
        }

        // Feed-forward with residual
        let normWeight2 = MLXArray.ones([hiddenStates.dim(-1)])
        let normHidden2 = MLXFast.rmsNorm(hiddenStates, weight: normWeight2, eps: normEps)
        let ffOutput = ff(normHidden2)
        hiddenStates = hiddenStates + ffOutput

        if hiddenStates.ndim == 4 {
            hiddenStates = hiddenStates.squeezed(axis: 1)
        }

        return hiddenStates
    }
}

// MARK: - Embeddings 1D Connector

/// 1D embeddings connector for processing text features
/// Applies transformer blocks with RoPE and learnable registers
public class Embeddings1DConnector: Module {
    let numAttentionHeads: Int
    let innerDim: Int
    let positionalEmbeddingTheta: Float
    let positionalEmbeddingMaxPos: [Int]
    let normEps: Float
    let numLearnableRegisters: Int

    @ModuleInfo(key: "transformer_1d_blocks") var transformer1DBlocks: [BasicTransformerBlock1D]
    @ParameterInfo(key: "learnable_registers") var learnableRegisters: MLXArray

    public init(
        attentionHeadDim: Int = 128,
        numAttentionHeads: Int = 30,
        numLayers: Int = 2,
        positionalEmbeddingTheta: Float = 10000.0,
        positionalEmbeddingMaxPos: [Int]? = nil,
        numLearnableRegisters: Int = 128,
        normEps: Float = 1e-6
    ) {
        let computedInnerDim = numAttentionHeads * attentionHeadDim

        self.numAttentionHeads = numAttentionHeads
        self.innerDim = computedInnerDim
        self.positionalEmbeddingTheta = positionalEmbeddingTheta
        self.positionalEmbeddingMaxPos = positionalEmbeddingMaxPos ?? [1]
        self.normEps = normEps
        self.numLearnableRegisters = numLearnableRegisters

        // Transformer blocks
        self._transformer1DBlocks.wrappedValue = (0..<numLayers).map { _ in
            BasicTransformerBlock1D(
                dim: computedInnerDim,
                heads: numAttentionHeads,
                dimHead: attentionHeadDim,
                normEps: normEps
            )
        }

        // Learnable registers (initialized with uniform random [-1, 1])
        self._learnableRegisters.wrappedValue = MLXRandom.uniform(
            low: -1.0,
            high: 1.0,
            [numLearnableRegisters, computedInnerDim]
        )
    }

    /// Replace padded positions with learnable register tokens
    private func replacePaddedWithLearnableRegisters(
        hiddenStates: MLXArray,
        attentionMask: MLXArray
    ) -> (MLXArray, MLXArray) {
        let seqLen = hiddenStates.dim(1)
        let batchSize = hiddenStates.dim(0)

        guard seqLen % numLearnableRegisters == 0 else {
            fatalError("Sequence length \(seqLen) must be divisible by numLearnableRegisters \(numLearnableRegisters)")
        }

        // Tile registers to match sequence length
        let numDuplications = seqLen / numLearnableRegisters
        let tiledRegisters = MLX.tiled(
            learnableRegisters.expandedDimensions(axis: 0),
            repetitions: [batchSize, numDuplications, 1]
        )

        // Create binary mask from attention mask
        // attention_mask is additive: 0 = attend, large negative = don't attend
        let maskSqueezed = attentionMask.squeezed(axes: [1, 2])  // [B, T]
        let isValid = maskSqueezed .>= -9000.0  // [B, T]

        // Move valid tokens to the front (matching PyTorch behavior)
        let idx = MLXArray(0..<seqLen).expandedDimensions(axis: 0)
        let validInt = isValid.asType(.int32)
        let sortKey = (1 - validInt) * seqLen + idx
        let order = MLX.argSort(sortKey, axis: 1)

        // Gather along axis 1
        let orderExpanded = order.expandedDimensions(axis: -1)
        let adjustedHiddenStates = MLX.takeAlong(hiddenStates, orderExpanded, axis: 1)

        // Flip mask so registers fill the padded tail positions
        // Reverse along axis 1
        let reverseIndices = MLXArray(Array(stride(from: seqLen - 1, through: 0, by: -1)))
        let reversed = MLX.take(isValid, reverseIndices, axis: 1)
        let flippedMask = reversed.asType(hiddenStates.dtype).expandedDimensions(axis: -1)

        let newHiddenStates = flippedMask * adjustedHiddenStates + (1 - flippedMask) * tiledRegisters

        // Clear the attention mask (all positions now valid)
        let newMask = MLXArray.zeros(like: attentionMask)

        return (newHiddenStates, newMask)
    }

    public func callAsFunction(
        _ hiddenStates: MLXArray,
        attentionMask: MLXArray? = nil
    ) -> (MLXArray, MLXArray) {
        var x = hiddenStates
        var mask = attentionMask

        // Replace padded positions with learnable registers
        if let m = mask {
            (x, mask) = replacePaddedWithLearnableRegisters(
                hiddenStates: x,
                attentionMask: m
            )
        }

        // Create position indices for RoPE: [1, 1, T] (1D positions)
        let seqLen = x.dim(1)
        let indicesGrid = MLXArray(0..<seqLen).asType(.float32).reshaped([1, 1, seqLen])

        // Use the main precomputeFreqsCis with interleaved RoPE
        // This produces cos/sin of shape [1, T, innerDim] for flat application
        let freqsCis = precomputeFreqsCis(
            indicesGrid: indicesGrid,
            dim: innerDim,
            theta: positionalEmbeddingTheta,
            maxPos: positionalEmbeddingMaxPos,
            numAttentionHeads: numAttentionHeads,
            ropeType: .interleaved
        )

        // Process through transformer blocks
        for block in transformer1DBlocks {
            x = block(x, mask: mask, pe: freqsCis)
        }

        // Final normalization (no learnable weight, identity)
        let normWeight = MLXArray.ones([x.dim(-1)])
        x = MLXFast.rmsNorm(x, weight: normWeight, eps: normEps)

        let outMask = mask ?? MLXArray.zeros([x.dim(0), 1, 1, x.dim(1)])

        return (x, outMask)
    }
}

// MARK: - Video Gemma Text Encoder Model

/// Video Gemma Text Encoder Model for LTX-2
///
/// Processes text prompts through:
/// 1. Gemma language model (external, extracts hidden states)
/// 2. Feature extractor (projects multi-layer hidden states)
/// 3. Embeddings connector (1D transformer refinement)
///
/// Output is 3840-dim embeddings. Caption projection (3840 → 4096)
/// is handled by the transformer model.
public class VideoGemmaTextEncoderModel: Module {
    @ModuleInfo(key: "feature_extractor") public var featureExtractor: GemmaFeaturesExtractor
    @ModuleInfo(key: "embeddings_connector") var embeddingsConnector: Embeddings1DConnector

    public init(
        featureExtractor: GemmaFeaturesExtractor? = nil,
        embeddingsConnector: Embeddings1DConnector? = nil
    ) {
        self._featureExtractor.wrappedValue = featureExtractor ?? GemmaFeaturesExtractor()
        self._embeddingsConnector.wrappedValue = embeddingsConnector ?? Embeddings1DConnector()
    }

    /// Convert binary attention mask to additive mask for softmax
    private func convertToAdditiveMask(
        _ attentionMask: MLXArray,
        dtype: DType = .float32
    ) -> MLXArray {
        let largeValue: Float
        switch dtype {
        case .float16:
            largeValue = 65504.0
        case .bfloat16:
            largeValue = 3.38e38
        default:
            largeValue = 3.40e38
        }

        // (mask - 1) makes 1 -> 0, 0 -> -1
        let additiveMask = (attentionMask.asType(dtype) - 1) * largeValue
        // Reshape for attention: [B, 1, 1, T]
        return additiveMask.reshaped([attentionMask.dim(0), 1, 1, attentionMask.dim(-1)])
    }

    /// Encode text from pre-computed Gemma hidden states
    ///
    /// - Parameters:
    ///   - hiddenStates: List of hidden states from each Gemma layer (49 layers)
    ///   - attentionMask: Binary attention mask [B, T]
    ///   - paddingSide: Side where padding was applied
    /// - Returns: VideoGemmaEncoderOutput with encoded text (3840 dim)
    public func encodeFromHiddenStates(
        hiddenStates: [MLXArray],
        attentionMask: MLXArray,
        paddingSide: String = "left"
    ) -> VideoGemmaEncoderOutput {
        // Extract features from hidden states
        let encoded = featureExtractor.extractFromHiddenStates(
            hiddenStates: hiddenStates,
            attentionMask: attentionMask,
            paddingSide: paddingSide
        )

        // Convert mask to additive format
        let connectorMask = convertToAdditiveMask(attentionMask, dtype: encoded.dtype)

        // Process through connector
        let (processed, outputMask) = embeddingsConnector(encoded, attentionMask: connectorMask)

        // Convert mask back to binary for output
        let binaryMask = (outputMask.squeezed(axes: [1, 2]) .>= -0.5).asType(.int32)

        // Apply mask to zero out padded positions
        let finalEncoded = processed * binaryMask.expandedDimensions(axis: -1).asType(processed.dtype)

        return VideoGemmaEncoderOutput(
            videoEncoding: finalEncoded,
            attentionMask: binaryMask
        )
    }

    /// Encode from already-projected features
    public func encodeProjected(
        projectedFeatures: MLXArray,
        attentionMask: MLXArray
    ) -> VideoGemmaEncoderOutput {
        let connectorMask = convertToAdditiveMask(attentionMask, dtype: projectedFeatures.dtype)

        let (processed, outputMask) = embeddingsConnector(projectedFeatures, attentionMask: connectorMask)

        let binaryMask = (outputMask.squeezed(axes: [1, 2]) .>= -0.5).asType(.int32)
        let finalEncoded = processed * binaryMask.expandedDimensions(axis: -1).asType(processed.dtype)

        return VideoGemmaEncoderOutput(
            videoEncoding: finalEncoded,
            attentionMask: binaryMask
        )
    }

    public func callAsFunction(
        hiddenStates: [MLXArray],
        attentionMask: MLXArray,
        paddingSide: String = "left"
    ) -> VideoGemmaEncoderOutput {
        return encodeFromHiddenStates(
            hiddenStates: hiddenStates,
            attentionMask: attentionMask,
            paddingSide: paddingSide
        )
    }
}

// MARK: - Factory Functions

/// Create a text encoder with default LTX-2 configuration
public func createTextEncoder(
    config: TextEncoderConfig = .default
) -> VideoGemmaTextEncoderModel {
    let featureExtractor = GemmaFeaturesExtractor(
        hiddenDim: config.hiddenDim,
        numLayers: config.numGemmaLayers
    )

    let embeddingsConnector = Embeddings1DConnector(
        attentionHeadDim: config.connectorHeadDim,
        numAttentionHeads: config.connectorHeads,
        numLayers: config.connectorLayers,
        numLearnableRegisters: config.numRegisters
    )

    return VideoGemmaTextEncoderModel(
        featureExtractor: featureExtractor,
        embeddingsConnector: embeddingsConnector
    )
}

// MARK: - LTX Text Encoder (High-Level Wrapper)

/// High-level text encoder for LTX-2 video generation
///
/// This class wraps the text encoding pipeline and provides a simple interface
/// for encoding prompts. The actual Gemma model should be loaded separately
/// and hidden states passed to the encode methods.
public class LTXTextEncoder: Module {
    public let config: TextEncoderConfig

    @ModuleInfo(key: "encoder") var encoder: VideoGemmaTextEncoderModel

    public init(config: TextEncoderConfig = .default) {
        self.config = config
        self._encoder.wrappedValue = createTextEncoder(config: config)
    }

    /// Encode from pre-extracted Gemma hidden states
    ///
    /// - Parameters:
    ///   - hiddenStates: All 49 hidden states from Gemma (embedding + 48 layers)
    ///   - attentionMask: Binary attention mask [B, T]
    ///   - paddingSide: Padding side ("left" or "right")
    /// - Returns: Encoded features (B, T, 3840) and attention mask
    public func encode(
        hiddenStates: [MLXArray],
        attentionMask: MLXArray,
        paddingSide: String = "left"
    ) -> VideoGemmaEncoderOutput {
        return encoder(
            hiddenStates: hiddenStates,
            attentionMask: attentionMask,
            paddingSide: paddingSide
        )
    }

    /// Get the output dimension of the encoder
    public var outputDimension: Int {
        return config.hiddenDim
    }
}
