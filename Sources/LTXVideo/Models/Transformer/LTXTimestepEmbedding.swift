// LTXTimestepEmbedding.swift - Timestep Embedding for LTX-2
// Copyright 2025

import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - Sinusoidal Timestep Embedding

/// Generate sinusoidal timestep embeddings
///
/// - Parameters:
///   - timesteps: Timestep values of shape (B,) or (B, T)
///   - embeddingDim: Dimension of the embedding
///   - maxPeriod: Maximum period for sinusoidal encoding
/// - Returns: Timestep embeddings of shape (B, embeddingDim) or (B, T, embeddingDim)
func getTimestepEmbedding(
    timesteps: MLXArray,
    embeddingDim: Int,
    maxPeriod: Float = 10000.0
) -> MLXArray {
    let half = embeddingDim / 2
    // Match Python: arange(half_dim) / half_dim (NOT linspace)
    let freqIndices = MLXArray(Array(0..<half)).asType(.float32) / Float(half)
    let freqs = MLX.exp(-log(Float32(maxPeriod)) * freqIndices)

    // Handle different input shapes
    let originalShape = timesteps.shape
    let flatTimesteps = timesteps.reshaped([-1, 1])

    // Compute arguments for sin/cos
    let args = flatTimesteps * freqs.reshaped([1, -1])

    // Concatenate sin and cos
    let embedding = MLX.concatenated([MLX.cos(args), MLX.sin(args)], axis: -1)

    // Handle odd embedding dimensions
    var result = embedding
    if embeddingDim % 2 == 1 {
        result = MLX.concatenated([
            embedding,
            MLX.zeros([embedding.dim(0), 1], dtype: embedding.dtype)
        ], axis: -1)
    }

    // Restore original shape
    if originalShape.count > 1 {
        var newShape = originalShape
        newShape.append(embeddingDim)
        result = result.reshaped(newShape)
    }

    return result.asType(.float32)
}

// MARK: - AdaLN Single

/// Adaptive Layer Normalization (Single)
///
/// Produces scale, shift, and gate parameters from timestep embeddings.
/// Used for conditioning transformer blocks on timestep.
class AdaLayerNormSingle: Module {
    @ModuleInfo var emb: TimestepMLP
    @ModuleInfo var linear: Linear

    let numEmbeddings: Int
    let innerDim: Int

    init(
        innerDim: Int,
        numEmbeddings: Int = 6
    ) {
        self.innerDim = innerDim
        self.numEmbeddings = numEmbeddings

        self._emb.wrappedValue = TimestepMLP(innerDim: innerDim)
        self._linear.wrappedValue = Linear(innerDim, numEmbeddings * innerDim, bias: true)
    }

    func callAsFunction(_ timesteps: MLXArray) -> (emb: MLXArray, embeddedTimestep: MLXArray) {
        // Get timestep embeddings: (B,) -> (B, D)
        let embeddedTimestep = emb(timesteps)

        // Project to scale/shift/gate values: (B, D) -> (B, num_embeddings * D)
        let ada = linear(MLXNN.silu(embeddedTimestep))

        return (ada, embeddedTimestep)
    }
}

// MARK: - Timestep MLP

/// MLP for processing timestep embeddings
///
/// Python reference: Timesteps(num_channels=256) → TimestepEmbedding(in_channels=256, time_embed_dim=inner_dim)
class TimestepMLP: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    let innerDim: Int
    let frequencyEmbeddingSize: Int

    init(innerDim: Int, frequencyEmbeddingSize: Int = 256) {
        self.innerDim = innerDim
        self.frequencyEmbeddingSize = frequencyEmbeddingSize

        // linear_1: projects sinusoidal embedding (256) to inner_dim
        self._linear1.wrappedValue = Linear(frequencyEmbeddingSize, innerDim, bias: true)
        // linear_2: inner_dim to inner_dim
        self._linear2.wrappedValue = Linear(innerDim, innerDim, bias: true)
    }

    func callAsFunction(_ timesteps: MLXArray) -> MLXArray {
        // Get sinusoidal embeddings (fixed 256-dim, matching Python num_channels=256)
        var emb = getTimestepEmbedding(timesteps: timesteps, embeddingDim: frequencyEmbeddingSize)

        // MLP: Linear(256→innerDim) -> SiLU -> Linear(innerDim→innerDim)
        emb = linear1(emb)
        emb = MLXNN.silu(emb)
        emb = linear2(emb)

        return emb
    }
}

// MARK: - PixArt-Alpha Text Projection

/// Projects caption embeddings with GELU activation
///
/// Adapted from PixArt-alpha implementation.
class PixArtAlphaTextProjection: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(
        inFeatures: Int,
        hiddenSize: Int,
        outFeatures: Int? = nil
    ) {
        let outputSize = outFeatures ?? hiddenSize

        self._linear1.wrappedValue = Linear(inFeatures, hiddenSize, bias: true)
        self._linear2.wrappedValue = Linear(hiddenSize, outputSize, bias: true)
    }

    func callAsFunction(_ caption: MLXArray) -> MLXArray {
        var hiddenStates = linear1(caption)
        hiddenStates = MLXNN.geluApproximate(hiddenStates)
        hiddenStates = linear2(hiddenStates)
        return hiddenStates
    }
}
