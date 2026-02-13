// LatentUtils.swift - Latent Space Utilities for LTX-2
// Copyright 2025

import Foundation
@preconcurrency import MLX
import MLXRandom

// MARK: - Patchify / Unpatchify

/// Patchify video latent from (B, C, F, H, W) to (B, T, C)
///
/// This flattens the spatial-temporal dimensions into a sequence
/// suitable for transformer processing.
///
/// - Parameters:
///   - latent: Input tensor of shape (B, C, F, H, W)
/// - Returns: Patchified tensor of shape (B, T, C) where T = F*H*W
public func patchify(_ latent: MLXArray) -> MLXArray {
    let b = latent.dim(0)
    let c = latent.dim(1)
    let f = latent.dim(2)
    let h = latent.dim(3)
    let w = latent.dim(4)

    let t = f * h * w

    // (B, C, F, H, W) -> (B, F, H, W, C) -> (B, T, C)
    var out = latent.transposed(0, 2, 3, 4, 1)  // (B, F, H, W, C)
    out = out.reshaped([b, t, c])

    return out
}

/// Unpatchify from (B, T, C) back to (B, C, F, H, W)
///
/// - Parameters:
///   - x: Input tensor of shape (B, T, C)
///   - shape: Target latent shape
/// - Returns: Unpatchified tensor of shape (B, C, F, H, W)
public func unpatchify(_ x: MLXArray, shape: VideoLatentShape) -> MLXArray {
    let b = shape.batch
    let c = shape.channels
    let f = shape.frames
    let h = shape.height
    let w = shape.width

    // (B, T, C) -> (B, F, H, W, C) -> (B, C, F, H, W)
    var out = x.reshaped([b, f, h, w, c])
    out = out.transposed(0, 4, 1, 2, 3)

    return out
}

// MARK: - Noise Generation

/// Generate initial noise for video generation
///
/// - Parameters:
///   - shape: Target latent shape
///   - seed: Optional random seed for reproducibility
///   - dtype: Data type for the noise tensor
/// - Returns: Random noise tensor
public func generateNoise(
    shape: VideoLatentShape,
    seed: UInt64? = nil,
    dtype: DType = .float32
) -> MLXArray {
    if let seed = seed {
        MLXRandom.seed(seed)
    }

    let noise = MLXRandom.normal(shape.shape).asType(dtype)
    return noise
}

/// Generate noise with specific sigma level
public func generateScaledNoise(
    shape: VideoLatentShape,
    sigma: Float,
    seed: UInt64? = nil,
    dtype: DType = .float32
) -> MLXArray {
    let noise = generateNoise(shape: shape, seed: seed, dtype: dtype)
    return noise * sigma
}

// MARK: - CFG Preparation

/// Prepare latents for Classifier-Free Guidance by doubling the batch
///
/// Creates [unconditional_latent, conditional_latent] for CFG computation.
///
/// - Parameter latent: Input latent tensor
/// - Returns: Doubled latent tensor for CFG
public func prepareForCFG(_ latent: MLXArray) -> MLXArray {
    return MLX.concatenated([latent, latent], axis: 0)
}

/// Split CFG output back into unconditional and conditional parts
///
/// - Parameter output: Combined output from CFG forward pass
/// - Returns: Tuple of (unconditional, conditional) outputs
public func splitCFGOutput(_ output: MLXArray) -> (uncond: MLXArray, cond: MLXArray) {
    let batchSize = output.dim(0) / 2
    let uncond = output[0..<batchSize]
    let cond = output[batchSize...]
    return (uncond, cond)
}

/// Apply Classifier-Free Guidance
///
/// output = uncond + guidance_scale * (cond - uncond)
///
/// - Parameters:
///   - uncond: Unconditional output
///   - cond: Conditional output
///   - guidanceScale: CFG scale factor
/// - Returns: Guided output
public func applyCFG(
    uncond: MLXArray,
    cond: MLXArray,
    guidanceScale: Float
) -> MLXArray {
    return uncond + guidanceScale * (cond - uncond)
}

/// Combined CFG application from concatenated output
public func applyCFG(
    output: MLXArray,
    guidanceScale: Float
) -> MLXArray {
    let (uncond, cond) = splitCFGOutput(output)
    return applyCFG(uncond: uncond, cond: cond, guidanceScale: guidanceScale)
}

// MARK: - Latent Normalization

/// Normalize latent to have zero mean and unit variance per channel
public func normalizeLatent(_ latent: MLXArray, eps: Float = 1e-6) -> MLXArray {
    // Compute mean and std per channel
    let mean = MLX.mean(latent, axes: [2, 3, 4], keepDims: true)
    let variance = MLX.variance(latent, axes: [2, 3, 4], keepDims: true)
    let std = MLX.sqrt(variance + eps)

    return (latent - mean) / std
}

/// Denormalize latent using per-channel statistics
public func denormalizeLatent(
    _ latent: MLXArray,
    mean: MLXArray,
    std: MLXArray
) -> MLXArray {
    // Reshape statistics for broadcasting: (C,) -> (1, C, 1, 1, 1)
    let meanExp = mean.reshaped([1, -1, 1, 1, 1])
    let stdExp = std.reshaped([1, -1, 1, 1, 1])

    return latent * stdExp + meanExp
}

// MARK: - Utility Functions

/// Get the number of tokens for a given video shape
public func tokenCount(frames: Int, height: Int, width: Int) -> Int {
    // In latent space
    let scaleFactors = SpatioTemporalScaleFactors.default
    let latent = scaleFactors.pixelToLatent(frames: frames, height: height, width: width)
    return latent.frames * latent.height * latent.width
}

/// Validate and adjust dimensions to meet LTX-2 constraints
public func adjustDimensions(
    frames: Int,
    height: Int,
    width: Int
) -> (frames: Int, height: Int, width: Int) {
    // Adjust frames to nearest valid value (8n+1)
    var adjustedFrames = frames
    let remainder = (frames - 1) % 8
    if remainder != 0 {
        if remainder < 4 {
            adjustedFrames = frames - remainder
        } else {
            adjustedFrames = frames + (8 - remainder)
        }
        if adjustedFrames < 1 {
            adjustedFrames = 9  // Minimum valid
        }
    }

    // Adjust height and width to nearest multiples of 32
    let adjustedHeight = ((height + 15) / 32) * 32
    let adjustedWidth = ((width + 15) / 32) * 32

    return (adjustedFrames, max(adjustedHeight, 32), max(adjustedWidth, 32))
}

// MARK: - Memory Estimation

/// Estimate memory usage for video generation in bytes
public func estimateMemoryUsage(
    shape: VideoLatentShape,
    numSteps: Int,
    cfg: Bool = true,
    dtype: DType = .float32
) -> Int64 {
    let bytesPerElement: Int64 = dtype == .float16 ? 2 : 4

    // Latent memory
    let latentElements = Int64(shape.batch * shape.channels * shape.frames * shape.height * shape.width)
    var latentMemory = latentElements * bytesPerElement

    // Double for CFG
    if cfg {
        latentMemory *= 2
    }

    // Token memory (patchified)
    let tokenMemory = Int64(shape.batch * shape.tokenCount * shape.channels) * bytesPerElement

    // Rough estimate for model activations (approximately 2x latent size per step)
    let activationMemory = latentMemory * 2

    // Total estimate
    return latentMemory + tokenMemory + activationMemory
}

/// Format bytes as human-readable string
public func formatBytes(_ bytes: Int64) -> String {
    let gb = Double(bytes) / (1024 * 1024 * 1024)
    if gb >= 1.0 {
        return String(format: "%.1f GB", gb)
    }
    let mb = Double(bytes) / (1024 * 1024)
    return String(format: "%.1f MB", mb)
}
