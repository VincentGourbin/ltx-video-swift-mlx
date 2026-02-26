// LatentUtils.swift - Latent Space Utilities for LTX-2
// Copyright 2025

import CoreGraphics
import Foundation
import ImageIO
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
func patchify(_ latent: MLXArray) -> MLXArray {
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
func unpatchify(_ x: MLXArray, shape: VideoLatentShape) -> MLXArray {
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
/// Generates float32 noise then casts to target dtype, matching Python's
/// `mx.random.normal(..., dtype=model_dtype)` behavior. The default dtype is
/// bfloat16 to match the Python mlx-video reference implementation.
///
/// - Parameters:
///   - shape: Target latent shape
///   - seed: Optional random seed for reproducibility
///   - dtype: Data type for the noise tensor (default: bfloat16 matching Python)
/// - Returns: Random noise tensor
func generateNoise(
    shape: VideoLatentShape,
    seed: UInt64? = nil,
    dtype: DType = .float32
) -> MLXArray {
    if let seed = seed {
        MLXRandom.seed(seed)
    }

    // Generate noise in float32 (matching Diffusers prepare_latents)
    // Latents stay in float32 throughout the denoising loop for numerical precision
    // Only cast to bfloat16 when entering the transformer
    let noise = MLXRandom.normal(shape.shape, dtype: dtype)
    return noise
}

/// Generate noise with specific sigma level
func generateScaledNoise(
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
func prepareForCFG(_ latent: MLXArray) -> MLXArray {
    return MLX.concatenated([latent, latent], axis: 0)
}

/// Split CFG output back into unconditional and conditional parts
///
/// - Parameter output: Combined output from CFG forward pass
/// - Returns: Tuple of (unconditional, conditional) outputs
func splitCFGOutput(_ output: MLXArray) -> (uncond: MLXArray, cond: MLXArray) {
    let batchSize = output.dim(0) / 2
    let uncond = output[0..<batchSize]
    let cond = output[batchSize...]
    return (uncond, cond)
}

/// Apply Classifier-Free Guidance
///
/// output = uncond + guidance_scale * (cond - uncond)
///
/// Matches Python behavior where the scalar * bfloat16 stays bfloat16
/// (Python float scalars don't promote MLX array dtype).
///
/// - Parameters:
///   - uncond: Unconditional output
///   - cond: Conditional output
///   - guidanceScale: CFG scale factor
/// - Returns: Guided output (same dtype as inputs)
func applyCFG(
    uncond: MLXArray,
    cond: MLXArray,
    guidanceScale: Float
) -> MLXArray {
    // Match Python formula exactly: vel_pos + (cfg_scale - 1.0) * (vel_pos - vel_neg)
    // This is algebraically identical to uncond + scale * (cond - uncond)
    // but uses different intermediate values, producing identical bfloat16 rounding.
    let scaleMinus1 = MLXArray(guidanceScale - 1.0).asType(cond.dtype)
    return cond + scaleMinus1 * (cond - uncond)
}

/// Combined CFG application from concatenated output
func applyCFG(
    output: MLXArray,
    guidanceScale: Float
) -> MLXArray {
    let (uncond, cond) = splitCFGOutput(output)
    return applyCFG(uncond: uncond, cond: cond, guidanceScale: guidanceScale)
}

/// Apply guidance rescale to reduce overexposure from CFG
///
/// Rescales the CFG output so its per-channel standard deviation matches
/// the conditional output's std, then blends with the original CFG output.
///
/// Formula: rescaled = cfgOutput * (condStd / cfgStd), result = phi * rescaled + (1 - phi) * cfgOutput
///
/// - Parameters:
///   - cfgOutput: Output after CFG application (B, C, F, H, W)
///   - condOutput: Conditional-only output (B, C, F, H, W)
///   - phi: Rescale factor (0.0 = no rescale, 0.7 = recommended)
/// - Returns: Rescaled output
func applyGuidanceRescale(
    cfgOutput: MLXArray,
    condOutput: MLXArray,
    phi: Float
) -> MLXArray {
    guard phi > 0.0 else { return cfgOutput }

    let eps: Float = 1e-8

    // Std over all dims except batch (axes: 1=C, 2=F, 3=H, 4=W), matching Diffusers
    // rescale_noise_cfg: std(dim=list(range(1, ndim)), keepdim=True)
    let cfgStd = MLX.sqrt(MLX.variance(cfgOutput, axes: [1, 2, 3, 4], keepDims: true) + eps)
    let condStd = MLX.sqrt(MLX.variance(condOutput, axes: [1, 2, 3, 4], keepDims: true) + eps)

    // Rescale CFG output to match conditional std
    let rescaled = cfgOutput * (condStd / cfgStd)

    // Blend between rescaled and original
    return MLXArray(phi) * rescaled + MLXArray(1.0 - phi) * cfgOutput
}

// MARK: - AdaIN Filtering

/// Apply Adaptive Instance Normalization (AdaIN) to a latent tensor
///
/// Normalizes each channel of the input latent to match the per-channel mean
/// and standard deviation of the reference latent. This prevents distribution
/// shift when upsampling latents between pipeline stages.
///
/// Matches Lightricks/LTX-Video `adain_filter_latent` exactly.
///
/// - Parameters:
///   - latent: Input latent tensor (B, C, F, H, W) — to be normalized
///   - reference: Reference latent tensor (B, C, F', H', W') — statistics target
///     Spatial dimensions can differ; only per-channel stats are used.
///   - factor: Blending factor (1.0 = full AdaIN, 0.0 = no change)
/// - Returns: AdaIN-filtered latent with same shape as input
func adainFilterLatent(
    _ latent: MLXArray,
    reference: MLXArray,
    factor: Float = 1.0
) -> MLXArray {
    guard factor > 0 else { return latent }

    // Compute per-channel mean and std for both tensors
    // Axes [2, 3, 4] = F, H, W (spatial-temporal dims)
    let latentMean = MLX.mean(latent, axes: [2, 3, 4], keepDims: true)
    let latentVar = MLX.variance(latent, axes: [2, 3, 4], keepDims: true)
    let latentStd = MLX.sqrt(latentVar)

    let refMean = MLX.mean(reference, axes: [2, 3, 4], keepDims: true)
    let refVar = MLX.variance(reference, axes: [2, 3, 4], keepDims: true)
    let refStd = MLX.sqrt(refVar)

    // AdaIN: normalize input to zero-mean/unit-var, then apply reference stats
    let normalized = (latent - latentMean) / (latentStd + 1e-8)
    let result = normalized * refStd + refMean

    // Blend between original and AdaIN result
    if factor >= 1.0 {
        return result
    }
    return MLXArray(factor) * result + MLXArray(1.0 - factor) * latent
}

// MARK: - Latent Normalization

/// Normalize latent to have zero mean and unit variance per channel
func normalizeLatent(_ latent: MLXArray, eps: Float = 1e-6) -> MLXArray {
    // Compute mean and std per channel
    let mean = MLX.mean(latent, axes: [2, 3, 4], keepDims: true)
    let variance = MLX.variance(latent, axes: [2, 3, 4], keepDims: true)
    let std = MLX.sqrt(variance + eps)

    return (latent - mean) / std
}

/// Denormalize latent using per-channel statistics
func denormalizeLatent(
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
func tokenCount(frames: Int, height: Int, width: Int) -> Int {
    // In latent space
    let scaleFactors = SpatioTemporalScaleFactors.default
    let latent = scaleFactors.pixelToLatent(frames: frames, height: height, width: width)
    return latent.frames * latent.height * latent.width
}

/// Validate and adjust dimensions to meet LTX-2 constraints
func adjustDimensions(
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
func estimateMemoryUsage(
    shape: VideoLatentShape,
    numSteps: Int,
    cfg: Bool = true,
    dtype: DType = .float32
) -> Int64 {
    let bytesPerElement: Int64 = (dtype == .float16 || dtype == .bfloat16) ? 2 : 4

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
func formatBytes(_ bytes: Int64) -> String {
    let gb = Double(bytes) / (1024 * 1024 * 1024)
    if gb >= 1.0 {
        return String(format: "%.1f GB", gb)
    }
    let mb = Double(bytes) / (1024 * 1024)
    return String(format: "%.1f MB", mb)
}

// MARK: - Image Loading

/// Load an image from disk, resize to target dimensions, and normalize to [-1, 1]
///
/// Returns shape (1, 3, 1, H, W) — batch, channels, temporal, height, width
///
/// - Parameters:
///   - path: Path to the image file (PNG, JPEG, etc.)
///   - width: Target width in pixels
///   - height: Target height in pixels
/// - Returns: Normalized image tensor ready for VAE encoding
/// - Throws: LTXError.fileNotFound if image cannot be loaded
func loadImage(from path: String, width: Int, height: Int) throws -> MLXArray {
    let url = URL(fileURLWithPath: path)

    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
          let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil)
    else {
        throw LTXError.fileNotFound("Cannot load image from: \(path)")
    }

    LTXDebug.log("Loaded image: \(cgImage.width)x\(cgImage.height) -> resizing to \(width)x\(height)")

    // Resize to target dimensions using CoreGraphics
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    guard let context = CGContext(
        data: nil,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: width * 4,
        space: colorSpace,
        bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
    ) else {
        throw LTXError.videoProcessingFailed("Failed to create graphics context for image resize")
    }

    context.interpolationQuality = .high
    context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

    guard let data = context.data else {
        throw LTXError.videoProcessingFailed("Failed to get pixel data from resized image")
    }

    // Convert RGBA pixels to float RGB normalized to [-1, 1]
    let ptr = data.bindMemory(to: UInt8.self, capacity: height * width * 4)
    var pixels = [Float](repeating: 0, count: height * width * 3)
    for i in 0..<(height * width) {
        pixels[i * 3 + 0] = Float(ptr[i * 4 + 0]) / 127.5 - 1.0  // R
        pixels[i * 3 + 1] = Float(ptr[i * 4 + 1]) / 127.5 - 1.0  // G
        pixels[i * 3 + 2] = Float(ptr[i * 4 + 2]) / 127.5 - 1.0  // B
    }

    // Build tensor: (H, W, 3) -> (3, H, W) -> (1, 3, 1, H, W)
    let hwc = MLXArray(pixels, [height, width, 3])
    let chw = hwc.transposed(2, 0, 1)  // (3, H, W)
    let result = chw.reshaped([1, 3, 1, height, width])

    LTXDebug.log("Image tensor: \(result.shape), mean=\(result.mean().item(Float.self)), range=[\(result.min().item(Float.self)), \(result.max().item(Float.self))]")

    return result
}
