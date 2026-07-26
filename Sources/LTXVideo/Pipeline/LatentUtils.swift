// LatentUtils.swift - Latent Space Utilities for LTX-2
// Copyright 2025

import AVFoundation
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
    dtype: DType = .float32
) -> Int64 {
    let bytesPerElement: Int64 = (dtype == .float16 || dtype == .bfloat16) ? 2 : 4

    // Latent memory
    let latentElements = Int64(shape.batch * shape.channels * shape.frames * shape.height * shape.width)
    let latentMemory = latentElements * bytesPerElement

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

    LTXDebug.log("Loaded image: \(cgImage.width)x\(cgImage.height) -> center-crop+resize to \(width)x\(height)")

    // Center-crop to target aspect ratio, then resize (preserves proportions)
    let srcW = cgImage.width
    let srcH = cgImage.height
    let targetAspect = Double(width) / Double(height)
    let srcAspect = Double(srcW) / Double(srcH)

    let cropRect: CGRect
    if srcAspect > targetAspect {
        // Source is wider — crop horizontally
        let cropW = Int(Double(srcH) * targetAspect)
        let offsetX = (srcW - cropW) / 2
        cropRect = CGRect(x: offsetX, y: 0, width: cropW, height: srcH)
    } else {
        // Source is taller — crop vertically
        let cropH = Int(Double(srcW) / targetAspect)
        let offsetY = (srcH - cropH) / 2
        cropRect = CGRect(x: 0, y: offsetY, width: srcW, height: cropH)
    }

    guard let croppedImage = cgImage.cropping(to: cropRect) else {
        throw LTXError.videoProcessingFailed("Failed to center-crop image")
    }

    // Resize cropped image to exact target dimensions
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
    context.draw(croppedImage, in: CGRect(x: 0, y: 0, width: width, height: height))

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

// MARK: - Video Loading

/// Load a video from disk, extract frames at target resolution, and normalize to [-1, 1]
///
/// Returns shape (1, 3, numFrames, H, W) — batch, channels, temporal, height, width
///
/// Uses AVFoundation to extract uniformly-spaced frames from the video file,
/// resize each frame to the target dimensions, and normalize pixel values.
///
/// - Parameters:
///   - path: Path to the video file (MP4, MOV, etc.)
///   - width: Target width in pixels
///   - height: Target height in pixels
///   - numFrames: Number of frames to extract (must be 8n+1)
///   - tail: When true, read the video's **last** `numFrames` frames instead of
///     sampling uniformly across its duration. Frame times are computed from the
///     track's frame rate and requested mid-frame with a half-frame tolerance, so
///     a clip whose PTS do not start exactly at zero still resolves (the uniform
///     path keeps its zero tolerance and exact-boundary requests).
/// - Returns: Normalized video tensor ready for VAE encoding
/// - Throws: LTXError.fileNotFound if video cannot be loaded
func loadVideo(
    from path: String, width: Int, height: Int, numFrames: Int, tail: Bool = false
) async throws -> MLXArray {
    let url = URL(fileURLWithPath: path)
    guard FileManager.default.fileExists(atPath: path) else {
        throw LTXError.fileNotFound("Video file not found: \(path)")
    }

    let asset = AVURLAsset(url: url)
    let duration = try await asset.load(.duration)
    let durationSeconds = CMTimeGetSeconds(duration)

    guard durationSeconds > 0 else {
        throw LTXError.videoProcessingFailed("Video has zero duration: \(path)")
    }

    LTXDebug.log("Loading video: \(path), duration=\(String(format: "%.2f", durationSeconds))s, extracting \(numFrames) frames at \(width)x\(height)")

    let generator = AVAssetImageGenerator(asset: asset)
    generator.appliesPreferredTrackTransform = true
    generator.maximumSize = CGSize(width: width, height: height)

    var requestTimes: [CMTime] = []
    if tail {
        // Last numFrames frames. Frame times come from the track's own frame rate;
        // each request lands mid-frame with a half-frame tolerance so a clip whose
        // PTS are offset (a stream copy, a container edit list) still resolves to
        // the intended frame instead of failing outright.
        let track = try await asset.loadTracks(withMediaType: .video).first
        let nominalFPS = try await track?.load(.nominalFrameRate) ?? 0
        let fps = Double(nominalFPS > 0 ? nominalFPS : 24)
        let totalFrames = max(1, Int((durationSeconds * fps).rounded()))
        guard totalFrames >= numFrames else {
            throw LTXError.videoProcessingFailed(
                "Video has \(totalFrames) frame(s) at \(String(format: "%.2f", fps)) fps but "
                + "\(numFrames) are needed for the tail read: \(path)")
        }
        let halfFrame = CMTime(seconds: 0.5 / fps, preferredTimescale: 600)
        generator.requestedTimeToleranceBefore = halfFrame
        generator.requestedTimeToleranceAfter = halfFrame
        for i in 0..<numFrames {
            let frameIndex = totalFrames - numFrames + i
            let seconds = (Double(frameIndex) + 0.5) / fps
            requestTimes.append(CMTime(seconds: seconds, preferredTimescale: 600))
        }
        LTXDebug.log("Tail read: frames \(totalFrames - numFrames)..<\(totalFrames) of \(totalFrames) at \(String(format: "%.2f", fps)) fps")
    } else {
        // Uniformly sample numFrames times from the video duration
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        for i in 0..<numFrames {
            let fraction = Double(i) / Double(max(numFrames - 1, 1))
            let seconds = fraction * durationSeconds
            requestTimes.append(CMTime(seconds: seconds, preferredTimescale: 600))
        }
    }

    // Extract and process frames
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    var allPixels = [Float]()
    allPixels.reserveCapacity(numFrames * height * width * 3)

    for (frameIdx, time) in requestTimes.enumerated() {
        let cgImage: CGImage
        do {
            let (image, _) = try await generator.image(at: time)
            cgImage = image
        } catch {
            throw LTXError.videoProcessingFailed("Failed to extract frame \(frameIdx) at \(CMTimeGetSeconds(time))s: \(error)")
        }

        // Resize to exact target dimensions using CoreGraphics
        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else {
            throw LTXError.videoProcessingFailed("Failed to create graphics context for frame \(frameIdx)")
        }

        context.interpolationQuality = .high
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        guard let data = context.data else {
            throw LTXError.videoProcessingFailed("Failed to get pixel data from frame \(frameIdx)")
        }

        // Convert RGBA to float RGB normalized to [-1, 1]
        let ptr = data.bindMemory(to: UInt8.self, capacity: height * width * 4)
        for i in 0..<(height * width) {
            allPixels.append(Float(ptr[i * 4 + 0]) / 127.5 - 1.0)  // R
            allPixels.append(Float(ptr[i * 4 + 1]) / 127.5 - 1.0)  // G
            allPixels.append(Float(ptr[i * 4 + 2]) / 127.5 - 1.0)  // B
        }
    }

    // Build tensor: (F, H, W, 3) -> (3, F, H, W) -> (1, 3, F, H, W)
    let fhwc = MLXArray(allPixels, [numFrames, height, width, 3])
    let cfhw = fhwc.transposed(3, 0, 1, 2)  // (3, F, H, W)
    let result = cfhw.reshaped([1, 3, numFrames, height, width])

    LTXDebug.log("Video tensor: \(result.shape), mean=\(result.mean().item(Float.self)), range=[\(result.min().item(Float.self)), \(result.max().item(Float.self))]")

    return result
}

