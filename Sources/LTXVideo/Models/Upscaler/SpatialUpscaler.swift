// SpatialUpscaler.swift - Latent Spatial Upscaler for LTX-2 Two-Stage Pipeline
// Uses native MLX Conv3d for proper cross-frame convolution
// Matches Blaizzy/mlx-video LatentUpsampler architecture exactly
// Copyright 2025

import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - GroupNorm3D (Channels-Last)

/// Group normalization for 5D tensors in channels-last (N, D, H, W, C) format.
/// Computes statistics over all spatial+temporal positions within each group.
/// Matches Blaizzy/mlx-video GroupNorm3d exactly.
public class UpscalerGroupNorm3D: Module {
    let numGroups: Int
    let numChannels: Int
    let eps: Float
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray

    public init(numGroups: Int = 32, numChannels: Int, eps: Float = 1e-5) {
        self.numGroups = numGroups
        self.numChannels = numChannels
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([numChannels])
        self._bias.wrappedValue = MLXArray.zeros([numChannels])
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: (N, D, H, W, C) — channels last
        let n = x.dim(0), d = x.dim(1), h = x.dim(2), w = x.dim(3), c = x.dim(4)
        let inputDtype = x.dtype

        // Cast to float32 for numerical stability (matches Blaizzy)
        var y = x.asType(.float32)

        // Reshape to (N, D*H*W, num_groups, C//num_groups)
        let channelsPerGroup = c / numGroups
        y = y.reshaped([n, d * h * w, numGroups, channelsPerGroup])

        // Compute mean and var over spatial (axis 1) and channel-within-group (axis 3)
        let mean = MLX.mean(y, axes: [1, 3], keepDims: true)
        let variance = MLX.variance(y, axes: [1, 3], keepDims: true)

        // Normalize
        y = (y - mean) / MLX.sqrt(variance + eps)

        // Reshape back to (N, D, H, W, C)
        y = y.reshaped([n, d, h, w, c])

        // Apply affine (weight and bias are (C,) shape)
        let w32 = weight.asType(.float32)
        let b32 = bias.asType(.float32)
        y = y * w32 + b32

        // Cast back to original dtype
        return y.asType(inputDtype)
    }
}

// MARK: - Upscaler ResBlock3D

/// Residual block with two Conv3d and GroupNorm3D (channels-last)
/// Architecture: conv1 → norm1 → SiLU → conv2 → norm2 → SiLU(x + residual)
public class UpscalerResBlock3D: Module {
    let channels: Int

    @ModuleInfo(key: "conv1") var conv1: Conv3d
    @ModuleInfo(key: "norm1") var norm1: UpscalerGroupNorm3D
    @ModuleInfo(key: "conv2") var conv2: Conv3d
    @ModuleInfo(key: "norm2") var norm2: UpscalerGroupNorm3D

    public init(channels: Int) {
        self.channels = channels

        // Conv3d: kernel_size=3, padding=1 — native MLX Conv3d (NDHWC format)
        self._conv1.wrappedValue = Conv3d(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: 3,
            padding: 1
        )
        self._norm1.wrappedValue = UpscalerGroupNorm3D(numGroups: 32, numChannels: channels)
        self._conv2.wrappedValue = Conv3d(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: 3,
            padding: 1
        )
        self._norm2.wrappedValue = UpscalerGroupNorm3D(numGroups: 32, numChannels: channels)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: (N, D, H, W, C) — channels last
        let residual = x
        var h = conv1(x)
        h = norm1(h)
        h = MLXNN.silu(h)
        h = conv2(h)
        h = norm2(h)
        // Activation AFTER residual addition (matches Blaizzy)
        h = MLXNN.silu(h + residual)
        return h
    }
}

// MARK: - Pixel Shuffle 2D (Channels-Last)

/// 2D Pixel Shuffle in channels-last format:
/// (N, H, W, C*r*r) → (N, H*r, W*r, C)
public func pixelShuffle2DNHWC(_ x: MLXArray, upscaleFactor: Int = 2) -> MLXArray {
    let n = x.dim(0)
    let h = x.dim(1)
    let w = x.dim(2)
    let c = x.dim(3)
    let r = upscaleFactor
    let outC = c / (r * r)

    // (N, H, W, out_c, r, r) → (N, H, r, W, r, out_c) → (N, H*r, W*r, out_c)
    var out = x.reshaped([n, h, w, outC, r, r])
    out = out.transposed(0, 1, 4, 2, 5, 3)  // (N, H, r, W, r, out_c)
    out = out.reshaped([n, h * r, w * r, outC])
    return out
}

// MARK: - Spatial Rational Resampler

/// Per-frame 2D convolution + PixelShuffle for 2x spatial upsampling
/// Weight key: upsampler.conv.weight / upsampler.conv.bias
public class SpatialRationalResampler: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d
    let midChannels: Int

    public init(midChannels: Int = 1024) {
        self.midChannels = midChannels

        // Conv2d: mid → 4*mid for PixelShuffle(2)
        self._conv.wrappedValue = Conv2d(
            inputChannels: midChannels,
            outputChannels: 4 * midChannels,
            kernelSize: .init((3, 3)),
            padding: .init((1, 1))
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: (N, D, H, W, C) — channels last 3D
        let n = x.dim(0), d = x.dim(1), h = x.dim(2), w = x.dim(3), c = x.dim(4)

        // Reshape to (N*D, H, W, C) for 2D conv (already channels-last)
        var frames = x.reshaped([n * d, h, w, c])

        // Apply 2D conv (MLX Conv2d expects NHWC)
        frames = conv(frames)  // (N*D, H, W, 4*C)

        // PixelShuffle: (N*D, H, W, 4*C) → (N*D, H*2, W*2, C)
        frames = pixelShuffle2DNHWC(frames, upscaleFactor: 2)

        // Reshape back to (N, D, H*2, W*2, C)
        let hNew = frames.dim(1), wNew = frames.dim(2)
        return frames.reshaped([n, d, hNew, wNew, c])
    }
}

// MARK: - Spatial Upscaler Model

/// Latent spatial upscaler for LTX-2 two-stage pipeline
///
/// Architecture (matches Blaizzy/mlx-video LatentUpsampler):
/// All processing in channels-last (NDHWC) format using native Conv3d.
///
/// 1. initial_conv: Conv3d(128→1024, k=3, p=1) + GroupNorm(32, 1024) + SiLU
/// 2. 4× ResBlock3D(1024) — pre-upsample
/// 3. SpatialRationalResampler: per-frame Conv2d(1024→4096) + PixelShuffle(2) → 1024ch, 2×H, 2×W
/// 4. 4× ResBlock3D(1024) — post-upsample
/// 5. final_conv: Conv3d(1024→128, k=3, p=1)
///
/// Input:  (B, C, F, H, W)  — channels first (standard latent format)
/// Output: (B, C, F, H*2, W*2)  — channels first
public class SpatialUpscaler: Module {
    @ModuleInfo(key: "initial_conv") var initialConv: Conv3d
    @ModuleInfo(key: "initial_norm") var initialNorm: UpscalerGroupNorm3D
    @ModuleInfo(key: "res_blocks") var resBlocks: [UpscalerResBlock3D]
    @ModuleInfo(key: "upsampler") var upsampler: SpatialRationalResampler
    @ModuleInfo(key: "post_upsample_res_blocks") var postResBlocks: [UpscalerResBlock3D]
    @ModuleInfo(key: "final_conv") var finalConv: Conv3d

    let inChannels: Int
    let midChannels: Int
    let numBlocksPerStage: Int

    public init(
        inChannels: Int = 128,
        midChannels: Int = 1024,
        numBlocksPerStage: Int = 4
    ) {
        self.inChannels = inChannels
        self.midChannels = midChannels
        self.numBlocksPerStage = numBlocksPerStage

        // All Conv3d use kernel_size=3, padding=1 (native MLX NDHWC format)
        self._initialConv.wrappedValue = Conv3d(
            inputChannels: inChannels,
            outputChannels: midChannels,
            kernelSize: 3,
            padding: 1
        )
        self._initialNorm.wrappedValue = UpscalerGroupNorm3D(numGroups: 32, numChannels: midChannels)

        self._resBlocks.wrappedValue = (0..<numBlocksPerStage).map { _ in
            UpscalerResBlock3D(channels: midChannels)
        }

        self._upsampler.wrappedValue = SpatialRationalResampler(midChannels: midChannels)

        self._postResBlocks.wrappedValue = (0..<numBlocksPerStage).map { _ in
            UpscalerResBlock3D(channels: midChannels)
        }

        self._finalConv.wrappedValue = Conv3d(
            inputChannels: midChannels,
            outputChannels: inChannels,
            kernelSize: 3,
            padding: 1
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Input: (B, C, F, H, W) — channels first
        // Convert to channels last (B, F, H, W, C) for all internal processing
        var h = x.transposed(0, 2, 3, 4, 1)

        // Initial conv + norm + activation
        h = initialConv(h)
        h = initialNorm(h)
        h = MLXNN.silu(h)

        // Pre-upsample residual blocks
        for block in resBlocks {
            h = block(h)
        }

        // Spatial upsample (2x, per-frame Conv2d + PixelShuffle)
        h = upsampler(h)

        // Post-upsample residual blocks
        for block in postResBlocks {
            h = block(h)
        }

        // Final conv
        h = finalConv(h)

        // Convert back to channels first (B, C, F, H, W)
        return h.transposed(0, 4, 1, 2, 3)
    }
}

// MARK: - Upscaler Weight Loading

/// Load SpatialUpscaler from safetensors file with proper weight format conversion
///
/// Handles:
/// - Conv3d weights: PyTorch (O, I, D, H, W) → MLX (O, D, H, W, I)
/// - Conv2d weights: PyTorch (O, I, H, W) → MLX (O, H, W, I)
/// - GroupNorm/bias: no transpose needed
///
/// - Parameter weightsPath: Path to the upscaler safetensors file
/// - Returns: Loaded and configured SpatialUpscaler
public func loadSpatialUpscaler(from weightsPath: String) throws -> SpatialUpscaler {
    LTXDebug.log("Loading spatial upscaler from \(weightsPath)...")

    let rawWeights = try MLX.loadArrays(url: URL(fileURLWithPath: weightsPath))

    // Detect mid_channels from weight shape
    var midChannels = 1024
    if let sampleWeight = rawWeights["res_blocks.0.conv1.weight"] {
        midChannels = sampleWeight.dim(0)
        LTXDebug.log("Detected mid_channels: \(midChannels)")
    }

    // Create model
    let upscaler = SpatialUpscaler(inChannels: 128, midChannels: midChannels, numBlocksPerStage: 4)

    // Sanitize weights: transpose conv weights from PyTorch to MLX format
    var sanitized: [(String, MLXArray)] = []

    for (key, value) in rawWeights {
        var newValue = value

        // Conv3d weights (5D): PyTorch (O, I, D, H, W) → MLX (O, D, H, W, I)
        if key.contains("conv") && key.hasSuffix(".weight") && value.ndim == 5 {
            newValue = value.transposed(0, 2, 3, 4, 1)
        }
        // Conv2d weights (4D): PyTorch (O, I, H, W) → MLX (O, H, W, I)
        else if key.contains("conv") && key.hasSuffix(".weight") && value.ndim == 4 {
            newValue = value.transposed(0, 2, 3, 1)
        }

        // Skip blur_down_kernel (it's a fixed constant, not a learnable parameter)
        if key.contains("blur_down") {
            continue
        }

        sanitized.append((key, newValue))
    }

    // Cast all weights to float32 (safetensors may be bfloat16)
    sanitized = sanitized.map { (key, value) in
        (key, value.asType(.float32))
    }

    // Load weights into model
    let params = ModuleParameters.unflattened(sanitized)
    upscaler.update(parameters: params)
    MLX.eval(upscaler.parameters())

    // Verify weights were loaded correctly
    let modelParams = Dictionary(uniqueKeysWithValues: upscaler.parameters().flattened())
    let modelKeyCount = modelParams.count
    LTXDebug.log("Upscaler model has \(modelKeyCount) parameter tensors, loaded \(sanitized.count) from file")

    // Check a specific weight to verify loading
    if let convWeight = modelParams["initial_conv.weight"] {
        LTXDebug.log("Verify initial_conv.weight: shape=\(convWeight.shape), mean=\(convWeight.mean().item(Float.self)), std=\(MLX.sqrt(MLX.variance(convWeight)).item(Float.self))")
    } else {
        LTXDebug.log("WARNING: initial_conv.weight not found in model parameters!")
    }

    // Check res_blocks loaded
    if let rbWeight = modelParams["res_blocks.0.conv1.weight"] {
        LTXDebug.log("Verify res_blocks.0.conv1.weight: shape=\(rbWeight.shape), mean=\(rbWeight.mean().item(Float.self))")
    } else {
        LTXDebug.log("WARNING: res_blocks.0.conv1.weight not found — array indexing may be broken!")
    }

    // Print all model parameter keys for debugging
    let sortedKeys = modelParams.keys.sorted()
    LTXDebug.log("Upscaler parameter keys (\(sortedKeys.count)):")
    for key in sortedKeys.prefix(10) {
        LTXDebug.log("  \(key): \(modelParams[key]!.shape)")
    }
    if sortedKeys.count > 10 {
        LTXDebug.log("  ... and \(sortedKeys.count - 10) more")
    }

    return upscaler
}

// MARK: - Upsample Latents Helper

/// Upsample latents using the spatial upscaler with proper denormalization
///
/// Flow: denormalize → upscale → renormalize
///
/// - Parameters:
///   - latent: Input latent (B, C, F, H, W)
///   - upscaler: Loaded SpatialUpscaler
///   - latentMean: Per-channel mean from VAE (mean-of-means)
///   - latentStd: Per-channel std from VAE (std-of-means)
/// - Returns: Upsampled latent (B, C, F, H*2, W*2)
public func upsampleLatents(
    _ latent: MLXArray,
    upscaler: SpatialUpscaler,
    latentMean: MLXArray,
    latentStd: MLXArray
) -> MLXArray {
    // Reshape stats for broadcasting: (C,) → (1, C, 1, 1, 1)
    let mean5d = latentMean.reshaped([1, -1, 1, 1, 1])
    let std5d = latentStd.reshaped([1, -1, 1, 1, 1])

    // Denormalize: latent * std + mean
    var x = latent * std5d + mean5d

    // Upsample 2x spatially
    x = upscaler(x)

    // Renormalize: (latent - mean) / std
    x = (x - mean5d) / std5d

    return x
}
