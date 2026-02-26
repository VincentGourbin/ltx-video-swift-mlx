// VideoConvolution.swift - 3D Convolution for Video VAE
// Copyright 2025

import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - Padding Mode

/// Padding mode types for convolutions
enum PaddingModeType: String, Sendable {
    case zeros = "zeros"
    case reflect = "reflect"
    case replicate = "replicate"
}

// MARK: - Normalization Type

/// Normalization layer types for VAE
enum NormLayerType: String, Sendable {
    case groupNorm = "group_norm"
    case pixelNorm = "pixel_norm"
}

// MARK: - Pixel Norm

/// Per-pixel (per-location) RMS normalization layer
class PixelNorm: Module, UnaryLayer {
    let dim: Int
    let eps: Float

    init(dim: Int = 1, eps: Float = 1e-8) {
        self.dim = dim
        self.eps = eps
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let rms = MLX.sqrt(MLX.mean(x * x, axis: dim, keepDims: true) + eps)
        return x / rms
    }
}

// MARK: - Dual Conv3D

/// 3D convolution decomposed into 2D spatial + 1D temporal convolutions
///
/// This approach avoids native 3D convolutions by:
/// 1. Applying 2D conv across spatial dimensions for each frame
/// 2. Applying 1D conv across temporal dimension for each spatial location
class DualConv3d: Module {
    @ModuleInfo var conv1: Conv2d
    @ModuleInfo var conv2: Conv1d

    let kernelSize: (Int, Int, Int)
    let stride: (Int, Int, Int)
    let padding: (Int, Int, Int)

    init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: (Int, Int, Int),
        stride: (Int, Int, Int) = (1, 1, 1),
        padding: (Int, Int, Int) = (0, 0, 0),
        bias: Bool = true
    ) {
        self.kernelSize = kernelSize
        self.stride = stride
        self.padding = padding

        // Intermediate channels
        let intermediateChannels = max(outChannels, inChannels)

        // First conv: 2D spatial
        self._conv1.wrappedValue = Conv2d(
            inputChannels: inChannels,
            outputChannels: intermediateChannels,
            kernelSize: .init((kernelSize.1, kernelSize.2)),
            stride: .init((stride.1, stride.2)),
            padding: .init((padding.1, padding.2)),
            bias: bias
        )

        // Second conv: 1D temporal
        self._conv2.wrappedValue = Conv1d(
            inputChannels: intermediateChannels,
            outputChannels: outChannels,
            kernelSize: kernelSize.0,
            stride: stride.0,
            padding: padding.0,
            bias: bias
        )
    }

    func callAsFunction(_ x: MLXArray, skipTimeConv: Bool = false) -> MLXArray {
        let b = x.dim(0)
        let c = x.dim(1)
        let d = x.dim(2)
        let h = x.dim(3)
        let w = x.dim(4)

        // Step 1: 2D spatial convolution on each frame
        // Reshape: (B, C, D, H, W) -> (B*D, H, W, C) for MLX Conv2d
        var out = x.transposed(0, 2, 1, 3, 4)  // (B, D, C, H, W)
        out = out.reshaped([b * d, c, h, w])  // (B*D, C, H, W)
        out = out.transposed(0, 2, 3, 1)  // (B*D, H, W, C) - MLX format

        out = conv1(out)  // (B*D, H_out, W_out, C_inter)

        let hOut = out.dim(1)
        let wOut = out.dim(2)
        let cInter = out.dim(3)

        if skipTimeConv {
            // Reshape back
            out = out.transposed(0, 3, 1, 2)  // (B*D, C_inter, H_out, W_out)
            out = out.reshaped([b, d, cInter, hOut, wOut])
            out = out.transposed(0, 2, 1, 3, 4)  // (B, C_inter, D, H_out, W_out)
            return out
        }

        // Step 2: 1D temporal convolution
        // Reshape: (B*D, H_out, W_out, C_inter) -> (B*H*W, D, C_inter)
        out = out.transposed(0, 3, 1, 2)  // (B*D, C_inter, H_out, W_out)
        out = out.reshaped([b, d, cInter, hOut, wOut])
        out = out.transposed(0, 3, 4, 1, 2)  // (B, H_out, W_out, D, C_inter)
        out = out.reshaped([b * hOut * wOut, d, cInter])  // (B*H*W, D, C_inter)

        out = conv2(out)  // (B*H*W, D_out, C_out)

        let dOut = out.dim(1)
        let cOut = out.dim(2)

        // Reshape back
        out = out.reshaped([b, hOut, wOut, dOut, cOut])
        out = out.transposed(0, 4, 3, 1, 2)  // (B, C_out, D_out, H_out, W_out)

        return out
    }
}

// MARK: - Causal Conv3D

/// Causal 3D convolution with temporal causal padding
class CausalConv3d: Module {
    @ModuleInfo var conv: DualConv3d

    let timeKernelSize: Int

    init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int = 3,
        stride: (Int, Int, Int) = (1, 1, 1),
        bias: Bool = true
    ) {
        self.timeKernelSize = kernelSize

        // Spatial padding (symmetric)
        let heightPad = kernelSize / 2
        let widthPad = kernelSize / 2

        self._conv.wrappedValue = DualConv3d(
            inChannels: inChannels,
            outChannels: outChannels,
            kernelSize: (kernelSize, kernelSize, kernelSize),
            stride: stride,
            padding: (0, heightPad, widthPad),  // No temporal padding, we handle it causally
            bias: bias
        )
    }

    func callAsFunction(_ x: MLXArray, causal: Bool = true) -> MLXArray {
        var input = x

        if causal {
            // Causal: replicate first frame to fill temporal receptive field
            let firstFrame = input[0..., 0..., 0..<1, 0..., 0...]  // (B, C, 1, H, W)
            let padFrames = MLX.repeated(firstFrame, count: timeKernelSize - 1, axis: 2)
            input = MLX.concatenated([padFrames, input], axis: 2)
        } else {
            // Non-causal: symmetric padding with edge frames
            let padSize = (timeKernelSize - 1) / 2
            let numFrames = input.dim(2)
            let firstFrame = input[0..., 0..., 0..<1, 0..., 0...]
            let lastFrame = input[0..., 0..., (numFrames - 1)..<numFrames, 0..., 0...]
            let firstPad = MLX.repeated(firstFrame, count: padSize, axis: 2)
            let lastPad = MLX.repeated(lastFrame, count: padSize, axis: 2)
            input = MLX.concatenated([firstPad, input, lastPad], axis: 2)
        }

        return conv(input)
    }
}

// MARK: - Conv3D Full (PyTorch Compatible)

/// 3D convolution that stores full 3D weights (PyTorch compatible).
///
/// Uses 2D+1D decomposition for forward pass but stores weights in
/// standard 3D conv format (out_channels, in_channels, T, H, W).
/// This allows direct loading of PyTorch Conv3d weights from safetensors.
class Conv3dFull: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray?

    let inChannels: Int
    let outChannels: Int
    let kernelSize: (Int, Int, Int)
    let stride: (Int, Int, Int)
    let padding: (Int, Int, Int)
    let spatialPaddingMode: PaddingModeType

    init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int = 3,
        stride: (Int, Int, Int) = (1, 1, 1),
        padding: (Int, Int, Int) = (1, 1, 1),
        bias: Bool = true,
        spatialPaddingMode: PaddingModeType = .reflect
    ) {
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.kernelSize = (kernelSize, kernelSize, kernelSize)
        self.stride = stride
        self.padding = padding
        self.spatialPaddingMode = spatialPaddingMode

        // Weight shape: (out_channels, in_channels, T, H, W) - PyTorch format
        self._weight.wrappedValue = MLXArray.zeros([outChannels, inChannels, kernelSize, kernelSize, kernelSize])
        if bias {
            self._bias.wrappedValue = MLXArray.zeros([outChannels])
        } else {
            self._bias.wrappedValue = nil
        }
    }

    func callAsFunction(_ x: MLXArray, causal: Bool = false) -> MLXArray {
        let b = x.dim(0)
        let c = x.dim(1)
        let (kt, kh, kw) = kernelSize
        let p = padding.1  // Spatial padding (symmetric)

        var input = x

        // Apply spatial padding
        if p > 0 {
            switch spatialPaddingMode {
            case .zeros:
                // Zero padding
                let hPadTop = MLXArray.zeros(like: input[0..., 0..., 0..., 0..<p, 0...])
                let hPadBot = MLXArray.zeros(like: input[0..., 0..., 0..., 0..<p, 0...])
                input = MLX.concatenated([hPadTop, input, hPadBot], axis: 3)
                let wPadLeft = MLXArray.zeros(like: input[0..., 0..., 0..., 0..., 0..<p])
                let wPadRight = MLXArray.zeros(like: input[0..., 0..., 0..., 0..., 0..<p])
                input = MLX.concatenated([wPadLeft, input, wPadRight], axis: 4)
            case .reflect:
                // Reflect: mirror interior neighbors (row 1 at top, row h-2 at bottom for p=1)
                let h = input.dim(3)
                let w = input.dim(4)
                let hPadTop = input[0..., 0..., 0..., 1..<(1 + p), 0...]
                let hPadBot = input[0..., 0..., 0..., (h - 1 - p)..<(h - 1), 0...]
                input = MLX.concatenated([hPadTop, input, hPadBot], axis: 3)
                let wPadLeft = input[0..., 0..., 0..., 0..., 1..<(1 + p)]
                let wPadRight = input[0..., 0..., 0..., 0..., (w - 1 - p)..<(w - 1)]
                input = MLX.concatenated([wPadLeft, input, wPadRight], axis: 4)
            case .replicate:
                // Replicate: repeat edge pixels
                let h = input.dim(3)
                let hPadTop = MLX.repeated(input[0..., 0..., 0..., 0..<1, 0...], count: p, axis: 3)
                let hPadBot = MLX.repeated(input[0..., 0..., 0..., (h - 1)..<h, 0...], count: p, axis: 3)
                input = MLX.concatenated([hPadTop, input, hPadBot], axis: 3)
                let newW = input.dim(4)
                let wPadLeft = MLX.repeated(input[0..., 0..., 0..., 0..., 0..<1], count: p, axis: 4)
                let wPadRight = MLX.repeated(input[0..., 0..., 0..., 0..., (newW - 1)..<newW], count: p, axis: 4)
                input = MLX.concatenated([wPadLeft, input, wPadRight], axis: 4)
            }
        }

        // Temporal padding
        let tPadNeeded = kt - 1
        if causal && tPadNeeded > 0 {
            // Causal: replicate first frame
            let firstFrames = MLX.repeated(input[0..., 0..., 0..<1, 0..., 0...], count: tPadNeeded, axis: 2)
            input = MLX.concatenated([firstFrames, input], axis: 2)
        } else if tPadNeeded > 0 {
            // Non-causal: symmetric padding with frame replication
            let padBefore = tPadNeeded / 2
            let padAfter = tPadNeeded - padBefore
            let tDim = input.dim(2)
            let firstFrames = MLX.repeated(input[0..., 0..., 0..<1, 0..., 0...], count: padBefore, axis: 2)
            let lastFrames = MLX.repeated(input[0..., 0..., (tDim - 1)..<tDim, 0..., 0...], count: padAfter, axis: 2)
            input = MLX.concatenated([firstFrames, input, lastFrames], axis: 2)
        }

        let tPad = input.dim(2)
        let hPad = input.dim(3)
        let wPad = input.dim(4)

        // Output spatial dimensions after 2D conv (no padding since we pre-padded)
        let hOut = hPad - kh + 1
        let wOut = wPad - kw + 1
        let tOutLen = tPad - kt + 1

        // Full 3D convolution via iterating over temporal kernel positions
        // For each temporal position kt, extract 2D weight slice and corresponding
        // input temporal slice, apply 2D conv, and accumulate results
        var output: MLXArray? = nil

        for kti in 0..<kt {
            // Extract 2D kernel slice: weight[:, :, kti, :, :] → (out_C, in_C, kH, kW)
            let wSlice = weight[0..., 0..., kti, 0..., 0...]
            let wSliceMLX = wSlice.transposed(0, 2, 3, 1)  // MLX format: (out_C, kH, kW, in_C)

            // Get temporal slice of input for this kernel position
            let xSlice = input[0..., 0..., kti..<(kti + tOutLen), 0..., 0...]  // (B, C, T_out, H_pad, W_pad)

            // Reshape for 2D conv: (B, C, T_out, H, W) -> (B*T_out, H, W, C)
            var x2D = xSlice.transposed(0, 2, 1, 3, 4)  // (B, T_out, C, H, W)
            x2D = x2D.reshaped([b * tOutLen, c, hPad, wPad])
            x2D = x2D.transposed(0, 2, 3, 1)  // (B*T_out, H, W, C)

            // Apply 2D spatial convolution
            let convOut = MLX.conv2d(x2D, wSliceMLX, stride: [stride.1, stride.2], padding: 0)
            // convOut: (B*T_out, H_out, W_out, C_out)

            let cOut = convOut.dim(3)

            // Reshape: (B*T_out, H_out, W_out, C_out) -> (B, C_out, T_out, H_out, W_out)
            var result = convOut.reshaped([b, tOutLen, hOut, wOut, cOut])
            result = result.transposed(0, 4, 1, 2, 3)  // (B, C_out, T_out, H_out, W_out)

            // Accumulate
            if let existing = output {
                output = existing + result
            } else {
                output = result
            }
        }

        // Add bias
        if let biasVal = bias {
            output = output! + biasVal.reshaped([1, -1, 1, 1, 1])
        }

        return output!
    }
}

// MARK: - Causal Conv3D Full (PyTorch Compatible)

/// Causal 3D convolution that stores full 3D weights (PyTorch compatible).
/// This is the main convolution class for VAE decoder weight loading.
class CausalConv3dFull: Module {
    @ModuleInfo(key: "conv") var conv: Conv3dFull

    let timeKernelSize: Int

    init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int = 3,
        stride: (Int, Int, Int) = (1, 1, 1),
        bias: Bool = true,
        spatialPaddingMode: PaddingModeType = .reflect
    ) {
        self.timeKernelSize = kernelSize

        // Spatial padding (symmetric) - handled in Conv3dFull
        let heightPad = kernelSize / 2
        let widthPad = kernelSize / 2

        self._conv.wrappedValue = Conv3dFull(
            inChannels: inChannels,
            outChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: (0, heightPad, widthPad),  // Temporal padding handled causally
            bias: bias,
            spatialPaddingMode: spatialPaddingMode
        )
    }

    func callAsFunction(_ x: MLXArray, causal: Bool = true) -> MLXArray {
        return conv(x, causal: causal)
    }
}

// MARK: - Pointwise Conv3D

/// Pointwise (1x1x1) 3D convolution
class PointwiseConv3d: Module {
    @ModuleInfo var conv: Conv2d

    init(inChannels: Int, outChannels: Int, bias: Bool = true) {
        self._conv.wrappedValue = Conv2d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: 1,
            bias: bias
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0)
        let c = x.dim(1)
        let d = x.dim(2)
        let h = x.dim(3)
        let w = x.dim(4)

        // Reshape: (B, C, D, H, W) -> (B*D, H, W, C)
        var out = x.transposed(0, 2, 3, 4, 1)  // (B, D, H, W, C)
        out = out.reshaped([b * d, h, w, c])

        out = conv(out)  // (B*D, H, W, C_out)

        let cOut = out.dim(3)

        // Reshape back
        out = out.reshaped([b, d, h, w, cOut])
        out = out.transposed(0, 4, 1, 2, 3)  // (B, C_out, D, H, W)

        return out
    }
}
