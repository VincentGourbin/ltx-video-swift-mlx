// AudioVAE.swift - Audio VAE Decoder for LTX-2
// Matches Python AutoencoderKLLTX2Audio decoder architecture
// Decodes audio latents to stereo mel spectrograms
// Copyright 2025

import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - Audio Pixel Norm (Parameter-free)

/// Per-pixel RMS normalization across channels (no learnable parameters)
/// Matches Python LTX2AudioPixelNorm
class AudioPixelNorm: Module, UnaryLayer {
    let dim: Int
    let eps: Float

    init(dim: Int = 1, eps: Float = 1e-6) {
        self.dim = dim
        self.eps = eps
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let meanSq = MLX.mean(x * x, axis: dim, keepDims: true)
        return x / MLX.sqrt(meanSq + eps)
    }
}

// MARK: - Audio Causal Conv2d

/// 2D convolution with causal padding along the height axis
/// Matches Python LTX2AudioCausalConv2d with causality_axis="height"
///
/// For causality_axis="height", all temporal padding goes to the top (before),
/// ensuring the output at row t only sees rows <= t.
/// Width gets standard symmetric padding.
class AudioCausalConv2d: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d

    let kernelSize: (Int, Int)
    let padding: (Int, Int, Int, Int)  // (left, right, top, bottom)

    init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int = 3,
        stride: Int = 1,
        dilation: Int = 1,
        bias: Bool = true
    ) {
        self.kernelSize = (kernelSize, kernelSize)

        let padH = (kernelSize - 1) * dilation
        let padW = (kernelSize - 1) * dilation

        // causality_axis="height": all H padding to top, symmetric W padding
        self.padding = (padW / 2, padW - padW / 2, padH, 0)

        self._conv.wrappedValue = Conv2d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: .init((kernelSize, kernelSize)),
            stride: .init((stride, stride)),
            padding: .init((0, 0)),
            bias: bias
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Input: (B, C, H, W) in PyTorch format
        // MLX Conv2d expects (B, H, W, C)
        var input = x.transposed(0, 2, 3, 1)  // (B, H, W, C)

        // Apply causal padding: (left, right, top, bottom) = (padW/2, padW/2, padH, 0)
        let (padLeft, padRight, padTop, padBottom) = padding

        if padTop > 0 || padBottom > 0 {
            let hPadTop = MLXArray.zeros(
                [input.dim(0), padTop, input.dim(2), input.dim(3)]
            )
            if padBottom > 0 {
                let hPadBot = MLXArray.zeros(
                    [input.dim(0), padBottom, input.dim(2), input.dim(3)]
                )
                input = MLX.concatenated([hPadTop, input, hPadBot], axis: 1)
            } else {
                input = MLX.concatenated([hPadTop, input], axis: 1)
            }
        }

        if padLeft > 0 || padRight > 0 {
            let wPadLeft = MLXArray.zeros(
                [input.dim(0), input.dim(1), padLeft, input.dim(3)]
            )
            let wPadRight = MLXArray.zeros(
                [input.dim(0), input.dim(1), padRight, input.dim(3)]
            )
            input = MLX.concatenated([wPadLeft, input, wPadRight], axis: 2)
        }

        let out = conv(input)  // (B, H_out, W_out, C_out) in MLX format
        return out.transposed(0, 3, 1, 2)  // Back to (B, C, H, W)
    }
}

// MARK: - Audio Resnet Block

/// Residual block for the audio VAE
/// Matches Python LTX2AudioResnetBlock with pixel norm
///
/// Architecture: PixelNorm -> SiLU -> Conv -> PixelNorm -> SiLU -> Conv + residual
class AudioResnetBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: AudioPixelNorm
    @ModuleInfo(key: "conv1") var conv1: AudioCausalConv2d
    @ModuleInfo(key: "norm2") var norm2: AudioPixelNorm
    @ModuleInfo(key: "conv2") var conv2: AudioCausalConv2d
    @ModuleInfo(key: "nin_shortcut") var ninShortcut: AudioCausalConv2d?

    let inChannels: Int
    let outChannels: Int

    init(inChannels: Int, outChannels: Int? = nil) {
        self.inChannels = inChannels
        self.outChannels = outChannels ?? inChannels

        self._norm1.wrappedValue = AudioPixelNorm()
        self._conv1.wrappedValue = AudioCausalConv2d(
            inChannels: inChannels, outChannels: self.outChannels, kernelSize: 3
        )
        self._norm2.wrappedValue = AudioPixelNorm()
        self._conv2.wrappedValue = AudioCausalConv2d(
            inChannels: self.outChannels, outChannels: self.outChannels, kernelSize: 3
        )

        // Channel projection if dimensions differ
        if inChannels != self.outChannels {
            self._ninShortcut.wrappedValue = AudioCausalConv2d(
                inChannels: inChannels, outChannels: self.outChannels, kernelSize: 1
            )
        } else {
            self._ninShortcut.wrappedValue = nil
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = norm1(x)
        h = MLXNN.silu(h)
        h = conv1(h)

        h = norm2(h)
        h = MLXNN.silu(h)
        h = conv2(h)

        var residual = x
        if let shortcut = ninShortcut {
            residual = shortcut(x)
        }

        return h + residual
    }
}

// MARK: - Audio Upsample

/// 2x upsampling with causal convolution
/// Matches Python LTX2AudioUpsample
///
/// Interpolates 2x, applies causal conv, then trims first row (causal output adjustment)
class AudioUpsample: Module {
    @ModuleInfo(key: "conv") var conv: AudioCausalConv2d

    init(inChannels: Int) {
        self._conv.wrappedValue = AudioCausalConv2d(
            inChannels: inChannels, outChannels: inChannels, kernelSize: 3
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: (B, C, H, W) - upsample 2x in both spatial dims
        let b = x.dim(0)
        let c = x.dim(1)
        let h = x.dim(2)
        let w = x.dim(3)

        // Nearest-neighbor upsample 2x: repeat each pixel in H and W
        // (B, C, H, W) -> (B, C, H, 1, W, 1) -> (B, C, H, 2, W, 2) -> (B, C, 2H, 2W)
        var upsampled = x.reshaped([b, c, h, 1, w, 1])
        upsampled = MLX.broadcast(upsampled, to: [b, c, h, 2, w, 2])
        upsampled = upsampled.reshaped([b, c, h * 2, w * 2])

        // Causal conv
        var out = conv(upsampled)

        // Trim first row (causal output adjustment: 2H -> 2H-1)
        out = out[0..., 0..., 1..., 0...]

        return out
    }
}

// MARK: - Audio Decoder Up Level

/// One level of the decoder upsampling path
/// Contains 3 resblocks and an optional upsample layer
class AudioDecoderUpLevel: Module {
    @ModuleInfo(key: "block") var blocks: [AudioResnetBlock]
    @ModuleInfo(key: "upsample") var upsample: AudioUpsample?

    init(inChannels: Int, outChannels: Int, numBlocks: Int = 3, hasUpsample: Bool) {
        // First block may change channels
        var blockList: [AudioResnetBlock] = []
        blockList.append(AudioResnetBlock(inChannels: inChannels, outChannels: outChannels))
        for _ in 1..<numBlocks {
            blockList.append(AudioResnetBlock(inChannels: outChannels))
        }
        self._blocks.wrappedValue = blockList

        if hasUpsample {
            self._upsample.wrappedValue = AudioUpsample(inChannels: outChannels)
        } else {
            self._upsample.wrappedValue = nil
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for block in blocks {
            h = block(h)
        }
        if let up = upsample {
            h = up(h)
        }
        return h
    }
}

// MARK: - Audio VAE Decoder

/// Audio VAE decoder: latent -> mel spectrogram
/// Matches Python LTX2AudioDecoder
///
/// Architecture:
/// - conv_in: 8 -> 512
/// - mid: 2 resblocks (512)
/// - up[2]: 3 resblocks (512->512) + upsample 2x
/// - up[1]: 3 resblocks (512->256) + upsample 2x
/// - up[0]: 3 resblocks (256->128), no upsample
/// - norm_out + SiLU + conv_out: 128 -> 2 (stereo)
class AudioDecoder: Module {
    let baseChannels: Int
    let latentChannels: Int

    @ModuleInfo(key: "conv_in") var convIn: AudioCausalConv2d
    @ModuleInfo(key: "conv_out") var convOut: AudioCausalConv2d

    // Mid block
    @ModuleInfo(key: "mid") var mid: AudioDecoderMidBlock

    // Up levels (reversed: level 2=highest channels first)
    @ModuleInfo(key: "up") var upLevels: [AudioDecoderUpLevel]

    // Final norm
    @ModuleInfo(key: "norm_out") var normOut: AudioPixelNorm

    init(
        latentChannels: Int = 8,
        outputChannels: Int = 2,
        baseChannels: Int = 128,
        chMult: [Int] = [1, 2, 4],
        numResBlocks: Int = 2
    ) {
        self.baseChannels = baseChannels
        self.latentChannels = latentChannels

        let topChannels = baseChannels * chMult.last!  // 128 * 4 = 512

        // conv_in: latent_channels -> top_channels
        self._convIn.wrappedValue = AudioCausalConv2d(
            inChannels: latentChannels, outChannels: topChannels, kernelSize: 3
        )

        // Mid block
        self._mid.wrappedValue = AudioDecoderMidBlock(channels: topChannels)

        // Up levels — stored in Python level order [0, 1, 2]:
        //   up[0] = level 0: 256->128, no upsample (processed LAST)
        //   up[1] = level 1: 512->256, upsample (processed second)
        //   up[2] = level 2: 512->512, upsample (processed FIRST)
        // Forward pass traverses them reversed: up[2], up[1], up[0]
        let numLevels = chMult.count  // 3

        var levels: [AudioDecoderUpLevel] = []

        // Python builds levels in reversed order (2, 1, 0) to track block_in
        // We pre-compute the channels for each level
        var levelSpecs: [(inCh: Int, outCh: Int, hasUpsample: Bool)] = []
        var tempBlockIn = topChannels
        for iLevel in stride(from: numLevels - 1, through: 0, by: -1) {
            let blockOut = baseChannels * chMult[iLevel]
            let hasUpsample = (iLevel != 0)
            levelSpecs.append((inCh: tempBlockIn, outCh: blockOut, hasUpsample: hasUpsample))
            tempBlockIn = blockOut  // After the level, block_in becomes block_out
        }
        // levelSpecs is in reverse order [level2, level1, level0]
        // Reverse to get [level0, level1, level2] for storage
        levelSpecs.reverse()

        for spec in levelSpecs {
            levels.append(AudioDecoderUpLevel(
                inChannels: spec.inCh,
                outChannels: spec.outCh,
                numBlocks: numResBlocks + 1,  // decoder has num_res_blocks + 1
                hasUpsample: spec.hasUpsample
            ))
        }
        self._upLevels.wrappedValue = levels

        // Final norm + conv
        self._normOut.wrappedValue = AudioPixelNorm()
        self._convOut.wrappedValue = AudioCausalConv2d(
            inChannels: baseChannels, outChannels: outputChannels, kernelSize: 3
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: (B, latent_channels, T_latent, mel_bins/4)
        var h = convIn(x)

        // Mid block
        h = mid(h)
        eval(h)

        // Up path — traverse in reverse order (level 2, 1, 0) matching Python
        for i in stride(from: upLevels.count - 1, through: 0, by: -1) {
            h = upLevels[i](h)
            eval(h)
            Memory.clearCache()
            LTXDebug.log("Audio decoder up[\(i)]: \(h.shape)")
        }

        // Final
        h = normOut(h)
        h = MLXNN.silu(h)
        h = convOut(h)

        return h
    }
}

// MARK: - Audio Decoder Mid Block

/// Mid block with 2 resblocks (no attention)
class AudioDecoderMidBlock: Module {
    @ModuleInfo(key: "block_1") var block1: AudioResnetBlock
    // attn_1 is Identity in Python — we skip it entirely
    @ModuleInfo(key: "block_2") var block2: AudioResnetBlock

    init(channels: Int) {
        self._block1.wrappedValue = AudioResnetBlock(inChannels: channels)
        self._block2.wrappedValue = AudioResnetBlock(inChannels: channels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = block1(x)
        // attn_1 is Identity — skip
        h = block2(h)
        return h
    }
}

// MARK: - Audio VAE (Top-level)

/// Top-level Audio VAE model for LTX-2
/// Matches Python AutoencoderKLLTX2Audio (decoder only for generation)
///
/// Decodes audio latents to stereo mel spectrograms.
/// The mel spectrograms are then passed to the vocoder for waveform synthesis.
class AudioVAE: Module {
    let latentDownsampleFactor: Int = 4

    @ModuleInfo(key: "decoder") var decoder: AudioDecoder
    @ParameterInfo(key: "latents_mean") var latentsMean: MLXArray
    @ParameterInfo(key: "latents_std") var latentsStd: MLXArray

    init(
        latentChannels: Int = 8,
        outputChannels: Int = 2,
        baseChannels: Int = 128,
        chMult: [Int] = [1, 2, 4]
    ) {
        self._decoder.wrappedValue = AudioDecoder(
            latentChannels: latentChannels,
            outputChannels: outputChannels,
            baseChannels: baseChannels,
            chMult: chMult
        )

        // Per-channel statistics for denormalization (128 channels = latent_ch * mel_bins/4)
        self._latentsMean.wrappedValue = MLXArray.zeros([128])
        self._latentsStd.wrappedValue = MLXArray.ones([128])
    }

    /// Decode audio latents to mel spectrogram
    ///
    /// - Parameter latents: Audio latents (B, 8, T_latent, 16)
    /// - Returns: Stereo mel spectrogram (B, 2, T_mel, 64)
    func decode(_ latents: MLXArray) -> MLXArray {
        let b = latents.dim(0)
        let latentT = latents.dim(2)
        let latentMel = latents.dim(3)

        // Target output frames (causal adjustment)
        let targetFrames = max(latentT * latentDownsampleFactor - (latentDownsampleFactor - 1), 1)

        // Patchify: (B, 8, T, 16) -> (B, T, 8*16) = (B, T, 128)
        var sample = latents.transposed(0, 2, 1, 3)  // (B, T, 8, 16)
        sample = sample.reshaped([b, latentT, -1])  // (B, T, 128)

        // Denormalize per-channel
        let mean = latentsMean.reshaped([1, 1, 128])
        let std = latentsStd.reshaped([1, 1, 128])
        sample = sample * std + mean

        // Unpatchify: (B, T, 128) -> (B, 8, T, 16)
        sample = sample.reshaped([b, latentT, 8, latentMel])
        sample = sample.transposed(0, 2, 1, 3)  // (B, 8, T, 16)

        LTXDebug.log("Audio VAE decode input: \(sample.shape)")

        // Decode
        var output = decoder(sample)

        LTXDebug.log("Audio VAE raw output: \(output.shape)")

        // Crop/pad to target shape (B, 2, targetFrames, 64)
        let outT = output.dim(2)
        if outT > targetFrames {
            output = output[0..., 0..., 0..<targetFrames, 0...]
        }
        // Crop mel bins to 64 if needed
        let outMel = output.dim(3)
        if outMel > 64 {
            output = output[0..., 0..., 0..., 0..<64]
        }

        LTXDebug.log("Audio VAE output: \(output.shape)")
        return output
    }

    /// Sanitize weights from Diffusers safetensors format
    ///
    /// Handles Conv2d weight transposition:
    ///   PyTorch Conv2d:  (out_ch, in_ch, H, W)
    ///   MLX Conv2d:      (out_ch, H, W, in_ch)
    ///
    /// All CausalConv2d layers have a nested `.conv` module, so weight keys
    /// look like `decoder.conv_in.conv.weight`.
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]

        for (key, value) in weights {
            var newValue = value

            // Conv2d weight transposition: (out, in, H, W) -> (out, H, W, in)
            if key.hasSuffix(".conv.weight") && newValue.ndim == 4 {
                newValue = newValue.transposed(0, 2, 3, 1)
            }

            result[key] = newValue
        }

        return result
    }
}
