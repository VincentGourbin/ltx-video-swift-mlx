// Vocoder.swift - HiFi-GAN Vocoder for LTX-2 Audio
// Matches Python LTX2Vocoder from Diffusers
// Converts mel spectrograms to PCM waveforms at 24kHz
// Copyright 2025

import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - Vocoder ResBlock (HiFi-GAN)

/// HiFi-GAN residual block with multiple dilated convolutions
/// Matches Python ResBlock from vocoder.py
///
/// Each block has 3 dilated conv pairs (dilations [1,3,5]).
/// Each pair: LeakyReLU -> DilConv -> LeakyReLU -> Conv1d(dil=1) + residual
class VocoderResBlock: Module {
    @ModuleInfo(key: "convs1") var convs1: [Conv1d]
    @ModuleInfo(key: "convs2") var convs2: [Conv1d]

    let leakySlope: Float

    init(
        channels: Int,
        kernelSize: Int = 3,
        dilations: [Int] = [1, 3, 5],
        leakySlope: Float = 0.1
    ) {
        self.leakySlope = leakySlope

        var c1: [Conv1d] = []
        var c2: [Conv1d] = []

        for dil in dilations {
            // Dilated conv: "same" padding = dilation * (kernel_size - 1) / 2
            let pad = dil * (kernelSize - 1) / 2
            c1.append(Conv1d(
                inputChannels: channels,
                outputChannels: channels,
                kernelSize: kernelSize,
                padding: pad,
                dilation: dil
            ))
            // Fixed dilation=1 conv
            let pad2 = (kernelSize - 1) / 2
            c2.append(Conv1d(
                inputChannels: channels,
                outputChannels: channels,
                kernelSize: kernelSize,
                padding: pad2
            ))
        }

        self._convs1.wrappedValue = c1
        self._convs2.wrappedValue = c2
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Input: (B, T, C) in MLX format
        var h = x
        for i in 0..<convs1.count {
            let xt = MLXNN.leakyRelu(h, negativeSlope: leakySlope)
            let xt2 = convs1[i](xt)
            let xt3 = MLXNN.leakyRelu(xt2, negativeSlope: leakySlope)
            let xt4 = convs2[i](xt3)
            h = h + xt4  // Residual
        }
        return h
    }
}

// MARK: - Vocoder Upsampler (ConvTranspose1d wrapper)

/// Wraps MLXNN.ConvTransposed1d for vocoder upsampling
/// Handles PyTorch <-> MLX weight format conversion:
///   PyTorch: (in_channels, out_channels, kernel_size)
///   MLX:     (out_channels, kernel_size, in_channels)
///
/// Weight keys in safetensors use PyTorch format, which gets transposed
/// during weight loading via the `sanitize` method.
class VocoderUpsampler: Module {
    @ModuleInfo(key: "inner") var inner: ConvTransposed1d

    init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int,
        padding: Int
    ) {
        self._inner.wrappedValue = ConvTransposed1d(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: padding
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return inner(x)
    }

    /// Sanitize PyTorch weights to MLX format
    /// PyTorch: (in_ch, out_ch, K) -> MLX: (out_ch, K, in_ch)
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = weights
        if let w = result["inner.weight"] ?? result["weight"] {
            if w.ndim == 3 {
                // PyTorch (in_ch, out_ch, K) -> MLX (out_ch, K, in_ch)
                let key = result["inner.weight"] != nil ? "inner.weight" : "weight"
                result[key] = w.transposed(1, 2, 0)
            }
        }
        return result
    }
}

// MARK: - LTX2 Vocoder

/// HiFi-GAN style vocoder for LTX-2 audio
/// Matches Python LTX2Vocoder from Diffusers
///
/// Architecture:
/// - conv_in: Conv1d(128, 1024, k=7)
/// - 5 upsample stages (strides: 6,5,2,2,2 = 240x total)
///   - Each: LeakyReLU -> ConvTranspose1d -> 3 parallel ResBlocks (averaged)
/// - conv_out: Conv1d(32, 2, k=7) + tanh
///
/// Total upsample: 6*5*2*2*2 = 240x temporal expansion
/// Output: 24kHz stereo waveform
class LTX2Vocoder: Module {
    let outputSampleRate: Int

    @ModuleInfo(key: "conv_in") var convIn: Conv1d
    @ModuleInfo(key: "conv_out") var convOut: Conv1d
    @ModuleInfo(key: "upsamplers") var upsamplers: [VocoderUpsampler]
    @ModuleInfo(key: "resnets") var resnets: [VocoderResBlock]

    let leakySlope: Float
    let numUpsampleStages: Int
    let resnetsPerStage: Int

    init(
        inChannels: Int = 128,
        hiddenChannels: Int = 1024,
        outChannels: Int = 2,
        upsampleFactors: [Int] = [6, 5, 2, 2, 2],
        upsampleKernelSizes: [Int] = [16, 15, 8, 4, 4],
        resnetKernelSizes: [Int] = [3, 7, 11],
        resnetDilations: [[Int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        leakySlope: Float = 0.1,
        outputSampleRate: Int = 24000
    ) {
        self.leakySlope = leakySlope
        self.outputSampleRate = outputSampleRate
        self.numUpsampleStages = upsampleFactors.count
        self.resnetsPerStage = resnetKernelSizes.count

        // conv_in: (in_channels, hidden_channels, k=7, padding=3)
        self._convIn.wrappedValue = Conv1d(
            inputChannels: inChannels,
            outputChannels: hiddenChannels,
            kernelSize: 7,
            padding: 3
        )

        // Upsampler stages
        var ups: [VocoderUpsampler] = []
        var resBlocks: [VocoderResBlock] = []

        var ch = hiddenChannels
        for i in 0..<upsampleFactors.count {
            let outCh = ch / 2
            let stride = upsampleFactors[i]
            let kSize = upsampleKernelSizes[i]
            // Padding = (kSize - stride) / 2
            let pad = (kSize - stride) / 2

            ups.append(VocoderUpsampler(
                inputChannels: ch,
                outputChannels: outCh,
                kernelSize: kSize,
                stride: stride,
                padding: pad
            ))

            // 3 parallel ResBlocks for this stage
            for j in 0..<resnetKernelSizes.count {
                resBlocks.append(VocoderResBlock(
                    channels: outCh,
                    kernelSize: resnetKernelSizes[j],
                    dilations: resnetDilations[j],
                    leakySlope: leakySlope
                ))
            }

            ch = outCh
        }

        self._upsamplers.wrappedValue = ups
        self._resnets.wrappedValue = resBlocks

        // conv_out: (last_ch, out_channels, k=7, padding=3)
        self._convOut.wrappedValue = Conv1d(
            inputChannels: ch,  // 32
            outputChannels: outChannels,
            kernelSize: 7,
            padding: 3
        )
    }

    /// Convert mel spectrogram to waveform
    ///
    /// - Parameter melSpectrogram: Mel spectrogram (B, 2, T_mel, 64)
    /// - Returns: Stereo waveform (B, 2, audio_samples) at 24kHz
    func callAsFunction(_ melSpectrogram: MLXArray) -> MLXArray {
        var x = melSpectrogram

        // Transpose time and mel: (B, 2, T, 64) -> (B, 2, 64, T)
        x = x.transposed(0, 1, 3, 2)

        // Flatten channels and mel bins: (B, 2, 64, T) -> (B, 128, T)
        let b = x.dim(0)
        let t = x.dim(3)
        x = x.reshaped([b, 128, t])

        // Transpose to MLX format: (B, 128, T) -> (B, T, 128)
        x = x.transposed(0, 2, 1)

        // Pre-conv
        x = convIn(x)  // (B, T, 1024)

        // 5 upsample stages
        for i in 0..<numUpsampleStages {
            x = MLXNN.leakyRelu(x, negativeSlope: leakySlope)
            x = upsamplers[i](x)

            // 3 parallel ResBlocks, averaged
            let startIdx = i * resnetsPerStage
            let endIdx = startIdx + resnetsPerStage
            var resOutputs: [MLXArray] = []
            for j in startIdx..<endIdx {
                resOutputs.append(resnets[j](x))
            }
            // Average the outputs
            x = MLX.stacked(resOutputs, axis: 0).mean(axis: 0)

            eval(x)
            Memory.clearCache()
            LTXDebug.log("Vocoder stage \(i): \(x.shape)")
        }

        // Final: leaky_relu with default slope 0.01 (NOT 0.1)
        x = MLXNN.leakyRelu(x, negativeSlope: 0.01)
        x = convOut(x)
        x = MLX.tanh(x)

        // Transpose back: (B, T_audio, 2) -> (B, 2, T_audio)
        x = x.transposed(0, 2, 1)

        return x
    }

    /// Sanitize vocoder weights from Diffusers safetensors format
    ///
    /// Handles:
    /// - Conv1d weight transposition (PyTorch -> MLX format)
    /// - ConvTranspose1d weight transposition
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]

        for (key, value) in weights {
            var newKey = key
            var newValue = value

            // ConvTranspose1d weights need special transposition
            // PyTorch: (in_ch, out_ch, K) -> MLX: (out_ch, K, in_ch)
            if key.contains("upsamplers") && key.hasSuffix(".weight") && !key.contains("bias") {
                if newValue.ndim == 3 {
                    newValue = newValue.transposed(1, 2, 0)
                }
                // Add inner. prefix for VocoderUpsampler wrapper
                newKey = key.replacingOccurrences(
                    of: "upsamplers.",
                    with: "upsamplers."
                )
                // Insert "inner." before the final ".weight" or ".bias"
                if let range = newKey.range(of: ".weight", options: .backwards) {
                    newKey = newKey[..<range.lowerBound] + ".inner.weight"
                }
            } else if key.contains("upsamplers") && key.hasSuffix(".bias") {
                if let range = newKey.range(of: ".bias", options: .backwards) {
                    newKey = newKey[..<range.lowerBound] + ".inner.bias"
                }
            }

            // Conv1d weights: PyTorch (out, in, K) -> MLX (out, K, in)
            // MLX Conv1d expects the weight to be transposed
            if key.hasSuffix(".weight") && newValue.ndim == 3 && !key.contains("upsamplers") {
                newValue = newValue.transposed(0, 2, 1)
            }

            result[newKey] = newValue
        }

        return result
    }
}

// MARK: - Decode Audio

/// Decode audio latents to PCM waveform
///
/// Full pipeline: latent -> Audio VAE decoder -> mel spectrogram -> vocoder -> waveform
///
/// - Parameters:
///   - latents: Audio latents (B, 8, T_latent, 16)
///   - audioVAE: The audio VAE model
///   - vocoder: The HiFi-GAN vocoder
/// - Returns: Stereo waveform (B, 2, audio_samples) at 24kHz
func decodeAudio(
    latents: MLXArray,
    audioVAE: AudioVAE,
    vocoder: LTX2Vocoder
) -> MLXArray {
    var input = latents

    // Add batch dimension if needed
    if input.ndim == 3 {
        input = MLX.expandedDimensions(input, axis: 0)
    }

    LTXDebug.log("Audio decode: latent shape \(input.shape)")

    // Step 1: Decode latents to mel spectrogram
    let melSpectrogram = audioVAE.decode(input)
    eval(melSpectrogram)
    Memory.clearCache()

    LTXDebug.log("Mel spectrogram: \(melSpectrogram.shape)")

    // Step 2: Vocoder: mel -> waveform
    let waveform = vocoder(melSpectrogram)
    eval(waveform)
    Memory.clearCache()

    LTXDebug.log("Waveform: \(waveform.shape), sample rate: \(vocoder.outputSampleRate)Hz")

    return waveform
}
