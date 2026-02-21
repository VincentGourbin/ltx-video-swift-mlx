// VideoDecoder.swift - Video VAE Decoder for LTX-2
// Matches Python SimpleVideoDecoder architecture exactly
// Copyright 2025

import Foundation
@preconcurrency import MLX
import MLXNN
import MLXRandom

// MARK: - Sinusoidal Timestep Embedding

/// Create sinusoidal timestep embeddings (matching Python get_timestep_embedding)
public func getTimestepEmbedding(_ timesteps: MLXArray, embeddingDim: Int = 256) -> MLXArray {
    var t = timesteps
    if t.ndim == 0 {
        t = t.reshaped([1])
    }
    let halfDim = embeddingDim / 2
    let freqs = MLX.exp(
        -Float(log(10000.0)) * MLXArray(0..<halfDim).asType(.float32) / Float(halfDim)
    )
    let args = MLX.expandedDimensions(t, axis: 1) * MLX.expandedDimensions(freqs, axis: 0)
    return MLX.concatenated([MLX.cos(args), MLX.sin(args)], axis: -1)
}

// MARK: - Pixel Norm (Channel-wise)

/// Pixel normalization across channels (parameter-free)
func vaePixelNorm(_ x: MLXArray, eps: Float = 1e-8) -> MLXArray {
    let meanSquared = MLX.mean(x * x, axis: 1, keepDims: true)
    return x / MLX.sqrt(meanSquared + eps)
}

// MARK: - VAE Timestep Embedder

/// MLP for processing timestep embeddings
public class VAETimestepEmbedder: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    public init(hiddenDim: Int, outputDim: Int, inputDim: Int = 256) {
        self._linear1.wrappedValue = Linear(inputDim, hiddenDim)
        self._linear2.wrappedValue = Linear(hiddenDim, outputDim)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = linear1(x)
        h = MLXNN.silu(h)
        h = linear2(h)
        return h
    }
}

// MARK: - VAE Timestep Embedder Wrapper

/// Wrapper to match safetensors key nesting: time_embedder.timestep_embedder.linear_1.weight
public class VAETimestepEmbedderWrapper: Module {
    @ModuleInfo(key: "timestep_embedder") var timestepEmbedder: VAETimestepEmbedder

    public init(hiddenDim: Int, outputDim: Int, inputDim: Int = 256) {
        self._timestepEmbedder.wrappedValue = VAETimestepEmbedder(
            hiddenDim: hiddenDim, outputDim: outputDim, inputDim: inputDim
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return timestepEmbedder(x)
    }
}

// MARK: - VAE ResBlock 3D (with Scale/Shift Table)

/// 3D residual block with pixel norm and scale/shift conditioning
/// Matches Python ResBlock3d: PixelNorm -> AdaLN(scale/shift) -> SiLU -> Conv
public class VAEResBlock3d: Module {
    let channels: Int
    @ModuleInfo(key: "conv1") var conv1: CausalConv3dFull
    @ModuleInfo(key: "conv2") var conv2: CausalConv3dFull
    @ParameterInfo(key: "scale_shift_table") var scaleShiftTable: MLXArray

    public init(channels: Int) {
        self.channels = channels
        self._conv1.wrappedValue = CausalConv3dFull(
            inChannels: channels, outChannels: channels, kernelSize: 3
        )
        self._conv2.wrappedValue = CausalConv3dFull(
            inChannels: channels, outChannels: channels, kernelSize: 3
        )
        // (4, C): rows are shift1, scale1, shift2, scale2
        self._scaleShiftTable.wrappedValue = MLXArray.zeros([4, channels])
    }

    public func callAsFunction(
        _ x: MLXArray, causal: Bool = false, timeEmb: MLXArray? = nil
    ) -> MLXArray {
        let residual = x

        let shift1: MLXArray, scale1: MLXArray, shift2: MLXArray, scale2: MLXArray

        if let timeEmb = timeEmb {
            // time_emb shape: (B, 4*C) -> reshape to (B, 4, C)
            let b = timeEmb.dim(0)
            let te = timeEmb.reshaped([b, 4, channels])
            // Add to table: (1, 4, C) + (B, 4, C) -> (B, 4, C)
            let ssTable = MLX.expandedDimensions(scaleShiftTable, axis: 0) + te
            shift1 = ssTable[0..., 0, 0...].reshaped([b, channels, 1, 1, 1])
            scale1 = (ssTable[0..., 1, 0...] + 1).reshaped([b, channels, 1, 1, 1])
            shift2 = ssTable[0..., 2, 0...].reshaped([b, channels, 1, 1, 1])
            scale2 = (ssTable[0..., 3, 0...] + 1).reshaped([b, channels, 1, 1, 1])
        } else {
            shift1 = scaleShiftTable[0].reshaped([1, -1, 1, 1, 1])
            scale1 = (scaleShiftTable[1] + 1).reshaped([1, -1, 1, 1, 1])
            shift2 = scaleShiftTable[2].reshaped([1, -1, 1, 1, 1])
            scale2 = (scaleShiftTable[3] + 1).reshaped([1, -1, 1, 1, 1])
        }

        // Block 1: norm -> scale/shift -> activation -> conv
        var h = vaePixelNorm(x)
        h = h * scale1 + shift1
        h = MLXNN.silu(h)
        h = conv1(h, causal: causal)

        // Block 2: norm -> scale/shift -> activation -> conv
        h = vaePixelNorm(h)
        h = h * scale2 + shift2
        h = MLXNN.silu(h)
        h = conv2(h, causal: causal)

        return h + residual
    }
}

// MARK: - VAE ResBlock Group

/// Group of 5 residual blocks with shared timestep embedding
public class VAEResBlockGroup: Module {
    let channels: Int
    @ModuleInfo(key: "res_blocks") var resBlocks: [VAEResBlock3d]
    @ModuleInfo(key: "time_embedder") var timeEmbedder: VAETimestepEmbedderWrapper

    public init(channels: Int, numBlocks: Int = 5) {
        self.channels = channels
        self._resBlocks.wrappedValue = (0..<numBlocks).map { _ in
            VAEResBlock3d(channels: channels)
        }
        // Time embedder: 256 -> 256 -> 4*channels
        self._timeEmbedder.wrappedValue = VAETimestepEmbedderWrapper(
            hiddenDim: 256, outputDim: 4 * channels, inputDim: 256
        )
    }

    public func callAsFunction(
        _ x: MLXArray, causal: Bool = false, timestep: MLXArray? = nil
    ) -> MLXArray {
        // Compute time embedding once for all blocks
        var timeEmb: MLXArray? = nil
        if let timestep = timestep {
            let tEmb = getTimestepEmbedding(timestep, embeddingDim: 256)
            timeEmb = timeEmbedder(tEmb)  // (B, 4*channels)
        }

        var h = x
        for block in resBlocks {
            h = block(h, causal: causal, timeEmb: timeEmb)
        }
        return h
    }
}

// MARK: - VAE Depth-to-Space Upsample 3D

/// Upsample using depth-to-space (pixel shuffle) in 3D with residual connection
///
/// Factor (2,2,2) = 8x channel reduction (halved) + 2x upsample in T,H,W
/// Frame trimming: removes first frame after D2S when temporal factor > 1
/// This produces the correct 8*(T-1)+1 frame count formula
public class VAEDepthToSpaceUpsample3d: Module {
    let factor: (Int, Int, Int)
    let upscaleFactor: Int
    let outChannels: Int
    let channelRepeats: Int
    let useResidual: Bool

    @ModuleInfo(key: "conv") var conv: CausalConv3dFull

    public init(inChannels: Int, factor: (Int, Int, Int) = (2, 2, 2), residual: Bool = true) {
        self.factor = factor
        self.useResidual = residual
        let (ft, fh, fw) = factor
        let factorProduct = ft * fh * fw
        self.upscaleFactor = 2
        self.outChannels = inChannels / upscaleFactor
        self.channelRepeats = factorProduct / upscaleFactor  // 8 / 2 = 4

        let convOutChannels = outChannels * factorProduct
        self._conv.wrappedValue = CausalConv3dFull(
            inChannels: inChannels, outChannels: convOutChannels, kernelSize: 3
        )
    }

    private func depthToSpace(_ x: MLXArray, cOut: Int) -> MLXArray {
        let b = x.dim(0)
        let t = x.dim(2)
        let h = x.dim(3)
        let w = x.dim(4)
        let (ft, fh, fw) = factor

        var out = x.reshaped([b, cOut, ft, fh, fw, t, h, w])
        out = out.transposed(0, 1, 5, 2, 6, 3, 7, 4)
        out = out.reshaped([b, cOut, t * ft, h * fh, w * fw])
        return out
    }

    public func callAsFunction(_ x: MLXArray, causal: Bool = false) -> MLXArray {
        let (ft, fh, fw) = factor
        let factorProduct = ft * fh * fw

        // Residual path: D2S on raw input, then tile channels
        var residualVal: MLXArray? = nil
        if useResidual {
            let cIn = x.dim(1)
            let cD2s = cIn / factorProduct
            var res = depthToSpace(x, cOut: cD2s)
            // Trim first frame when temporal factor > 1
            if ft > 1 {
                res = res[0..., 0..., 1..., 0..., 0...]
            }
            // Tile channels: concatenate copies along channel axis
            var parts = [MLXArray]()
            for _ in 0..<channelRepeats {
                parts.append(res)
            }
            residualVal = MLX.concatenated(parts, axis: 1)
        }

        // Main path: conv then D2S
        var h = conv(x, causal: causal)
        h = depthToSpace(h, cOut: outChannels)

        // Trim first frame
        if ft > 1 {
            h = h[0..., 0..., 1..., 0..., 0...]
        }

        // Add residual
        if let res = residualVal {
            h = h + res
        }

        return h
    }
}

// MARK: - Unpatchify

/// Unpatchify operation: expand spatial patches
public func unpatchify(
    _ x: MLXArray,
    patchSizeHW: Int = 4,
    patchSizeT: Int = 1
) -> MLXArray {
    let b = x.dim(0)
    let cPatched = x.dim(1)
    let t = x.dim(2)
    let h = x.dim(3)
    let w = x.dim(4)

    let c = cPatched / (patchSizeHW * patchSizeHW * patchSizeT)

    var out = x.reshaped([b, c, patchSizeT, patchSizeHW, patchSizeHW, t, h, w])
    out = out.transposed(0, 1, 5, 2, 6, 4, 7, 3)
    out = out.reshaped([b, c, t * patchSizeT, h * patchSizeHW, w * patchSizeHW])

    return out
}

// MARK: - Video Decoder

/// Video VAE Decoder for LTX-2 matching Python SimpleVideoDecoder
///
/// Architecture:
/// - conv_in: 128 -> 1024
/// - up_blocks_0: 5 res blocks (1024 ch) + timestep conditioning
/// - up_blocks_1: depth-to-space upsample (1024 -> 512, 2x2x2)
/// - up_blocks_2: 5 res blocks (512 ch)
/// - up_blocks_3: depth-to-space upsample (512 -> 256, 2x2x2)
/// - up_blocks_4: 5 res blocks (256 ch)
/// - up_blocks_5: depth-to-space upsample (256 -> 128, 2x2x2)
/// - up_blocks_6: 5 res blocks (128 ch)
/// - PixelNorm + AdaLN(last_scale_shift_table) + SiLU
/// - conv_out: 128 -> 48
/// - unpatchify: 48 -> 3 channels, 4x4 spatial
///
/// Frame formula: output_frames = 8 * (latent_frames - 1) + 1
public class VideoDecoder: Module {
    let patchSize: Int
    let causal: Bool
    let decodeNoiseScale: Float = 0.025
    /// Whether the VAE uses timestep conditioning (from config.json)
    public var timestepConditioning: Bool = false

    @ParameterInfo(key: "mean_of_means") var meanOfMeans: MLXArray
    @ParameterInfo(key: "std_of_means") var stdOfMeans: MLXArray
    @ParameterInfo(key: "timestep_scale_multiplier") var timestepScaleMultiplier: MLXArray
    @ParameterInfo(key: "last_scale_shift_table") var lastScaleShiftTable: MLXArray

    @ModuleInfo(key: "conv_in") var convIn: CausalConv3dFull
    @ModuleInfo(key: "conv_out") var convOut: CausalConv3dFull

    @ModuleInfo(key: "up_blocks_0") var upBlocks0: VAEResBlockGroup
    @ModuleInfo(key: "up_blocks_1") var upBlocks1: VAEDepthToSpaceUpsample3d
    @ModuleInfo(key: "up_blocks_2") var upBlocks2: VAEResBlockGroup
    @ModuleInfo(key: "up_blocks_3") var upBlocks3: VAEDepthToSpaceUpsample3d
    @ModuleInfo(key: "up_blocks_4") var upBlocks4: VAEResBlockGroup
    @ModuleInfo(key: "up_blocks_5") var upBlocks5: VAEDepthToSpaceUpsample3d
    @ModuleInfo(key: "up_blocks_6") var upBlocks6: VAEResBlockGroup

    @ModuleInfo(key: "last_time_embedder") var lastTimeEmbedder: VAETimestepEmbedderWrapper

    public init(patchSize: Int = 4, causal: Bool = false) {
        self.patchSize = patchSize
        self.causal = causal

        self._meanOfMeans.wrappedValue = MLXArray.zeros([128])
        self._stdOfMeans.wrappedValue = MLXArray.ones([128])
        self._timestepScaleMultiplier.wrappedValue = MLXArray(1000.0)
        self._lastScaleShiftTable.wrappedValue = MLXArray.zeros([2, 128])

        let actualOutChannels = 3 * patchSize * patchSize  // 48

        self._convIn.wrappedValue = CausalConv3dFull(
            inChannels: 128, outChannels: 1024, kernelSize: 3
        )
        self._convOut.wrappedValue = CausalConv3dFull(
            inChannels: 128, outChannels: actualOutChannels, kernelSize: 3
        )

        self._upBlocks0.wrappedValue = VAEResBlockGroup(channels: 1024, numBlocks: 5)
        self._upBlocks1.wrappedValue = VAEDepthToSpaceUpsample3d(
            inChannels: 1024, factor: (2, 2, 2), residual: true
        )
        self._upBlocks2.wrappedValue = VAEResBlockGroup(channels: 512, numBlocks: 5)
        self._upBlocks3.wrappedValue = VAEDepthToSpaceUpsample3d(
            inChannels: 512, factor: (2, 2, 2), residual: true
        )
        self._upBlocks4.wrappedValue = VAEResBlockGroup(channels: 256, numBlocks: 5)
        self._upBlocks5.wrappedValue = VAEDepthToSpaceUpsample3d(
            inChannels: 256, factor: (2, 2, 2), residual: true
        )
        self._upBlocks6.wrappedValue = VAEResBlockGroup(channels: 128, numBlocks: 5)

        // Last time embedder: 256 -> 256 -> 256 (output is 2*128=256 for scale+shift)
        self._lastTimeEmbedder.wrappedValue = VAETimestepEmbedderWrapper(
            hiddenDim: 256, outputDim: 256, inputDim: 256
        )
    }

    public func callAsFunction(_ sample: MLXArray, timestep: Float? = 0.05) -> MLXArray {
        let batchSize = sample.dim(0)

        LTXDebug.log("VAE Decoder input: \(sample.shape)")

        var x = sample

        // Step 1: Noise injection on NORMALIZED latent (before denormalization!)
        // Matching Python: noise is added in normalized space where values are ~N(0,1)
        var scaledTimestep: MLXArray? = nil
        if let ts = timestep {
            let noise = MLXRandom.normal(x.shape) * decodeNoiseScale
            x = noise + (1.0 - decodeNoiseScale) * x
            LTXDebug.log("After noise injection (normalized): mean=\(x.mean().item(Float.self))")

            let t = MLXArray(Array(repeating: ts, count: batchSize))
            scaledTimestep = t * timestepScaleMultiplier
        }

        // Step 2: Denormalize latent using per-channel statistics (AFTER noise)
        // Matching Python: sample = self.denormalize(sample)
        let meanExp = meanOfMeans.asType(.float32).reshaped([1, -1, 1, 1, 1])
        let stdExp = stdOfMeans.asType(.float32).reshaped([1, -1, 1, 1, 1])
        x = (x.asType(.float32) * stdExp + meanExp).asType(x.dtype)
        LTXDebug.log("After denormalize: mean=\(x.mean().item(Float.self))")

        // Conv in
        x = convIn(x, causal: causal)
        eval(x)
        LTXDebug.log("After conv_in: \(x.shape), mean=\(x.mean().item(Float.self))")

        // Up blocks with timestep conditioning
        x = upBlocks0(x, causal: causal, timestep: scaledTimestep)
        eval(x)
        LTXDebug.log("After up_blocks_0 (res): \(x.shape), mean=\(x.mean().item(Float.self))")

        x = upBlocks1(x, causal: causal)
        eval(x)
        LTXDebug.log("After up_blocks_1 (d2s): \(x.shape), mean=\(x.mean().item(Float.self))")

        x = upBlocks2(x, causal: causal, timestep: scaledTimestep)
        eval(x)
        LTXDebug.log("After up_blocks_2 (res): \(x.shape), mean=\(x.mean().item(Float.self))")

        x = upBlocks3(x, causal: causal)
        eval(x)
        LTXDebug.log("After up_blocks_3 (d2s): \(x.shape), mean=\(x.mean().item(Float.self))")

        x = upBlocks4(x, causal: causal, timestep: scaledTimestep)
        eval(x)
        LTXDebug.log("After up_blocks_4 (res): \(x.shape), mean=\(x.mean().item(Float.self))")

        x = upBlocks5(x, causal: causal)
        eval(x)
        LTXDebug.log("After up_blocks_5 (d2s): \(x.shape), mean=\(x.mean().item(Float.self))")

        x = upBlocks6(x, causal: causal, timestep: scaledTimestep)
        eval(x)
        LTXDebug.log("After up_blocks_6 (res): \(x.shape), mean=\(x.mean().item(Float.self))")

        // Final norm with timestep conditioning
        x = vaePixelNorm(x)

        let shift: MLXArray
        let scale: MLXArray
        if let st = scaledTimestep {
            let tEmb = getTimestepEmbedding(st, embeddingDim: 256)
            let timeEmb = lastTimeEmbedder(tEmb)  // (B, 256) = (B, 2*128)
            let te = timeEmb.reshaped([batchSize, 2, 128])
            let ssTable = MLX.expandedDimensions(lastScaleShiftTable, axis: 0) + te
            shift = ssTable[0..., 0, 0...].reshaped([batchSize, 128, 1, 1, 1])
            scale = (ssTable[0..., 1, 0...] + 1).reshaped([batchSize, 128, 1, 1, 1])
        } else {
            shift = lastScaleShiftTable[0].reshaped([1, -1, 1, 1, 1])
            scale = (lastScaleShiftTable[1] + 1).reshaped([1, -1, 1, 1, 1])
        }

        x = x * scale + shift
        x = MLXNN.silu(x)

        // Conv out
        x = convOut(x, causal: causal)
        eval(x)
        LTXDebug.log("After conv_out: \(x.shape), mean=\(x.mean().item(Float.self))")

        // Unpatchify: (B, 48, T, H, W) -> (B, 3, T, H*4, W*4)
        x = unpatchify(x, patchSizeHW: patchSize, patchSizeT: 1)
        eval(x)
        LTXDebug.log("After unpatchify: \(x.shape)")

        return x
    }
}

// MARK: - Decode Video

/// Decode a video latent tensor with the given decoder
public func decodeVideo(
    latent: MLXArray,
    decoder: VideoDecoder,
    timestep: Float? = 0.05
) -> MLXArray {
    var input = latent

    // Add batch dimension if needed
    if input.ndim == 4 {
        input = MLX.expandedDimensions(input, axis: 0)
    }

    // Decode: output is (B, C, F, H, W) with values roughly in [-1, 1]
    let decoded = decoder(input, timestep: timestep)

    LTXDebug.log("VAE raw output: mean=\(decoded.mean().item(Float.self)), min=\(decoded.min().item(Float.self)), max=\(decoded.max().item(Float.self))")

    // Normalize to [0, 1] range
    var frames = MLX.clip((decoded + 1.0) / 2.0, min: 0.0, max: 1.0)

    // Rearrange from (B, C, F, H, W) to (F, H, W, C)
    frames = frames[0]  // Remove batch dim
    frames = frames.transposed(1, 2, 3, 0)  // (C, F, H, W) -> (F, H, W, C)

    return frames
}
