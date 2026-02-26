// VideoEncoder.swift - Video VAE Encoder for LTX-2
// Encodes images/video frames into latent space for image-to-video conditioning
// Copyright 2025

import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - Encoder Patchify

/// Patchify input pixels for the encoder: (B, 3, T, H, W) -> (B, 48, T, H/4, W/4)
/// Groups 4x4 spatial patches into 48 channels (3 * 4 * 4 = 48)
func encoderPatchify(_ x: MLXArray) -> MLXArray {
    let b = x.dim(0)
    let c = x.dim(1)  // 3
    let t = x.dim(2)
    let h = x.dim(3)
    let w = x.dim(4)

    let pH = 4
    let pW = 4

    // (B, 3, T, H/4, 4, W/4, 4) -> rearrange to (B, 3*4*4, T, H/4, W/4)
    var out = x.reshaped([b, c, t, h / pH, pH, w / pW, pW])
    // Transpose: (B, C, T, H/4, pH, W/4, pW) -> (B, C, pW, pH, T, H/4, W/4)
    // Must match Python permute(0,1,3,7,5,2,4,6) which puts pW before pH in channels
    out = out.transposed(0, 1, 6, 4, 2, 3, 5)
    // Reshape: (B, C*pH*pW, T, H/4, W/4) = (B, 48, T, H/4, W/4)
    out = out.reshaped([b, c * pH * pW, t, h / pH, w / pW])

    return out
}

// MARK: - Space-to-Depth (inverse of Depth-to-Space)

/// Space-to-depth operation: reduces spatial/temporal dimensions, increases channels
/// Inverse of the depthToSpace used in the decoder
func spaceToDepth(_ x: MLXArray, factor: (Int, Int, Int)) -> MLXArray {
    let (ft, fh, fw) = factor
    let b = x.dim(0)
    let c = x.dim(1)
    let t = x.dim(2)
    let h = x.dim(3)
    let w = x.dim(4)

    // Pad temporal if needed (causal: replicate first frame)
    var input = x
    var tPad = t
    if t % ft != 0 {
        let padT = ft - (t % ft)
        let firstFrame = input[0..., 0..., 0..<1, 0..., 0...]
        let padFrames = MLX.repeated(firstFrame, count: padT, axis: 2)
        input = MLX.concatenated([padFrames, input], axis: 2)
        tPad = input.dim(2)
    }

    // Reshape: (B, C, T/ft, ft, H/fh, fh, W/fw, fw)
    var out = input.reshaped([b, c, tPad / ft, ft, h / fh, fh, w / fw, fw])
    // Transpose to: (B, C, ft, fh, fw, T/ft, H/fh, W/fw)
    out = out.transposed(0, 1, 3, 5, 7, 2, 4, 6)
    // Reshape: (B, C*ft*fh*fw, T/ft, H/fh, W/fw)
    out = out.reshaped([b, c * ft * fh * fw, tPad / ft, h / fh, w / fw])

    return out
}

// MARK: - Encoder ResBlock 3D (no timestep conditioning)

/// Simple 3D residual block for the encoder
/// Unlike the decoder's VAEResBlock3d, this has no scale_shift_table or timestep conditioning
class EncoderResBlock3d: Module {
    let channels: Int
    @ModuleInfo(key: "conv1") var conv1: CausalConv3dFull
    @ModuleInfo(key: "conv2") var conv2: CausalConv3dFull

    init(channels: Int, spatialPaddingMode: PaddingModeType = .reflect) {
        self.channels = channels
        self._conv1.wrappedValue = CausalConv3dFull(
            inChannels: channels, outChannels: channels, kernelSize: 3,
            spatialPaddingMode: spatialPaddingMode
        )
        self._conv2.wrappedValue = CausalConv3dFull(
            inChannels: channels, outChannels: channels, kernelSize: 3,
            spatialPaddingMode: spatialPaddingMode
        )
    }

    func callAsFunction(_ x: MLXArray, causal: Bool = true) -> MLXArray {
        let residual = x
        var h = vaePixelNorm(x)
        h = MLXNN.silu(h)
        h = conv1(h, causal: causal)
        h = vaePixelNorm(h)
        h = MLXNN.silu(h)
        h = conv2(h, causal: causal)
        return h + residual
    }
}

// MARK: - Encoder ResBlock Group

/// Group of N encoder residual blocks (no timestep conditioning)
class EncoderResBlockGroup: Module {
    @ModuleInfo(key: "resnets") var resBlocks: [EncoderResBlock3d]

    init(channels: Int, numBlocks: Int, spatialPaddingMode: PaddingModeType = .reflect) {
        self._resBlocks.wrappedValue = (0..<numBlocks).map { _ in
            EncoderResBlock3d(channels: channels, spatialPaddingMode: spatialPaddingMode)
        }
    }

    func callAsFunction(_ x: MLXArray, causal: Bool = true) -> MLXArray {
        var h = x
        for block in resBlocks {
            h = block(h, causal: causal)
        }
        return h
    }
}

// MARK: - VAE Space-to-Depth Downsampler

/// Downsample using space-to-depth with convolution and residual connection
/// Inverse of the decoder's VAEDepthToSpaceUpsample3d
class VAESpaceToDepthDownsample3d: Module {
    let factor: (Int, Int, Int)
    let strideProduct: Int
    let targetChannels: Int

    @ModuleInfo(key: "conv") var conv: CausalConv3dFull

    init(inChannels: Int, outChannels: Int, factor: (Int, Int, Int), spatialPaddingMode: PaddingModeType = .reflect) {
        self.factor = factor
        let (ft, fh, fw) = factor
        self.strideProduct = ft * fh * fw
        self.targetChannels = outChannels

        // Conv output channels: enough so that after s2d we get outChannels
        let convOutChannels = outChannels / strideProduct
        self._conv.wrappedValue = CausalConv3dFull(
            inChannels: inChannels, outChannels: convOutChannels, kernelSize: 3,
            spatialPaddingMode: spatialPaddingMode
        )
    }

    func callAsFunction(_ x: MLXArray, causal: Bool = true) -> MLXArray {
        // Main path: conv -> space-to-depth
        let convOut = conv(x, causal: causal)
        let main = spaceToDepth(convOut, factor: factor)

        // Residual path: space-to-depth on raw input, then average groups to match target channels
        let s2dRes = spaceToDepth(x, factor: factor)
        let inChannelsAfterS2d = s2dRes.dim(1)
        let groupSize = inChannelsAfterS2d / targetChannels

        let b = s2dRes.dim(0)
        let t2 = s2dRes.dim(2)
        let h2 = s2dRes.dim(3)
        let w2 = s2dRes.dim(4)

        let reshaped = s2dRes.reshaped([b, targetChannels, groupSize, t2, h2, w2])
        let averaged = reshaped.mean(axis: 2)

        return main + averaged
    }
}

// MARK: - Encoder Down Block

/// A single encoder down block containing resnets and an optional downsampler
class EncoderDownBlock: Module {
    @ModuleInfo(key: "resnets") var resnets: EncoderResBlockGroup
    @ModuleInfo(key: "downsamplers") var downsampler: VAESpaceToDepthDownsample3d?

    init(inChannels: Int, outChannels: Int, numResnets: Int, downsampleFactor: (Int, Int, Int)?, spatialPaddingMode: PaddingModeType = .reflect) {
        self._resnets.wrappedValue = EncoderResBlockGroup(
            channels: inChannels, numBlocks: numResnets, spatialPaddingMode: spatialPaddingMode
        )
        if let factor = downsampleFactor {
            self._downsampler.wrappedValue = VAESpaceToDepthDownsample3d(
                inChannels: inChannels, outChannels: outChannels, factor: factor,
                spatialPaddingMode: spatialPaddingMode
            )
        } else {
            self._downsampler.wrappedValue = nil
        }
    }

    func callAsFunction(_ x: MLXArray, causal: Bool = true) -> MLXArray {
        var h = resnets(x, causal: causal)
        if let ds = downsampler {
            h = ds(h, causal: causal)
        }
        return h
    }
}

// MARK: - Video Encoder

/// Video VAE Encoder for LTX-2
///
/// Encodes video/image pixels into latent space for image-to-video conditioning.
/// Architecture mirrors the decoder in reverse:
/// - Patchify: (B, 3, T, H, W) -> (B, 48, T, H/4, W/4)
/// - conv_in: 48 -> 128
/// - 4 down blocks with progressively increasing channels (128 -> 256 -> 512 -> 1024 -> 2048)
/// - mid block: 2 resnets at 2048
/// - PerChannelRMSNorm + SiLU
/// - conv_out: 2048 -> 129
/// - Take first 128 channels (mean of the distribution, ignoring logvar)
public class VideoEncoder: Module {
    @ModuleInfo(key: "conv_in") var convIn: CausalConv3dFull
    @ModuleInfo(key: "conv_out") var convOut: CausalConv3dFull

    @ModuleInfo(key: "down_blocks_0") var downBlocks0: EncoderDownBlock
    @ModuleInfo(key: "down_blocks_1") var downBlocks1: EncoderDownBlock
    @ModuleInfo(key: "down_blocks_2") var downBlocks2: EncoderDownBlock
    @ModuleInfo(key: "down_blocks_3") var downBlocks3: EncoderDownBlock

    @ModuleInfo(key: "mid_block") var midBlock: EncoderResBlockGroup

    let causal: Bool

    public init(causal: Bool = true) {
        self.causal = causal

        // Encoder uses zero spatial padding (decoder uses reflect)
        let padMode: PaddingModeType = .zeros

        self._convIn.wrappedValue = CausalConv3dFull(
            inChannels: 48, outChannels: 128, kernelSize: 3,
            spatialPaddingMode: padMode
        )
        self._convOut.wrappedValue = CausalConv3dFull(
            inChannels: 2048, outChannels: 129, kernelSize: 3,
            spatialPaddingMode: padMode
        )

        // down_blocks.0: 4 resnets @ 128ch, downsample stride (1,2,2) -> 256ch
        self._downBlocks0.wrappedValue = EncoderDownBlock(
            inChannels: 128, outChannels: 256, numResnets: 4,
            downsampleFactor: (1, 2, 2), spatialPaddingMode: padMode
        )
        // down_blocks.1: 6 resnets @ 256ch, downsample stride (2,1,1) -> 512ch
        self._downBlocks1.wrappedValue = EncoderDownBlock(
            inChannels: 256, outChannels: 512, numResnets: 6,
            downsampleFactor: (2, 1, 1), spatialPaddingMode: padMode
        )
        // down_blocks.2: 6 resnets @ 512ch, downsample stride (2,2,2) -> 1024ch
        self._downBlocks2.wrappedValue = EncoderDownBlock(
            inChannels: 512, outChannels: 1024, numResnets: 6,
            downsampleFactor: (2, 2, 2), spatialPaddingMode: padMode
        )
        // down_blocks.3: 2 resnets @ 1024ch, downsample stride (2,2,2) -> 2048ch
        self._downBlocks3.wrappedValue = EncoderDownBlock(
            inChannels: 1024, outChannels: 2048, numResnets: 2,
            downsampleFactor: (2, 2, 2), spatialPaddingMode: padMode
        )

        // mid_block: 2 resnets @ 2048ch
        self._midBlock.wrappedValue = EncoderResBlockGroup(channels: 2048, numBlocks: 2, spatialPaddingMode: padMode)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        LTXDebug.log("VAE Encoder input: \(x.shape)")

        // 1. Patchify: (B, 3, T, H, W) -> (B, 48, T, H/4, W/4)
        var h = encoderPatchify(x)
        LTXDebug.log("After patchify: \(h.shape)")

        // 2. conv_in
        h = convIn(h, causal: causal)
        eval(h)
        LTXDebug.log("After conv_in: \(h.shape)")

        // 3. Down blocks
        h = downBlocks0(h, causal: causal)
        eval(h)
        LTXDebug.log("After down_blocks_0: \(h.shape)")

        h = downBlocks1(h, causal: causal)
        eval(h)
        LTXDebug.log("After down_blocks_1: \(h.shape)")

        h = downBlocks2(h, causal: causal)
        eval(h)
        LTXDebug.log("After down_blocks_2: \(h.shape)")

        h = downBlocks3(h, causal: causal)
        eval(h)
        LTXDebug.log("After down_blocks_3: \(h.shape)")

        // 4. Mid block
        h = midBlock(h, causal: causal)
        eval(h)
        LTXDebug.log("After mid_block: \(h.shape)")

        // 5. PerChannelRMSNorm (same as vaePixelNorm) + SiLU
        h = vaePixelNorm(h)
        h = MLXNN.silu(h)

        // 6. conv_out: 2048 -> 129
        h = convOut(h, causal: causal)
        eval(h)
        LTXDebug.log("After conv_out: \(h.shape)")

        // 7. Take first 128 channels = mean (ignore logvar channel)
        h = h[0..., 0..<128, 0..., 0..., 0...]
        LTXDebug.log("VAE Encoder output (mean): \(h.shape)")

        return h
    }
}
