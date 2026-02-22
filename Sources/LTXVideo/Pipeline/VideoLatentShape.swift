// VideoLatentShape.swift - Video Latent Types for LTX-2
// Copyright 2025

import Foundation
@preconcurrency import MLX

// MARK: - Spatio-Temporal Scale Factors

/// Scale factors for converting between pixel and latent space
struct SpatioTemporalScaleFactors: Sendable {
    /// Temporal scale factor (frames to latent temporal dimension)
    let time: Int

    /// Height scale factor
    let height: Int

    /// Width scale factor
    let width: Int

    /// Default LTX-2 scale factors
    /// Output: F = 8*(F'-1) + 1, H = 32*H', W = 32*W'
    static let `default` = SpatioTemporalScaleFactors(
        time: 8,
        height: 32,
        width: 32
    )

    init(time: Int, height: Int, width: Int) {
        self.time = time
        self.height = height
        self.width = width
    }

    /// Compute latent dimensions from pixel dimensions
    func pixelToLatent(frames: Int, height: Int, width: Int) -> (frames: Int, height: Int, width: Int) {
        // For LTX-2: F' = (F - 1) / time_scale + 1, but we use simplified version
        let latentFrames = (frames - 1) / time + 1
        let latentHeight = height / self.height
        let latentWidth = width / self.width
        return (latentFrames, latentHeight, latentWidth)
    }

    /// Compute pixel dimensions from latent dimensions
    func latentToPixel(frames: Int, height: Int, width: Int) -> (frames: Int, height: Int, width: Int) {
        let pixelFrames = (frames - 1) * time + 1
        let pixelHeight = height * self.height
        let pixelWidth = width * self.width
        return (pixelFrames, pixelHeight, pixelWidth)
    }
}

// MARK: - Video Latent Shape

/// Shape information for video latents
struct VideoLatentShape: Sendable {
    /// Batch size
    let batch: Int

    /// Number of latent channels (128 for LTX-2)
    let channels: Int

    /// Number of latent frames (temporal dimension)
    let frames: Int

    /// Latent height
    let height: Int

    /// Latent width
    let width: Int

    /// Total number of spatial-temporal tokens
    var tokenCount: Int {
        return frames * height * width
    }

    /// Shape as array [B, C, F, H, W]
    var shape: [Int] {
        return [batch, channels, frames, height, width]
    }

    /// Shape for patchified representation [B, T, C] where T = F*H*W
    var patchifiedShape: [Int] {
        return [batch, tokenCount, channels]
    }

    init(batch: Int, channels: Int, frames: Int, height: Int, width: Int) {
        self.batch = batch
        self.channels = channels
        self.frames = frames
        self.height = height
        self.width = width
    }

    /// Create from pixel dimensions
    static func fromPixelDimensions(
        batch: Int = 1,
        channels: Int = 128,
        frames: Int,
        height: Int,
        width: Int,
        scaleFactors: SpatioTemporalScaleFactors = .default
    ) -> VideoLatentShape {
        let latent = scaleFactors.pixelToLatent(frames: frames, height: height, width: width)
        return VideoLatentShape(
            batch: batch,
            channels: channels,
            frames: latent.frames,
            height: latent.height,
            width: latent.width
        )
    }

    /// Get corresponding pixel dimensions
    func pixelDimensions(
        scaleFactors: SpatioTemporalScaleFactors = .default
    ) -> (frames: Int, height: Int, width: Int) {
        return scaleFactors.latentToPixel(frames: frames, height: height, width: width)
    }
}

// MARK: - Validation

extension VideoLatentShape {
    /// Validate that dimensions are compatible with LTX-2 constraints
    func validate() throws {
        // Get pixel dimensions
        let (pixelFrames, pixelHeight, pixelWidth) = pixelDimensions()

        // Frame count must be 8n+1
        guard (pixelFrames - 1) % 8 == 0 else {
            throw LTXError.invalidConfiguration(
                "Frame count must be 8n+1 (got \(pixelFrames)). Valid: 9, 17, 25, 33, ..."
            )
        }

        // Height and width must be divisible by 32
        guard pixelHeight % 32 == 0 else {
            throw LTXError.invalidConfiguration(
                "Height must be divisible by 32 (got \(pixelHeight))"
            )
        }

        guard pixelWidth % 32 == 0 else {
            throw LTXError.invalidConfiguration(
                "Width must be divisible by 32 (got \(pixelWidth))"
            )
        }

        // Channel count
        guard channels == 128 else {
            throw LTXError.invalidConfiguration(
                "LTX-2 requires 128 latent channels (got \(channels))"
            )
        }
    }
}

// MARK: - Convenience Initializers

extension VideoLatentShape {
    /// Standard 480p video at ~5 seconds
    static let standard480p = VideoLatentShape.fromPixelDimensions(
        frames: 121,  // 8*15 + 1 = ~5 seconds at 24fps
        height: 480,
        width: 704
    )

    /// Standard 512x512 square video
    static let standard512 = VideoLatentShape.fromPixelDimensions(
        frames: 25,   // 8*3 + 1 = ~1 second at 24fps
        height: 512,
        width: 512
    )

    /// 768x512 landscape
    static let landscape768x512 = VideoLatentShape.fromPixelDimensions(
        frames: 49,   // 8*6 + 1 = ~2 seconds at 24fps
        height: 512,
        width: 768
    )
}
