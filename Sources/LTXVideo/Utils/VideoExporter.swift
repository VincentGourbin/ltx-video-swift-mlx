// VideoExporter.swift - MP4 Video Encoding for LTX-2
// Copyright 2025

import AVFoundation
import CoreImage
import Foundation
@preconcurrency import MLX

// MARK: - Video Export Configuration

/// Configuration for video export
public struct VideoExportConfig: Sendable {
    /// Frames per second
    public var fps: Double

    /// Video codec
    public var codec: AVVideoCodecType

    /// Video quality (0.0 to 1.0)
    public var quality: Float

    /// Output pixel format
    public var pixelFormat: OSType

    public init(
        fps: Double = 24.0,
        codec: AVVideoCodecType = .h264,
        quality: Float = 0.8,
        pixelFormat: OSType = kCVPixelFormatType_32ARGB
    ) {
        self.fps = fps
        self.codec = codec
        self.quality = quality
        self.pixelFormat = pixelFormat
    }

    /// Default configuration for LTX-2 output
    public static let `default` = VideoExportConfig()

    /// High quality configuration
    public static let highQuality = VideoExportConfig(
        fps: 24.0,
        codec: .hevc,
        quality: 0.95,
        pixelFormat: kCVPixelFormatType_32ARGB
    )
}

// MARK: - Video Exporter Result

/// Result of video export containing CGImage frames
public struct VideoExportFrames: Sendable {
    /// Generated frames as CGImages
    public let frames: [CGImage]

    /// Frame rate
    public let fps: Double

    /// Video width
    public let width: Int

    /// Video height
    public let height: Int

    /// Number of frames
    public var frameCount: Int { frames.count }

    /// Duration in seconds
    public var duration: Double { Double(frameCount) / fps }

    public init(frames: [CGImage], fps: Double = 24.0, width: Int, height: Int) {
        self.frames = frames
        self.fps = fps
        self.width = width
        self.height = height
    }
}

// MARK: - Video Exporter

/// Exports video frames to MP4 file
public actor VideoExporter {
    /// Export configuration
    private let config: VideoExportConfig

    public init(config: VideoExportConfig = .default) {
        self.config = config
    }

    /// Export frames to MP4 file
    ///
    /// - Parameters:
    ///   - result: Video export frames
    ///   - outputURL: Output file URL
    /// - Returns: URL to the exported video
    public func export(
        _ result: VideoExportFrames,
        to outputURL: URL
    ) async throws -> URL {
        return try await export(
            frames: result.frames,
            width: result.width,
            height: result.height,
            fps: result.fps,
            to: outputURL
        )
    }

    /// Export frames to MP4 file
    ///
    /// - Parameters:
    ///   - frames: Array of CGImage frames
    ///   - width: Video width
    ///   - height: Video height
    ///   - fps: Frames per second
    ///   - outputURL: Output file URL
    /// - Returns: URL to the exported video
    public func export(
        frames: [CGImage],
        width: Int,
        height: Int,
        fps: Double? = nil,
        to outputURL: URL
    ) async throws -> URL {
        guard !frames.isEmpty else {
            throw LTXError.invalidConfiguration("No frames to export")
        }

        // Remove existing file
        if FileManager.default.fileExists(atPath: outputURL.path) {
            try FileManager.default.removeItem(at: outputURL)
        }

        // Create asset writer
        let writer = try AVAssetWriter(outputURL: outputURL, fileType: .mp4)

        // Video settings
        let videoSettings: [String: Any] = [
            AVVideoCodecKey: config.codec,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: [
                AVVideoQualityKey: config.quality
            ]
        ]

        let writerInput = AVAssetWriterInput(
            mediaType: .video,
            outputSettings: videoSettings
        )
        writerInput.expectsMediaDataInRealTime = false

        // Pixel buffer adaptor
        let sourcePixelBufferAttributes: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: config.pixelFormat,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height
        ]

        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: writerInput,
            sourcePixelBufferAttributes: sourcePixelBufferAttributes
        )

        writer.add(writerInput)
        writer.startWriting()
        writer.startSession(atSourceTime: .zero)

        // Frame timing
        let effectiveFps = fps ?? config.fps
        let frameDuration = CMTime(value: 1, timescale: CMTimeScale(effectiveFps))

        // Write frames
        for (index, frame) in frames.enumerated() {
            // Wait for input to be ready
            while !writerInput.isReadyForMoreMediaData {
                try await Task.sleep(nanoseconds: 10_000_000)  // 10ms
            }

            // Create pixel buffer from CGImage
            guard let pixelBuffer = createPixelBuffer(from: frame, width: width, height: height)
            else {
                throw LTXError.exportFailed("Failed to create pixel buffer for frame \(index)")
            }

            // Calculate presentation time
            let presentationTime = CMTimeMultiply(frameDuration, multiplier: Int32(index))

            // Append pixel buffer
            if !adaptor.append(pixelBuffer, withPresentationTime: presentationTime) {
                throw LTXError.exportFailed(
                    "Failed to append frame \(index): \(writer.error?.localizedDescription ?? "unknown error")"
                )
            }
        }

        // Finish writing
        writerInput.markAsFinished()

        await writer.finishWriting()

        if let error = writer.error {
            throw LTXError.exportFailed("Video export failed: \(error.localizedDescription)")
        }

        return outputURL
    }

    /// Create pixel buffer from CGImage
    private func createPixelBuffer(
        from image: CGImage,
        width: Int,
        height: Int
    ) -> CVPixelBuffer? {
        var pixelBuffer: CVPixelBuffer?

        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ]

        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            config.pixelFormat,
            attrs as CFDictionary,
            &pixelBuffer
        )

        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

        guard let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        ) else {
            return nil
        }

        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        return buffer
    }
}

// MARK: - MLXArray to CGImage Conversion

extension VideoExporter {
    /// Convert MLXArray video tensor to CGImages
    ///
    /// - Parameter tensor: Video tensor of shape (F, H, W, C) or (B, F, H, W, C) with values in [0, 1]
    /// - Returns: Array of CGImages
    public static func tensorToImages(_ tensor: MLXArray) -> [CGImage] {
        var frames: [CGImage] = []

        // Handle batch dimension if present
        let videoTensor: MLXArray
        if tensor.ndim == 5 {
            // (B, F, H, W, C) -> take first batch
            videoTensor = tensor[0]
        } else if tensor.ndim == 4 {
            // (F, H, W, C)
            videoTensor = tensor
        } else {
            LTXDebug.log("Invalid tensor shape for video: \(tensor.shape)")
            return []
        }

        let numFrames = videoTensor.dim(0)
        let height = videoTensor.dim(1)
        let width = videoTensor.dim(2)

        for f in 0..<numFrames {
            // Extract frame (H, W, C)
            let frame = videoTensor[f]

            // Convert to UInt8 [0, 255]
            let clipped = MLX.clip(frame, min: 0, max: 1)
            let scaled = (clipped * 255).asType(.uint8)

            // Evaluate and get data
            MLX.eval(scaled)

            if let image = createCGImage(from: scaled, width: width, height: height) {
                frames.append(image)
            }
        }

        return frames
    }

    /// Create CGImage from MLXArray frame
    private static func createCGImage(
        from frame: MLXArray,
        width: Int,
        height: Int
    ) -> CGImage? {
        // Frame should be (H, W, C) with C = 3 (RGB)
        guard frame.ndim == 3, frame.dim(2) == 3 else {
            LTXDebug.log("Invalid frame shape: \(frame.shape)")
            return nil
        }

        // Get raw bytes - need to copy since MLXArray data may not persist
        let mlxData = frame.asData(access: .copy)

        // Extract Foundation Data from MLXArrayData
        let nsData = mlxData.data

        // Create data provider
        guard
            let provider = CGDataProvider(data: nsData as CFData)
        else {
            return nil
        }

        // Create CGImage
        return CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 24,
            bytesPerRow: width * 3,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        )
    }
}

// MARK: - Convenience Export Functions

/// Export MLXArray video tensor directly to MP4
///
/// - Parameters:
///   - tensor: Video tensor of shape (F, H, W, C) or (B, F, H, W, C)
///   - outputURL: Output file URL
///   - fps: Frames per second
/// - Returns: URL to exported video
public func exportVideo(
    _ tensor: MLXArray,
    to outputURL: URL,
    fps: Double = 24.0
) async throws -> URL {
    let frames = VideoExporter.tensorToImages(tensor)

    guard !frames.isEmpty else {
        throw LTXError.exportFailed("Failed to convert tensor to images")
    }

    let width = tensor.ndim == 5 ? tensor.dim(3) : tensor.dim(2)
    let height = tensor.ndim == 5 ? tensor.dim(2) : tensor.dim(1)

    let exporter = VideoExporter()
    return try await exporter.export(
        frames: frames,
        width: width,
        height: height,
        fps: fps,
        to: outputURL
    )
}

/// Save a single frame as PNG
public func saveFrame(
    _ image: CGImage,
    to url: URL
) throws {
    guard let destination = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil)
    else {
        throw LTXError.exportFailed("Failed to create image destination")
    }

    CGImageDestinationAddImage(destination, image, nil)

    if !CGImageDestinationFinalize(destination) {
        throw LTXError.exportFailed("Failed to write image to \(url.path)")
    }
}
