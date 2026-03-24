// VideoExporter.swift - MP4 Video Encoding for LTX-2
// Copyright 2025

import AVFoundation
import CoreMedia
import CoreImage
import Foundation
@preconcurrency import MLX

// MARK: - Video Export Configuration

/// Configuration for MP4 video encoding.
///
/// Controls codec, quality, frame rate, and pixel format used when
/// exporting generated frames to an MP4 file.
public struct VideoExportConfig: Sendable {
    /// Frames per second
    public var fps: Double

    /// Video codec
    public var codec: AVVideoCodecType

    /// Video quality (0.0 to 1.0). Used when `averageBitRate` is nil.
    public var quality: Float

    /// Target average bit rate in bits per second. When set, overrides `quality`.
    /// For example, 1_000_000 = 1 Mbps (similar to HuggingFace Space output).
    public var averageBitRate: Int?

    /// Output pixel format
    public var pixelFormat: OSType

    public init(
        fps: Double = 24.0,
        codec: AVVideoCodecType = .hevc,
        quality: Float = 0.9,
        averageBitRate: Int? = nil,
        pixelFormat: OSType = kCVPixelFormatType_32ARGB
    ) {
        self.fps = fps
        self.codec = codec
        self.quality = quality
        self.averageBitRate = averageBitRate
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
struct VideoExportFrames: Sendable {
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

/// Encodes video frames into MP4 files using AVFoundation.
///
/// `VideoExporter` handles the conversion from raw CGImage frames to
/// an H.264/HEVC-encoded MP4 file. For the simplest usage, call the
/// static convenience method ``exportVideo(frames:width:height:fps:to:)``
/// which handles the full pipeline from MLX tensor to MP4.
///
/// ## Quick Export
/// ```swift
/// let result = try await pipeline.generateVideo(prompt: "...", config: config)
/// try await VideoExporter.exportVideo(
///     frames: result.frames,
///     width: result.width,
///     height: result.height,
///     to: URL(fileURLWithPath: "output.mp4")
/// )
/// ```
///
/// ## Custom Export
/// ```swift
/// let exporter = VideoExporter(config: .highQuality)
/// let images = VideoExporter.tensorToImages(result.frames)
/// try await exporter.export(
///     frames: images,
///     width: result.width,
///     height: result.height,
///     to: URL(fileURLWithPath: "output.mp4")
/// )
/// ```
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
    func export(
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

    /// Export frames to MP4 file, optionally muxing audio into the same container
    ///
    /// When audio is provided, writes video to a temp file first, then muxes
    /// video + audio via AVMutableComposition to avoid AVAssetWriter interleaving deadlocks.
    ///
    /// - Parameters:
    ///   - frames: Array of CGImage frames
    ///   - width: Video width
    ///   - height: Video height
    ///   - fps: Frames per second
    ///   - audioSamples: Optional interleaved Int16 stereo PCM data
    ///   - audioSampleRate: Audio sample rate (default 24000)
    ///   - audioChannels: Number of audio channels (default 2)
    ///   - outputURL: Output file URL
    /// - Returns: URL to the exported video
    public func export(
        frames: [CGImage],
        width: Int,
        height: Int,
        fps: Double? = nil,
        audioSamples: Data? = nil,
        audioSampleRate: Int = 24000,
        audioChannels: Int = 2,
        to outputURL: URL
    ) async throws -> URL {
        guard !frames.isEmpty else {
            throw LTXError.invalidConfiguration("No frames to export")
        }

        // Remove existing file
        if FileManager.default.fileExists(atPath: outputURL.path) {
            try FileManager.default.removeItem(at: outputURL)
        }

        if let pcmData = audioSamples {
            // Two-pass approach: write video to temp, write audio to temp, mux together
            let tempDir = FileManager.default.temporaryDirectory
            let videoTempURL = tempDir.appendingPathComponent("ltx_video_\(UUID().uuidString).mp4")
            let audioTempURL = tempDir.appendingPathComponent("ltx_audio_\(UUID().uuidString).m4a")

            defer {
                try? FileManager.default.removeItem(at: videoTempURL)
                try? FileManager.default.removeItem(at: audioTempURL)
            }

            // 1. Write video-only
            try await writeVideoOnly(
                frames: frames, width: width, height: height,
                fps: fps, to: videoTempURL
            )

            // 2. Write audio-only
            try await writeAudioOnly(
                pcmData: pcmData, sampleRate: audioSampleRate,
                channels: audioChannels, to: audioTempURL
            )

            // 3. Mux video + audio into final output
            try await muxVideoAndAudio(
                videoURL: videoTempURL,
                audioURL: audioTempURL,
                to: outputURL
            )

            return outputURL
        } else {
            // Video-only export (original path)
            return try await writeVideoOnly(
                frames: frames, width: width, height: height,
                fps: fps, to: outputURL
            )
        }
    }

    /// Write video frames to an MP4 file (no audio)
    @discardableResult
    func writeVideoOnly(
        frames: [CGImage],
        width: Int,
        height: Int,
        fps: Double?,
        to outputURL: URL
    ) async throws -> URL {
        if FileManager.default.fileExists(atPath: outputURL.path) {
            try FileManager.default.removeItem(at: outputURL)
        }

        let writer = try AVAssetWriter(outputURL: outputURL, fileType: .mp4)

        var compressionProperties: [String: Any] = [:]
        if let bitRate = config.averageBitRate {
            compressionProperties[AVVideoAverageBitRateKey] = bitRate
        } else {
            compressionProperties[AVVideoQualityKey] = config.quality
        }

        let videoSettings: [String: Any] = [
            AVVideoCodecKey: config.codec,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: compressionProperties
        ]

        let writerInput = AVAssetWriterInput(
            mediaType: .video,
            outputSettings: videoSettings
        )
        writerInput.expectsMediaDataInRealTime = false

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

        let effectiveFps = fps ?? config.fps
        let frameDuration = CMTime(value: 1, timescale: CMTimeScale(effectiveFps))

        for (index, frame) in frames.enumerated() {
            while !writerInput.isReadyForMoreMediaData {
                try await Task.sleep(nanoseconds: 10_000_000)
            }

            guard let pixelBuffer = createPixelBuffer(from: frame, width: width, height: height)
            else {
                throw LTXError.exportFailed("Failed to create pixel buffer for frame \(index)")
            }

            let presentationTime = CMTimeMultiply(frameDuration, multiplier: Int32(index))

            if !adaptor.append(pixelBuffer, withPresentationTime: presentationTime) {
                throw LTXError.exportFailed(
                    "Failed to append frame \(index): \(writer.error?.localizedDescription ?? "unknown error")"
                )
            }
        }

        writerInput.markAsFinished()
        await writer.finishWriting()

        if let error = writer.error {
            throw LTXError.exportFailed("Video export failed: \(error.localizedDescription)")
        }

        return outputURL
    }

    /// Write PCM audio data to an M4A file (AAC encoded)
    private func writeAudioOnly(
        pcmData: Data,
        sampleRate: Int,
        channels: Int,
        to outputURL: URL
    ) async throws {
        if FileManager.default.fileExists(atPath: outputURL.path) {
            try FileManager.default.removeItem(at: outputURL)
        }

        let writer = try AVAssetWriter(outputURL: outputURL, fileType: .m4a)

        let audioSettings: [String: Any] = [
            AVFormatIDKey: kAudioFormatMPEG4AAC,
            AVSampleRateKey: sampleRate,
            AVNumberOfChannelsKey: channels,
            AVEncoderBitRateKey: 128000,
        ]

        let audioInput = AVAssetWriterInput(mediaType: .audio, outputSettings: audioSettings)
        audioInput.expectsMediaDataInRealTime = false
        writer.add(audioInput)
        writer.startWriting()
        writer.startSession(atSourceTime: .zero)

        // Create audio format description for PCM input
        let bytesPerFrame = 2 * channels  // Int16 per channel
        let totalFrames = pcmData.count / bytesPerFrame
        let chunkSize = 4096

        var asbd = AudioStreamBasicDescription(
            mSampleRate: Float64(sampleRate),
            mFormatID: kAudioFormatLinearPCM,
            mFormatFlags: kLinearPCMFormatFlagIsSignedInteger | kLinearPCMFormatFlagIsPacked,
            mBytesPerPacket: UInt32(bytesPerFrame),
            mFramesPerPacket: 1,
            mBytesPerFrame: UInt32(bytesPerFrame),
            mChannelsPerFrame: UInt32(channels),
            mBitsPerChannel: 16,
            mReserved: 0
        )

        var formatDescription: CMAudioFormatDescription?
        let fmtStatus = CMAudioFormatDescriptionCreate(
            allocator: kCFAllocatorDefault,
            asbd: &asbd,
            layoutSize: 0,
            layout: nil,
            magicCookieSize: 0,
            magicCookie: nil,
            extensions: nil,
            formatDescriptionOut: &formatDescription
        )
        guard fmtStatus == noErr, let fmt = formatDescription else {
            throw LTXError.exportFailed("Failed to create audio format description (status: \(fmtStatus))")
        }

        var frameOffset = 0
        while frameOffset < totalFrames {
            while !audioInput.isReadyForMoreMediaData {
                try await Task.sleep(nanoseconds: 10_000_000)
            }

            let framesInChunk = min(chunkSize, totalFrames - frameOffset)
            let byteOffset = frameOffset * bytesPerFrame
            let byteLength = framesInChunk * bytesPerFrame

            var blockBuffer: CMBlockBuffer?
            CMBlockBufferCreateWithMemoryBlock(
                allocator: kCFAllocatorDefault,
                memoryBlock: nil,
                blockLength: byteLength,
                blockAllocator: kCFAllocatorDefault,
                customBlockSource: nil,
                offsetToData: 0,
                dataLength: byteLength,
                flags: 0,
                blockBufferOut: &blockBuffer
            )
            guard let block = blockBuffer else {
                throw LTXError.exportFailed("Failed to create audio block buffer")
            }

            pcmData.withUnsafeBytes { rawBuf in
                let src = rawBuf.baseAddress!.advanced(by: byteOffset)
                CMBlockBufferReplaceDataBytes(
                    with: src,
                    blockBuffer: block,
                    offsetIntoDestination: 0,
                    dataLength: byteLength
                )
            }

            let presentationTime = CMTime(
                value: CMTimeValue(frameOffset),
                timescale: CMTimeScale(sampleRate)
            )

            var sampleBuffer: CMSampleBuffer?
            CMAudioSampleBufferCreateReadyWithPacketDescriptions(
                allocator: kCFAllocatorDefault,
                dataBuffer: block,
                formatDescription: fmt,
                sampleCount: framesInChunk,
                presentationTimeStamp: presentationTime,
                packetDescriptions: nil,
                sampleBufferOut: &sampleBuffer
            )
            guard let sb = sampleBuffer else {
                throw LTXError.exportFailed("Failed to create audio sample buffer")
            }

            if !audioInput.append(sb) {
                throw LTXError.exportFailed(
                    "Failed to append audio: \(writer.error?.localizedDescription ?? "unknown")"
                )
            }

            frameOffset += framesInChunk
        }

        audioInput.markAsFinished()
        await writer.finishWriting()

        if let error = writer.error {
            throw LTXError.exportFailed("Audio export failed: \(error.localizedDescription)")
        }
    }

    /// Mux a video file and an audio file into a single MP4 using AVMutableComposition
    func muxVideoAndAudio(
        videoURL: URL,
        audioURL: URL,
        to outputURL: URL
    ) async throws {
        let videoAsset = AVURLAsset(url: videoURL)
        let audioAsset = AVURLAsset(url: audioURL)

        let composition = AVMutableComposition()

        // Add video track
        guard let videoTrack = try await videoAsset.loadTracks(withMediaType: .video).first,
              let compositionVideoTrack = composition.addMutableTrack(
                  withMediaType: .video,
                  preferredTrackID: kCMPersistentTrackID_Invalid
              )
        else {
            throw LTXError.exportFailed("Failed to create composition video track")
        }

        let videoDuration = try await videoAsset.load(.duration)
        try compositionVideoTrack.insertTimeRange(
            CMTimeRange(start: .zero, duration: videoDuration),
            of: videoTrack,
            at: .zero
        )

        // Add audio track (skip if source has no audio)
        if let audioTrack = try await audioAsset.loadTracks(withMediaType: .audio).first,
           let compositionAudioTrack = composition.addMutableTrack(
               withMediaType: .audio,
               preferredTrackID: kCMPersistentTrackID_Invalid
           ) {
            let audioDuration = try await audioAsset.load(.duration)
            let insertDuration = CMTimeMinimum(videoDuration, audioDuration)
            try compositionAudioTrack.insertTimeRange(
                CMTimeRange(start: .zero, duration: insertDuration),
                of: audioTrack,
                at: .zero
            )
        }

        // Export the composition
        guard let exportSession = AVAssetExportSession(
            asset: composition,
            presetName: AVAssetExportPresetPassthrough
        ) else {
            throw LTXError.exportFailed("Failed to create export session")
        }

        exportSession.outputURL = outputURL
        exportSession.outputFileType = .mp4

        await exportSession.export()

        if exportSession.status != .completed {
            throw LTXError.exportFailed(
                "Muxing failed: \(exportSession.error?.localizedDescription ?? "unknown")"
            )
        }
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

        // Batch convert entire tensor to uint8 in a single GPU operation
        let allScaled = MLX.clip(videoTensor, min: 0, max: 1) * 255
        let allUint8 = allScaled.asType(.uint8)
        MLX.eval(allUint8)  // Single eval for all frames

        // CPU loop just extracts already-evaluated frames
        for f in 0..<numFrames {
            let frame = allUint8[f]
            if let image = createCGImage(from: frame, width: width, height: height) {
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

extension VideoExporter {
    /// Export an MLXArray video tensor directly to MP4, optionally with muxed audio
    ///
    /// Convenience method that converts a raw tensor to CGImages and exports
    /// to an MP4 file in a single call. Handles both 4D ``(F, H, W, C)`` and
    /// 5D ``(B, F, H, W, C)`` tensor layouts automatically.
    ///
    /// When `audioWaveform` is provided, the audio is muxed into the MP4 as a
    /// second track — no separate WAV file needed.
    ///
    /// - Parameters:
    ///   - frames: Video tensor of shape `(F, H, W, C)` or `(B, F, H, W, C)`, uint8 [0, 255]
    ///   - width: Video width in pixels
    ///   - height: Video height in pixels
    ///   - fps: Frames per second (default: 24.0)
    ///   - audioWaveform: Optional audio waveform `(B, 2, samples)` or `(2, samples)`, float32 [-1, 1]
    ///   - audioSampleRate: Audio sample rate in Hz (default: 24000)
    ///   - outputURL: Output file URL (must end in `.mp4`)
    /// - Returns: URL to the exported video file
    /// - Throws: ``LTXError/exportFailed(_:)`` if conversion or encoding fails
    /// - Parameter audioGain: Linear gain applied to audio waveform before export.
    ///   1.0 = no change, 0.5 = -6 dB, 0.25 = -12 dB. Default 1.0.
    public static func exportVideo(
        frames tensor: MLXArray,
        width: Int,
        height: Int,
        fps: Double = 24.0,
        audioWaveform: MLXArray? = nil,
        audioSampleRate: Int = 24000,
        audioGain: Float = 1.0,
        sourceAudioURL: URL? = nil,
        config: VideoExportConfig = .default,
        to outputURL: URL
    ) async throws -> URL {
        LTXDebug.log("exportVideo: converting tensor \(tensor.shape) to images...")
        let images = tensorToImages(tensor)
        LTXDebug.log("exportVideo: converted \(images.count) images")

        guard !images.isEmpty else {
            throw LTXError.exportFailed("Failed to convert tensor to images")
        }

        // Source audio passthrough: mux audio track directly from source file (no re-encode)
        if let audioSourceURL = sourceAudioURL {
            LTXDebug.log("exportVideo: using source audio passthrough from \(audioSourceURL.lastPathComponent)")
            let exporter = VideoExporter(config: config)
            // Write video only first
            let tempDir = FileManager.default.temporaryDirectory
            let videoTempURL = tempDir.appendingPathComponent("ltx_video_\(UUID().uuidString).mp4")
            defer { try? FileManager.default.removeItem(at: videoTempURL) }

            try await exporter.writeVideoOnly(
                frames: images, width: width, height: height,
                fps: fps, to: videoTempURL
            )
            // Mux with source audio track
            try await exporter.muxVideoAndAudio(
                videoURL: videoTempURL, audioURL: audioSourceURL, to: outputURL
            )
            return outputURL
        }

        // Convert audio waveform to interleaved Int16 PCM data
        let audioSamples: Data?
        let audioChannels: Int
        if let waveform = audioWaveform {
            // Apply gain before PCM conversion
            let gained = audioGain != 1.0 ? waveform * audioGain : waveform
            LTXDebug.log("exportVideo: converting audio waveform \(waveform.shape) to PCM (gain=\(audioGain))...")
            let (pcm, channels) = waveformToInterleavedPCM(gained)
            LTXDebug.log("exportVideo: PCM data \(pcm.count) bytes, \(channels) channels")
            audioSamples = pcm
            audioChannels = channels
        } else {
            audioSamples = nil
            audioChannels = 2
        }

        let exporter = VideoExporter(config: config)
        return try await exporter.export(
            frames: images,
            width: width,
            height: height,
            fps: fps,
            audioSamples: audioSamples,
            audioSampleRate: audioSampleRate,
            audioChannels: audioChannels,
            to: outputURL
        )
    }

    /// Convert an MLXArray waveform to interleaved Int16 PCM data
    ///
    /// - Parameter waveform: `(B, C, samples)` or `(C, samples)` float32 in [-1, 1]
    /// - Returns: Interleaved PCM Data and channel count
    private static func waveformToInterleavedPCM(_ waveform: MLXArray) -> (Data, Int) {
        var audio = waveform
        if audio.ndim == 3 {
            audio = audio.squeezed(axis: 0)  // (C, samples)
        }
        if audio.ndim == 1 {
            // Mono (samples,) → stereo (2, samples) by duplicating
            audio = MLX.stacked([audio, audio], axis: 0)
        }

        let numChannels = audio.dim(0)
        let numSamples = audio.dim(1)

        // Clamp and convert to float32 for sample extraction
        let clamped = MLX.clip(audio, min: -1.0, max: 1.0).asType(.float32)
        MLX.eval(clamped)

        // Build interleaved PCM: [L0, R0, L1, R1, ...]
        // Use explicit per-sample interleaving (matches AudioExporter.exportToWAV)
        var pcmData = Data(capacity: numSamples * numChannels * 2)
        for i in 0..<numSamples {
            for ch in 0..<numChannels {
                let sample = clamped[ch, i].item(Float.self)
                let int16Val = Int16(max(-32768, min(32767, sample * 32767.0)))
                var le = int16Val.littleEndian
                pcmData.append(Data(bytes: &le, count: 2))
            }
        }

        return (pcmData, numChannels)
    }

    /// Save a single frame as a PNG file
    ///
    /// - Parameters:
    ///   - image: The CGImage to save
    ///   - url: Destination file URL (should end in `.png`)
    /// - Throws: ``LTXError/exportFailed(_:)`` if image encoding fails
    public static func saveFrame(
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
}
