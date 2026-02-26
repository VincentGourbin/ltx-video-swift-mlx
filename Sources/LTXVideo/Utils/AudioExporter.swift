// AudioExporter.swift - WAV Audio Export for LTX-2
// Exports MLXArray waveforms to WAV files
// Copyright 2025

import Foundation
@preconcurrency import MLX

// MARK: - Audio Exporter

/// Exports audio waveforms to WAV files
public enum AudioExporter {
    /// Export a stereo waveform to a WAV file
    ///
    /// - Parameters:
    ///   - waveform: Stereo waveform (B, 2, samples) or (2, samples), values in [-1, 1]
    ///   - sampleRate: Sample rate in Hz (e.g., 24000)
    ///   - path: Output file path (.wav)
    public static func exportToWAV(
        waveform: MLXArray,
        sampleRate: Int,
        path: String
    ) throws {
        // Extract stereo samples
        var audio = waveform
        if audio.ndim == 3 {
            audio = audio.squeezed(axis: 0)  // (2, samples)
        }

        let numChannels = audio.dim(0)  // Should be 2 for stereo
        let numSamples = audio.dim(1)

        // Convert to interleaved Int16 samples
        // Clamp to [-1, 1] and scale to Int16 range
        let clamped = MLX.clip(audio, min: -1.0, max: 1.0).asType(.float32)
        MLX.eval(clamped)

        // Interleave channels: [L0, R0, L1, R1, ...]
        var interleavedData = Data(capacity: numSamples * numChannels * 2)

        for i in 0..<numSamples {
            for ch in 0..<numChannels {
                let sample = clamped[ch, i].item(Float.self)
                let int16Sample = Int16(max(-32768, min(32767, sample * 32767.0)))
                var le = int16Sample.littleEndian
                interleavedData.append(Data(bytes: &le, count: 2))
            }
        }

        // Build WAV header
        let dataSize = UInt32(interleavedData.count)
        let fileSize = 36 + dataSize
        let bytesPerSample = UInt16(2)  // 16-bit
        let blockAlign = UInt16(numChannels) * bytesPerSample
        let byteRate = UInt32(sampleRate) * UInt32(blockAlign)

        var header = Data(capacity: 44)

        // RIFF header
        header.append(contentsOf: "RIFF".utf8)
        appendLittleEndian(&header, fileSize)
        header.append(contentsOf: "WAVE".utf8)

        // fmt chunk
        header.append(contentsOf: "fmt ".utf8)
        appendLittleEndian(&header, UInt32(16))           // chunk size
        appendLittleEndian(&header, UInt16(1))             // PCM format
        appendLittleEndian(&header, UInt16(numChannels))   // channels
        appendLittleEndian(&header, UInt32(sampleRate))    // sample rate
        appendLittleEndian(&header, byteRate)              // byte rate
        appendLittleEndian(&header, blockAlign)            // block align
        appendLittleEndian(&header, UInt16(16))            // bits per sample

        // data chunk
        header.append(contentsOf: "data".utf8)
        appendLittleEndian(&header, dataSize)

        // Write file
        let url = URL(fileURLWithPath: path)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        var fileData = header
        fileData.append(interleavedData)
        try fileData.write(to: url)

        let durationS = Double(numSamples) / Double(sampleRate)
        LTXDebug.log("Exported WAV: \(numChannels)ch, \(sampleRate)Hz, \(String(format: "%.1f", durationS))s, \(fileData.count) bytes → \(path)")
    }

    private static func appendLittleEndian(_ data: inout Data, _ value: UInt16) {
        var le = value.littleEndian
        data.append(Data(bytes: &le, count: 2))
    }

    private static func appendLittleEndian(_ data: inout Data, _ value: UInt32) {
        var le = value.littleEndian
        data.append(Data(bytes: &le, count: 4))
    }
}
