// AudioProcessor.swift - Mel spectrogram computation for audio encoding
// Copyright 2026

import AVFoundation
import Foundation
@preconcurrency import MLX

/// Computes mel spectrograms from audio waveforms for AudioVAE encoding.
///
/// Pipeline: waveform → STFT → mel filterbank → log scale → stereo padding
///
/// Parameters match the Lightricks LTX-2.3 audio configuration:
/// - Sample rate: 16000 Hz
/// - FFT size: 1024
/// - Hop length: 160
/// - Mel bins: 64
/// - Frequency range: 0–8000 Hz
///
/// ## Usage
/// ```swift
/// let processor = AudioProcessor()
/// let waveform = try await processor.loadAudio(from: videoPath)
/// let melSpec = processor.melSpectrogram(waveform)
/// let audioLatents = audioVAE.encode(melSpec)
/// ```
public class AudioProcessor {

    let sampleRate: Int
    let nFFT: Int
    let hopLength: Int
    let nMels: Int
    let fMin: Float
    let fMax: Float

    /// Precomputed mel filterbank (nMels, nFFT/2 + 1)
    let melFilterbank: MLXArray

    /// Precomputed Hann window (nFFT,)
    let window: MLXArray

    public init(
        sampleRate: Int = 16000,
        nFFT: Int = 1024,
        hopLength: Int = 160,
        nMels: Int = 64,
        fMin: Float = 0,
        fMax: Float = 8000
    ) {
        self.sampleRate = sampleRate
        self.nFFT = nFFT
        self.hopLength = hopLength
        self.nMels = nMels
        self.fMin = fMin
        self.fMax = fMax

        // Precompute mel filterbank (Slaney normalization)
        self.melFilterbank = Self.createMelFilterbank(
            nMels: nMels, nFFT: nFFT,
            sampleRate: sampleRate, fMin: fMin, fMax: fMax
        )

        // Precompute Hann window
        var hannValues = [Float](repeating: 0, count: nFFT)
        for i in 0..<nFFT {
            hannValues[i] = 0.5 * (1.0 - cos(2.0 * Float.pi * Float(i) / Float(nFFT)))
        }
        self.window = MLXArray(hannValues)
    }

    // MARK: - Public API

    /// Load audio from a video or audio file as a mono float32 waveform at 16kHz.
    ///
    /// - Parameter path: Path to video (.mp4) or audio (.wav, .m4a) file
    /// - Returns: Mono waveform as MLXArray of shape `(samples,)`
    public func loadAudio(from path: String) async throws -> MLXArray {
        let url = URL(fileURLWithPath: path)
        let asset = AVURLAsset(url: url)

        let audioTracks = try await asset.loadTracks(withMediaType: .audio)
        guard let audioTrack = audioTracks.first else {
            let duration = try await asset.load(.duration)
            let numSamples = Int(CMTimeGetSeconds(duration) * Double(sampleRate))
            return MLXArray.zeros([numSamples]).asType(.float32)
        }

        let reader = try AVAssetReader(asset: asset)
        let outputSettings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: sampleRate,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 32,
            AVLinearPCMIsFloatKey: true,
            AVLinearPCMIsNonInterleaved: false,
        ]
        let output = AVAssetReaderTrackOutput(track: audioTrack, outputSettings: outputSettings)
        reader.add(output)
        reader.startReading()

        var allSamples: [Float] = []
        while let sampleBuffer = output.copyNextSampleBuffer() {
            if let blockBuffer = CMSampleBufferGetDataBuffer(sampleBuffer) {
                let length = CMBlockBufferGetDataLength(blockBuffer)
                var data = [UInt8](repeating: 0, count: length)
                CMBlockBufferCopyDataBytes(blockBuffer, atOffset: 0, dataLength: length, destination: &data)

                let floatCount = length / MemoryLayout<Float>.size
                guard floatCount > 0 else { continue }
                data.withUnsafeBufferPointer { ptr in
                    guard let base = ptr.baseAddress else { return }
                    base.withMemoryRebound(to: Float.self, capacity: floatCount) { floatPtr in
                        allSamples.append(contentsOf: UnsafeBufferPointer(start: floatPtr, count: floatCount))
                    }
                }
            }
        }

        guard reader.status == .completed else {
            throw AudioProcessorError.extractionFailed(path)
        }

        return MLXArray(allSamples)
    }

    /// Compute a stereo mel spectrogram suitable for AudioVAE.encode().
    ///
    /// Mono input is duplicated to stereo. Output is log-scaled.
    ///
    /// - Parameter waveform: Mono waveform `(samples,)` float32 at 16kHz
    /// - Returns: Stereo mel spectrogram `(1, 2, T_mel, 64)` for AudioVAE
    public func melSpectrogram(_ waveform: MLXArray) throws -> MLXArray {
        // Compute mel spectrogram for mono signal
        let mel = try computeMelSpectrogram(waveform)  // (T_mel, nMels)

        // Duplicate mono → stereo: (1, 2, T_mel, nMels)
        let melBatched = mel.reshaped([1, 1, mel.dim(0), mel.dim(1)])
        let stereo = MLX.concatenated([melBatched, melBatched], axis: 1)

        return stereo
    }

    // MARK: - Internal

    /// Compute mel spectrogram: STFT → power spectrum → mel filterbank → log
    func computeMelSpectrogram(_ waveform: MLXArray) throws -> MLXArray {
        // Pad waveform to center the STFT frames (reflect padding)
        let padSize = nFFT / 2
        let padded = padReflect(waveform, padSize: padSize)

        // STFT via rfft on windowed frames
        let frames = try extractFrames(padded)  // (numFrames, nFFT)
        let windowed = frames * window      // Apply Hann window

        // rfft: (numFrames, nFFT) → (numFrames, nFFT/2+1) complex
        let spectrum = MLXFFT.rfft(windowed, axis: -1)

        // Power spectrum: |S|^2
        let real = spectrum.realPart()
        let imag = spectrum.imaginaryPart()
        let powerSpec = real * real + imag * imag  // (numFrames, nFFT/2+1)

        // Apply mel filterbank: (numFrames, nFFT/2+1) @ (nFFT/2+1, nMels) → (numFrames, nMels)
        let melSpec = MLX.matmul(powerSpec, melFilterbank.transposed(1, 0))

        // Log scale with floor to avoid log(0)
        let logMelSpec = MLX.log(MLX.maximum(melSpec, MLXArray(Float(1e-10))))

        return logMelSpec
    }

    /// Extract overlapping frames from a 1D signal
    private func extractFrames(_ signal: MLXArray) throws -> MLXArray {
        let signalLength = signal.dim(0)
        let numFrames = 1 + (signalLength - nFFT) / hopLength

        guard numFrames > 0 else {
            throw AudioProcessorError.audioTooShort(samples: signalLength, minimumSamples: nFFT)
        }

        // Build frame indices using MLX operations
        var frameArrays: [MLXArray] = []
        for i in 0..<numFrames {
            let start = i * hopLength
            frameArrays.append(signal[start..<(start + nFFT)])
        }

        return MLX.stacked(frameArrays, axis: 0)  // (numFrames, nFFT)
    }

    /// Reflect-pad a 1D array
    private func padReflect(_ array: MLXArray, padSize: Int) -> MLXArray {
        let n = array.dim(0)
        guard padSize > 0, n > 1 else { return array }

        // Left padding: reverse of [1..padSize]
        let leftPad = array[1...(min(padSize, n - 1))]
        let leftReversed = leftPad[.ellipsis, .stride(by: -1)]

        // Right padding: reverse of [n-padSize-1..n-2]
        let rightStart = max(0, n - padSize - 1)
        let rightPad = array[rightStart..<(n - 1)]
        let rightReversed = rightPad[.ellipsis, .stride(by: -1)]

        return MLX.concatenated([leftReversed, array, rightReversed])
    }

    // MARK: - Mel Filterbank (Slaney normalization)

    /// Create a Slaney-normalized mel filterbank matrix.
    ///
    /// - Returns: MLXArray of shape `(nMels, nFFT/2+1)`
    private static func createMelFilterbank(
        nMels: Int, nFFT: Int,
        sampleRate: Int, fMin: Float, fMax: Float
    ) -> MLXArray {
        let numBins = nFFT / 2 + 1

        // Compute mel center frequencies
        let melMin = hzToMel(fMin)
        let melMax = hzToMel(fMax)

        // nMels + 2 points (including edges)
        var melPoints = [Float](repeating: 0, count: nMels + 2)
        for i in 0..<(nMels + 2) {
            melPoints[i] = melMin + Float(i) * (melMax - melMin) / Float(nMels + 1)
        }

        // Convert back to Hz
        let hzPoints = melPoints.map { melToHz($0) }

        // Convert to FFT bin indices
        let binFreqs = hzPoints.map { $0 * Float(nFFT) / Float(sampleRate) }

        // Build filterbank matrix
        var filterbank = [[Float]](repeating: [Float](repeating: 0, count: numBins), count: nMels)

        for m in 0..<nMels {
            let fLeft = binFreqs[m]
            let fCenter = binFreqs[m + 1]
            let fRight = binFreqs[m + 2]

            // Slaney normalization: divide by bandwidth
            let enorm = 2.0 / (hzPoints[m + 2] - hzPoints[m])

            for k in 0..<numBins {
                let freq = Float(k)
                if freq >= fLeft && freq <= fCenter && fCenter > fLeft {
                    filterbank[m][k] = enorm * (freq - fLeft) / (fCenter - fLeft)
                } else if freq > fCenter && freq <= fRight && fRight > fCenter {
                    filterbank[m][k] = enorm * (fRight - freq) / (fRight - fCenter)
                }
            }
        }

        // Convert to MLXArray (nMels, numBins)
        let flat = filterbank.flatMap { $0 }
        return MLXArray(flat).reshaped([nMels, numBins])
    }

    /// Hz to mel (Slaney formula: linear below 1000 Hz, log above)
    private static func hzToMel(_ hz: Float) -> Float {
        let linearBoundary: Float = 1000.0
        let logStep: Float = 27.0 / log(6.4)

        if hz < linearBoundary {
            return hz * 3.0 / 200.0  // 1000 Hz = 15 mel
        } else {
            return 15.0 + log(hz / linearBoundary) * logStep
        }
    }

    /// Mel to Hz (inverse Slaney formula)
    private static func melToHz(_ mel: Float) -> Float {
        let linearBoundary: Float = 15.0  // mel value at 1000 Hz
        let logStep: Float = log(6.4) / 27.0

        if mel < linearBoundary {
            return mel * 200.0 / 3.0
        } else {
            return 1000.0 * exp((mel - linearBoundary) * logStep)
        }
    }
}

// MARK: - Errors

public enum AudioProcessorError: Error, LocalizedError, Sendable {
    case extractionFailed(String)
    case noAudioTrack(String)
    case audioTooShort(samples: Int, minimumSamples: Int)

    public var errorDescription: String? {
        switch self {
        case .extractionFailed(let path): return "Audio extraction failed for \(path)"
        case .noAudioTrack(let path): return "No audio track found in \(path)"
        case .audioTooShort(let samples, let minimum): return "Audio too short (\(samples) samples, minimum \(minimum) required for STFT)"
        }
    }
}
