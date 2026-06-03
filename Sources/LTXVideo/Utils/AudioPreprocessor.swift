// AudioPreprocessor.swift - Silence-aware time-stretching for LipDub target audio
// Copyright 2026

import AVFoundation
import Foundation
@preconcurrency import MLX

/// Aligns a target speech audio (e.g. TTS in a different language) to a source
/// video's speech-active window so LipDub can lip-sync without temporal mismatch.
///
/// LipDub conditions lip movements on the reference audio. When the target audio
/// has a different speech duration than the source video, naive muxing truncates
/// or stretches the audio, breaking lip-sync. This preprocessor:
/// 1. Detects the speech-active window in both source and target (RMS threshold).
/// 2. Time-stretches the target speech to match the source's speech duration,
///    preserving pitch via `AVAudioUnitTimePitch`.
/// 3. Pads with silence to reproduce the source's leading/trailing silence,
///    yielding an audio of the same total duration as the source video.
public enum AudioPreprocessor {

    // MARK: - Silence detection

    /// Detect the speech-active window `[startSample, endSample)` in a mono waveform.
    ///
    /// Uses fixed-size frame RMS vs a dBFS threshold to locate the first and last
    /// frame containing speech. Frames below threshold at the head/tail are treated
    /// as silence; mid-utterance pauses are NOT trimmed (only leading/trailing
    /// silence is removed). Returns `(0, count)` if no frame exceeds threshold.
    public static func detectSpeechWindow(
        waveform: [Float],
        sampleRate: Int,
        thresholdDB: Float = -35,
        frameMS: Int = 10
    ) -> (start: Int, end: Int) {
        let frameSamples = sampleRate * frameMS / 1000
        guard frameSamples > 0, waveform.count >= frameSamples else {
            return (0, waveform.count)
        }

        let threshold = pow(10.0, thresholdDB / 20.0)
        let numFrames = waveform.count / frameSamples

        var firstActive = -1
        var lastActive = -1
        for f in 0..<numFrames {
            var sumSq: Float = 0
            let base = f * frameSamples
            for i in 0..<frameSamples {
                let s = waveform[base + i]
                sumSq += s * s
            }
            let rms = sqrt(sumSq / Float(frameSamples))
            if rms > threshold {
                if firstActive < 0 { firstActive = f }
                lastActive = f
            }
        }

        if firstActive < 0 {
            return (0, waveform.count)
        }
        return (
            start: firstActive * frameSamples,
            end: min((lastActive + 1) * frameSamples, waveform.count)
        )
    }

    // MARK: - Time stretching (pitch-preserving)

    /// Time-stretch a mono float32 waveform with pitch preservation.
    ///
    /// Uses `AVAudioUnitTimePitch` in offline rendering mode (`AVAudioEngine`'s
    /// `enableManualRenderingMode(.offline)`). Output length is approximately
    /// `input.count / rate` samples.
    ///
    /// - Parameters:
    ///   - waveform: Input mono float32 samples
    ///   - sampleRate: Sample rate (Hz)
    ///   - rate: Speed multiplier. `>1` shortens output (faster speech),
    ///           `<1` lengthens it. Valid range: `[1/32, 32]`.
    /// - Returns: Time-stretched mono float32 samples.
    public static func timeStretch(
        waveform: [Float],
        sampleRate: Double,
        rate: Float
    ) throws -> [Float] {
        // No-op for negligible ratios
        if abs(rate - 1.0) < 0.005 { return waveform }
        guard waveform.count > 0 else { return waveform }

        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: 1,
            interleaved: false
        ) else {
            throw LTXError.invalidConfiguration("Failed to create audio format")
        }

        guard let inputBuffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: AVAudioFrameCount(waveform.count)
        ) else {
            throw LTXError.invalidConfiguration("Failed to allocate input buffer")
        }
        inputBuffer.frameLength = AVAudioFrameCount(waveform.count)
        let inChannel = inputBuffer.floatChannelData![0]
        waveform.withUnsafeBufferPointer { ptr in
            inChannel.update(from: ptr.baseAddress!, count: waveform.count)
        }

        let engine = AVAudioEngine()
        let player = AVAudioPlayerNode()
        let timePitch = AVAudioUnitTimePitch()
        timePitch.rate = rate

        engine.attach(player)
        engine.attach(timePitch)
        engine.connect(player, to: timePitch, format: format)
        engine.connect(timePitch, to: engine.mainMixerNode, format: format)

        let maxFrames: AVAudioFrameCount = 4096
        try engine.enableManualRenderingMode(.offline, format: format, maximumFrameCount: maxFrames)
        try engine.start()

        player.scheduleBuffer(inputBuffer, completionHandler: nil)
        player.play()

        let expectedLength = Int((Double(waveform.count) / Double(rate)).rounded())
        var output = [Float]()
        output.reserveCapacity(expectedLength + Int(maxFrames))

        guard let renderBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: maxFrames) else {
            throw LTXError.invalidConfiguration("Failed to allocate render buffer")
        }

        // Allow a small tail beyond expectedLength for AVAudioUnitTimePitch's
        // internal latency; we'll trim to expectedLength at the end.
        let renderBudget = expectedLength + Int(maxFrames) * 2
        var done = false
        while !done && output.count < renderBudget {
            let status = try engine.renderOffline(maxFrames, to: renderBuffer)
            switch status {
            case .success:
                let n = Int(renderBuffer.frameLength)
                if n == 0 { done = true; break }
                let chan = renderBuffer.floatChannelData![0]
                for i in 0..<n { output.append(chan[i]) }
            case .insufficientDataFromInputNode, .cannotDoInCurrentContext, .error:
                done = true
            @unknown default:
                done = true
            }
        }

        player.stop()
        engine.stop()

        if output.count > expectedLength {
            output.removeLast(output.count - expectedLength)
        }
        return output
    }

    // MARK: - Silence-aware alignment

    /// Align a target waveform's speech window to a source waveform's speech window.
    ///
    /// Builds an output waveform with:
    /// - The same total length as `source.count`.
    /// - Leading silence equal to the source's leading silence.
    /// - The target's speech (`target[targetStart..<targetEnd]`) time-stretched to
    ///   match the source's speech duration.
    /// - Trailing silence filling the remainder.
    ///
    /// The result is suitable as a LipDub `refWaveform` whose mouth-pose timing
    /// matches the source video frames it will be paired with.
    ///
    /// - Parameters:
    ///   - source: Mono source audio (for timing reference only).
    ///   - target: Mono target audio (the speech content to lip-sync to).
    ///   - sampleRate: Sample rate (Hz) — must match both inputs.
    ///   - thresholdDB: RMS threshold for silence detection (dBFS).
    /// - Returns: `(alignedWaveform, ratio, sourceWindow, targetWindow)` where ratio
    ///   is the stretch factor that was applied (1.0 means no stretch).
    public static func alignTargetToSource(
        source: [Float],
        target: [Float],
        sampleRate: Int,
        thresholdDB: Float = -35
    ) throws -> (waveform: [Float], rate: Float, sourceWindow: (Int, Int), targetWindow: (Int, Int)) {
        let srcWin = detectSpeechWindow(waveform: source, sampleRate: sampleRate, thresholdDB: thresholdDB)
        let tgtWin = detectSpeechWindow(waveform: target, sampleRate: sampleRate, thresholdDB: thresholdDB)

        let srcSpeechSamples = srcWin.end - srcWin.start
        let tgtSpeechSamples = tgtWin.end - tgtWin.start
        guard srcSpeechSamples > 0, tgtSpeechSamples > 0 else {
            throw LTXError.invalidConfiguration(
                "Speech window detection failed (source=\(srcSpeechSamples) target=\(tgtSpeechSamples) samples). " +
                "Lower the silence threshold or supply audio with detectable speech."
            )
        }

        let rate = Float(tgtSpeechSamples) / Float(srcSpeechSamples)
        let targetSpeech = Array(target[tgtWin.start..<tgtWin.end])
        let stretched = try timeStretch(
            waveform: targetSpeech,
            sampleRate: Double(sampleRate),
            rate: rate
        )

        // Reassemble: leading silence + stretched speech + trailing silence,
        // total length matches source.count.
        var aligned = [Float](repeating: 0, count: source.count)
        let placeStart = srcWin.start
        let placeEnd = min(placeStart + stretched.count, aligned.count)
        let copyCount = placeEnd - placeStart
        if copyCount > 0 {
            stretched.withUnsafeBufferPointer { src in
                aligned.withUnsafeMutableBufferPointer { dst in
                    (dst.baseAddress! + placeStart).update(from: src.baseAddress!, count: copyCount)
                }
            }
        }
        return (aligned, rate, srcWin, tgtWin)
    }

    // MARK: - MLXArray bridge

    /// Convert an MLXArray waveform `(samples,)` or `(channels, samples)` to a mono
    /// `[Float]` (averaging channels if multichannel).
    public static func mlxToMonoFloats(_ waveform: MLXArray) -> [Float] {
        MLX.eval(waveform)
        if waveform.ndim == 1 {
            return waveform.asArray(Float.self)
        }
        // Multi-channel: mean across channels axis 0
        let mono = waveform.mean(axis: 0)
        MLX.eval(mono)
        return mono.asArray(Float.self)
    }
}
