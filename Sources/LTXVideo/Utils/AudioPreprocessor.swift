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
    /// Uses fixed-size frame RMS against the *stricter* of two thresholds:
    /// - `thresholdDB`, an absolute dBFS floor, and
    /// - the waveform's own noise floor (10th-percentile frame RMS) plus
    ///   `noiseFloorMarginDB`.
    ///
    /// The floor-relative term matters for synthesized/enrolled voices whose
    /// "silences" carry a learned noise floor above the absolute threshold
    /// (measured case: -32.5 dB floor vs the -35 dB default — every frame counted
    /// as speech and no window was found). Anchoring to the signal's own floor
    /// keeps detection working regardless of the recording's noise level, while
    /// the absolute floor prevents a quiet noise-only clip from being promoted to
    /// "speech" (its floor plus the margin stays below `thresholdDB`, preserving
    /// the old behavior).
    ///
    /// Frames below threshold at the head/tail are treated as silence;
    /// mid-utterance pauses are NOT trimmed (only leading/trailing silence is
    /// removed). Returns `(0, count)` if no frame exceeds threshold — including
    /// when the clip has no dynamic range (all speech or all noise), where "the
    /// whole clip" is the only defensible window.
    public static func detectSpeechWindow(
        waveform: [Float],
        sampleRate: Int,
        thresholdDB: Float = -35,
        noiseFloorMarginDB: Float = 10,
        frameMS: Int = 10
    ) -> (start: Int, end: Int) {
        let frameSamples = sampleRate * frameMS / 1000
        guard frameSamples > 0, waveform.count >= frameSamples else {
            return (0, waveform.count)
        }

        let numFrames = waveform.count / frameSamples
        var frameRMS = [Float](repeating: 0, count: numFrames)
        for f in 0..<numFrames {
            var sumSq: Float = 0
            let base = f * frameSamples
            for i in 0..<frameSamples {
                let s = waveform[base + i]
                sumSq += s * s
            }
            frameRMS[f] = sqrt(sumSq / Float(frameSamples))
        }

        // Noise floor = 10th-percentile frame RMS: silence frames cluster there,
        // speech frames sit well above. (On an all-speech clip this is speech level
        // and the floor-relative threshold exceeds every frame — the absolute
        // threshold then decides, and the full-clip fallback covers the rest.)
        let sortedRMS = frameRMS.sorted()
        let noiseFloor = sortedRMS[numFrames / 10]

        let absoluteThreshold = pow(10.0, thresholdDB / 20.0)
        let floorRelativeThreshold = noiseFloor * pow(10.0, noiseFloorMarginDB / 20.0)
        let threshold = max(absoluteThreshold, floorRelativeThreshold)

        var firstActive = -1
        var lastActive = -1
        for f in 0..<numFrames where frameRMS[f] > threshold {
            if firstActive < 0 { firstActive = f }
            lastActive = f
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
    ///           `<1` lengthens it. Enforced range: `[0.25, 4.0]` (beyond this,
    ///           `AVAudioUnitTimePitch` quality degrades badly and the speech
    ///           becomes unintelligible — the function throws instead).
    /// - Returns: Time-stretched mono float32 samples.
    public static func timeStretch(
        waveform: [Float],
        sampleRate: Double,
        rate: Float
    ) throws -> [Float] {
        // No-op for negligible ratios
        if abs(rate - 1.0) < 0.005 { return waveform }
        guard waveform.count > 0 else { return waveform }
        guard rate >= 0.25 && rate <= 4.0 else {
            throw LTXError.invalidConfiguration(
                "Time-stretch rate \(rate) is outside the supported range [0.25, 4.0]. " +
                "AVAudioUnitTimePitch produces unintelligible output beyond this range."
            )
        }

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
    ///   - thresholdDB: Absolute RMS floor for silence detection (dBFS).
    ///   - noiseFloorMarginDB: Margin above each waveform's own estimated noise
    ///     floor; the stricter of the two thresholds wins (see ``detectSpeechWindow``).
    /// - Returns: `(alignedWaveform, ratio, sourceWindow, targetWindow)` where ratio
    ///   is the stretch factor that was applied (1.0 means no stretch).
    public static func alignTargetToSource(
        source: [Float],
        target: [Float],
        sampleRate: Int,
        thresholdDB: Float = -35,
        noiseFloorMarginDB: Float = 10
    ) throws -> (waveform: [Float], rate: Float, sourceWindow: (Int, Int), targetWindow: (Int, Int)) {
        let srcWin = detectSpeechWindow(waveform: source, sampleRate: sampleRate,
                                        thresholdDB: thresholdDB, noiseFloorMarginDB: noiseFloorMarginDB)
        let tgtWin = detectSpeechWindow(waveform: target, sampleRate: sampleRate,
                                        thresholdDB: thresholdDB, noiseFloorMarginDB: noiseFloorMarginDB)

        let srcSpeechSamples = srcWin.end - srcWin.start
        let tgtSpeechSamples = tgtWin.end - tgtWin.start
        guard srcSpeechSamples > 0, tgtSpeechSamples > 0 else {
            throw LTXError.invalidConfiguration(
                "Speech window detection failed (source=\(srcSpeechSamples) target=\(tgtSpeechSamples) samples). " +
                "Lower the silence threshold or supply audio with detectable speech."
            )
        }

        let rate = Float(tgtSpeechSamples) / Float(srcSpeechSamples)
        guard rate >= 0.25 && rate <= 4.0 else {
            let srcS = Float(srcSpeechSamples) / Float(sampleRate)
            let tgtS = Float(tgtSpeechSamples) / Float(sampleRate)
            throw LTXError.invalidConfiguration(String(format:
                "Computed stretch rate %.2f (target speech %.2fs / source speech %.2fs) " +
                "is outside the supported range [0.25, 4.0]. The target audio is too %@ " +
                "relative to the source's speech window. Use a target audio whose speech " +
                "duration is within 4x of the source's, or adjust the source video length.",
                rate, tgtS, srcS, rate > 4.0 ? "long" : "short"
            ))
        }
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
