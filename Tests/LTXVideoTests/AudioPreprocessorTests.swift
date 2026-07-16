// AudioPreprocessorTests.swift — Unit tests for silence-aware audio alignment
// Copyright 2026

import Foundation
import Testing
@testable import LTXVideo

@Suite("AudioPreprocessor")
struct AudioPreprocessorTests {

    // MARK: - Silence detection

    @Test("detectSpeechWindow returns full range for a constant-amplitude tone")
    func testDetectSpeechWindowFullRange() {
        let sr = 16000
        // 1 second of sine at 0.3 amplitude → well above -35 dBFS threshold (~0.0178)
        let count = sr
        var wave = [Float](repeating: 0, count: count)
        for i in 0..<count {
            wave[i] = 0.3 * sin(2 * .pi * 440 * Float(i) / Float(sr))
        }
        let (start, end) = AudioPreprocessor.detectSpeechWindow(waveform: wave, sampleRate: sr)
        #expect(start == 0)
        // End should be close to count (within one frame of 10ms = 160 samples)
        #expect(end >= count - 160 && end <= count)
    }

    @Test("detectSpeechWindow trims leading and trailing silence")
    func testDetectSpeechWindowTrimsSilence() {
        let sr = 16000
        let leadSamples = sr / 4   // 250 ms of silence
        let speechSamples = sr / 2 // 500 ms of speech
        let trailSamples = sr / 4  // 250 ms of silence
        var wave = [Float](repeating: 0, count: leadSamples + speechSamples + trailSamples)
        for i in 0..<speechSamples {
            wave[leadSamples + i] = 0.3 * sin(2 * .pi * 440 * Float(i) / Float(sr))
        }
        let (start, end) = AudioPreprocessor.detectSpeechWindow(waveform: wave, sampleRate: sr)
        // Speech detected within one frame (160 samples = 10ms) of true boundaries
        #expect(abs(start - leadSamples) <= 160)
        #expect(abs(end - (leadSamples + speechSamples)) <= 160)
    }

    @Test("detectSpeechWindow returns (0, count) for full silence")
    func testDetectSpeechWindowSilent() {
        let sr = 16000
        let wave = [Float](repeating: 0, count: sr)
        let (start, end) = AudioPreprocessor.detectSpeechWindow(waveform: wave, sampleRate: sr)
        #expect(start == 0)
        #expect(end == wave.count)
    }

    // Regression for the enrolled-voice case (Fluxforge asks, B2): the voice's
    // "silences" carry a learned noise floor of -32.5 dB — ABOVE the -35 dB
    // absolute threshold, so every frame counted as speech and no window was
    // found. The floor-relative term (noise floor + 10 dB) must recover the window.
    @Test("detectSpeechWindow finds the window despite a -32.5 dB noise floor")
    func testDetectSpeechWindowEnrolledVoiceNoiseFloor() {
        let sr = 16000
        let leadSamples = sr / 4
        let speechSamples = sr / 2
        let trailSamples = sr / 4
        let noiseFloor: Float = 0.0237  // ≈ -32.5 dBFS RMS (measured on enrolled voice)
        var wave = [Float](repeating: 0, count: leadSamples + speechSamples + trailSamples)
        for i in 0..<wave.count {
            // Deterministic pseudo-noise at the measured floor
            wave[i] = noiseFloor * 1.414 * sin(2 * .pi * 1731 * Float(i) / Float(sr))
        }
        for i in 0..<speechSamples {
            wave[leadSamples + i] = 0.3 * sin(2 * .pi * 440 * Float(i) / Float(sr))
        }
        let (start, end) = AudioPreprocessor.detectSpeechWindow(waveform: wave, sampleRate: sr)
        #expect(abs(start - leadSamples) <= 160)
        #expect(abs(end - (leadSamples + speechSamples)) <= 160)
    }

    // The absolute floor must keep quiet noise-only clips out of "speech": their
    // peak − 25 dB lands below -35 dBFS, so no window is promoted (old behavior).
    @Test("detectSpeechWindow still returns (0, count) for quiet noise-only audio")
    func testDetectSpeechWindowQuietNoiseOnly() {
        let sr = 16000
        var wave = [Float](repeating: 0, count: sr)
        for i in 0..<wave.count {
            wave[i] = 0.005 * sin(2 * .pi * 997 * Float(i) / Float(sr))  // ≈ -49 dBFS RMS
        }
        let (start, end) = AudioPreprocessor.detectSpeechWindow(waveform: wave, sampleRate: sr)
        #expect(start == 0)
        #expect(end == wave.count)
    }

    // MARK: - Time stretching

    @Test("timeStretch with rate=1.0 returns input unchanged")
    func testTimeStretchIdentity() throws {
        let sr: Double = 16000
        var wave = [Float](repeating: 0, count: 8000)
        for i in 0..<wave.count {
            wave[i] = 0.1 * sin(2 * .pi * 440 * Float(i) / Float(sr))
        }
        let out = try AudioPreprocessor.timeStretch(waveform: wave, sampleRate: sr, rate: 1.0)
        #expect(out.count == wave.count)
    }

    @Test("timeStretch with rate=1.5 produces ~2/3 the samples (faster)")
    func testTimeStretchFaster() throws {
        let sr: Double = 16000
        var wave = [Float](repeating: 0, count: 16000)  // 1 second
        for i in 0..<wave.count {
            wave[i] = 0.2 * sin(2 * .pi * 440 * Float(i) / Float(sr))
        }
        let out = try AudioPreprocessor.timeStretch(waveform: wave, sampleRate: sr, rate: 1.5)
        let expected = Int(Double(wave.count) / 1.5)
        // Allow ±5% tolerance for AVAudioUnitTimePitch latency
        let tolerance = Int(Double(expected) * 0.05)
        #expect(abs(out.count - expected) <= tolerance, "got \(out.count), expected ~\(expected)")
    }

    @Test("timeStretch with rate=0.75 produces ~4/3 the samples (slower)")
    func testTimeStretchSlower() throws {
        let sr: Double = 16000
        var wave = [Float](repeating: 0, count: 16000)
        for i in 0..<wave.count {
            wave[i] = 0.2 * sin(2 * .pi * 440 * Float(i) / Float(sr))
        }
        let out = try AudioPreprocessor.timeStretch(waveform: wave, sampleRate: sr, rate: 0.75)
        let expected = Int(Double(wave.count) / 0.75)
        let tolerance = Int(Double(expected) * 0.05)
        #expect(abs(out.count - expected) <= tolerance, "got \(out.count), expected ~\(expected)")
    }

    // MARK: - End-to-end alignment

    @Test("timeStretch throws when rate exceeds supported range")
    func testTimeStretchOutOfRange() throws {
        let wave: [Float] = Array(repeating: 0.1, count: 1000)
        #expect(throws: LTXError.self) {
            _ = try AudioPreprocessor.timeStretch(waveform: wave, sampleRate: 16000, rate: 5.0)
        }
        #expect(throws: LTXError.self) {
            _ = try AudioPreprocessor.timeStretch(waveform: wave, sampleRate: 16000, rate: 0.1)
        }
    }

    @Test("alignTargetToSource throws when computed rate is extreme")
    func testAlignTargetToSourceExtremeRate() throws {
        let sr = 16000
        // Source: 10 ms of speech (tiny window) in 1 second of total audio
        var source = [Float](repeating: 0, count: sr)
        for i in 0..<160 {  // 10 ms
            source[i] = 0.3 * sin(2 * .pi * 440 * Float(i) / Float(sr))
        }
        // Target: 1 s of speech → rate would be ~100 (way over 4.0)
        var target = [Float](repeating: 0, count: sr)
        for i in 0..<sr {
            target[i] = 0.3 * sin(2 * .pi * 660 * Float(i) / Float(sr))
        }
        #expect(throws: LTXError.self) {
            _ = try AudioPreprocessor.alignTargetToSource(source: source, target: target, sampleRate: sr)
        }
    }

    @Test("alignTargetToSource preserves total length and silence layout")
    func testAlignTargetToSource() throws {
        let sr = 16000
        // Source: 0.25s silence + 0.5s speech + 0.25s silence = 1.0s total
        var source = [Float](repeating: 0, count: sr)
        for i in 0..<sr/2 {
            source[sr/4 + i] = 0.3 * sin(2 * .pi * 440 * Float(i) / Float(sr))
        }
        // Target: 0.8s speech (no leading silence) — needs to be stretched (faster) to fit 0.5s
        let targetSpeechSamples = Int(0.8 * Double(sr))
        var target = [Float](repeating: 0, count: targetSpeechSamples)
        for i in 0..<target.count {
            target[i] = 0.3 * sin(2 * .pi * 660 * Float(i) / Float(sr))
        }
        let result = try AudioPreprocessor.alignTargetToSource(
            source: source, target: target, sampleRate: sr
        )
        #expect(result.waveform.count == source.count)
        // Rate should be ~1.6 (0.8 / 0.5)
        #expect(abs(result.rate - 1.6) < 0.1, "got rate=\(result.rate)")
        // Speech in aligned waveform should sit within source's detected window
        let (srcStart, srcEnd) = result.sourceWindow
        #expect(abs(srcStart - sr/4) <= 160)
        #expect(abs(srcEnd - 3*sr/4) <= 160)
    }
}
