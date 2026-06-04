//
//  LipDubSignatureFallbackTests.swift
//  ltx-video-swift-mlx
//
//  Tests the string repair logic that runs after VLM-driven prompt enhancement
//  in image-mode LipDub (--enhance-prompt). The LipDub IC-LoRA expects a
//  speaking-verb hint near the dialogue; if the VLM dropped it, we re-append
//  the original tail. See `LTXPipeline.applyLipDubSignatureFallback`.
//

import Testing
@testable import LTXVideo

@Suite("LipDub --enhance-prompt signature fallback")
struct LipDubSignatureFallbackTests {

    private let canonicalOriginal =
        #"A bearded man, speaking in Spanish saying: "Hola a todos.""#

    @Test func keepsEnhancedWhenSpeakingInPresent() {
        let enhanced =
            #"Style: documentary — The man is speaking in Spanish, "Hola a todos.""#
        let (final, reappended) = LTXPipeline.applyLipDubSignatureFallback(
            enhanced: enhanced, original: canonicalOriginal
        )
        #expect(final == enhanced)
        #expect(reappended == nil)
    }

    @Test func keepsEnhancedWhenSpeaksInPresent() {
        // This is the actual VLM-output shape we observed in practice.
        let enhanced =
            #"Style: documentary - The man holds a microphone and speaks in Spanish, "Hola a todos.""#
        let (final, reappended) = LTXPipeline.applyLipDubSignatureFallback(
            enhanced: enhanced, original: canonicalOriginal
        )
        #expect(final == enhanced)
        #expect(reappended == nil)
    }

    @Test func keepsEnhancedWhenSayingInPresent() {
        let enhanced = #"A woman, saying in English, "Hello.""#
        let (final, reappended) = LTXPipeline.applyLipDubSignatureFallback(
            enhanced: enhanced, original: canonicalOriginal
        )
        #expect(final == enhanced)
        #expect(reappended == nil)
    }

    @Test func reappendsSignatureWhenAllHintsDropped() {
        let enhanced = "A bearded man holds a microphone outdoors at golden hour."
        let (final, reappended) = LTXPipeline.applyLipDubSignatureFallback(
            enhanced: enhanced, original: canonicalOriginal
        )
        #expect(reappended == #"speaking in Spanish saying: "Hola a todos.""#)
        // Joiner is a space because the enhanced text ends with "."
        #expect(final ==
            #"A bearded man holds a microphone outdoors at golden hour. speaking in Spanish saying: "Hola a todos.""#)
    }

    @Test func reappendsWithCommaJoinerWhenNoTerminalPunctuation() {
        let enhanced = "A bearded man holding a microphone"
        let (final, reappended) = LTXPipeline.applyLipDubSignatureFallback(
            enhanced: enhanced, original: canonicalOriginal
        )
        #expect(reappended == #"speaking in Spanish saying: "Hola a todos.""#)
        #expect(final ==
            #"A bearded man holding a microphone, speaking in Spanish saying: "Hola a todos.""#)
    }

    @Test func returnsEnhancedWhenOriginalLacksSignature() {
        // Pathological input: user wrote no LipDub signature at all. We have nothing
        // to fall back to, so we return whatever the VLM produced.
        let enhanced = "A static portrait shot."
        let original = "Some scene"  // no "speaking in"
        let (final, reappended) = LTXPipeline.applyLipDubSignatureFallback(
            enhanced: enhanced, original: original
        )
        #expect(final == enhanced)
        #expect(reappended == nil)
    }

    @Test func detectsSignatureCaseInsensitively() {
        // User-written prompt starting with capital "Speaking" must still match.
        let original = #"SPEAKING IN French saying: "Bonjour.""#
        let enhanced = "An outdoor scene with no speaking-verb hint at all."
        let (final, reappended) = LTXPipeline.applyLipDubSignatureFallback(
            enhanced: enhanced, original: original
        )
        // The signature is extracted from the original at the lowercased match offset,
        // so the casing of the appended text mirrors the original.
        #expect(reappended == #"SPEAKING IN French saying: "Bonjour.""#)
        #expect(final ==
            #"An outdoor scene with no speaking-verb hint at all. SPEAKING IN French saying: "Bonjour.""#)
    }

    @Test func trimsWhitespaceBeforeJoining() {
        let enhanced = "A scene description with trailing whitespace.   \n"
        let (final, reappended) = LTXPipeline.applyLipDubSignatureFallback(
            enhanced: enhanced, original: canonicalOriginal
        )
        #expect(reappended != nil)
        // Trailing whitespace stripped before the joiner is applied.
        #expect(final ==
            #"A scene description with trailing whitespace. speaking in Spanish saying: "Hola a todos.""#)
    }

    @Test func emptyEnhancedFallsBackToBareSignature() {
        let (final, reappended) = LTXPipeline.applyLipDubSignatureFallback(
            enhanced: "", original: canonicalOriginal
        )
        // No leading content → no joiner — just the signature.
        #expect(reappended == #"speaking in Spanish saying: "Hola a todos.""#)
        #expect(final == #"speaking in Spanish saying: "Hola a todos.""#)
    }
}
