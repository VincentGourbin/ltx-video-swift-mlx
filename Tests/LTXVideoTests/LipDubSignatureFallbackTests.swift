//
//  LipDubSignatureFallbackTests.swift
//  ltx-video-swift-mlx
//
//  Tests the string repair logic that runs after VLM-driven prompt enhancement
//  in image-mode LipDub (--enhance-prompt). The LipDub IC-LoRA expects an
//  English speaking-verb wrapper around the dialogue; if the VLM dropped it,
//  we re-append the original tail. See `LTXPipeline.applyLipDubSignatureFallback`.
//

import Testing
@testable import LTXVideo

@Suite("LipDub --enhance-prompt signature fallback")
struct LipDubSignatureFallbackTests {

    private let canonicalOriginal =
        #"A bearded man, speaking in Spanish saying: "Hola a todos.""#

    // MARK: - Wrapper preserved (no fallback)

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
        // The actual VLM-output shape we observed in practice.
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

    // MARK: - Word-boundary check (regression: `speaking individually` must NOT match)

    @Test func speakingIndividuallyDoesNotMatchSpeakingIn() {
        // "speaking individually" / "speaks intermittently" used to register as
        // a positive substring hit and silently suppress the fallback.
        let enhanced = "A man speaking individually into a mic outdoors at sunset."
        let (final, reappended) = LTXPipeline.applyLipDubSignatureFallback(
            enhanced: enhanced, original: canonicalOriginal
        )
        // Fallback should fire because no whole-word `speaking in` survived.
        #expect(reappended == #"speaking in Spanish saying: "Hola a todos.""#)
        #expect(final.contains(#"speaking in Spanish saying:"#))
    }

    @Test func speaksIntermittentlyDoesNotMatchSpeaksIn() {
        let enhanced = "A man who says intermittently strange things in a forest."
        let (final, reappended) = LTXPipeline.applyLipDubSignatureFallback(
            enhanced: enhanced, original: canonicalOriginal
        )
        #expect(reappended == #"speaking in Spanish saying: "Hola a todos.""#)
        #expect(final.contains(#"speaking in Spanish saying:"#))
    }

    // MARK: - Fallback re-appends signature

    @Test func reappendsSignatureWhenAllHintsDropped() {
        let enhanced = "A bearded man holds a microphone outdoors at golden hour."
        let (final, reappended) = LTXPipeline.applyLipDubSignatureFallback(
            enhanced: enhanced, original: canonicalOriginal
        )
        #expect(reappended == #"speaking in Spanish saying: "Hola a todos.""#)
        // Period is terminal → space joiner.
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

    @Test func usesSpaceJoinerAfterClosingDoubleQuote() {
        // VLM ends with a quoted aside but no speaking-verb wrapper survives.
        // Closing `"` is treated as terminal — joiner must be a space, NOT `, `.
        let enhanced = #"A bearded man whispers, "Hola.""#
        let (final, reappended) = LTXPipeline.applyLipDubSignatureFallback(
            enhanced: enhanced, original: canonicalOriginal
        )
        #expect(reappended != nil)
        #expect(final ==
            #"A bearded man whispers, "Hola." speaking in Spanish saying: "Hola a todos.""#)
    }

    @Test func usesSpaceJoinerAfterClosingParen() {
        let enhanced = "A bearded man (mic in hand)"
        let (final, _) = LTXPipeline.applyLipDubSignatureFallback(
            enhanced: enhanced, original: canonicalOriginal
        )
        #expect(final ==
            #"A bearded man (mic in hand) speaking in Spanish saying: "Hola a todos.""#)
    }

    @Test func usesSpaceJoinerAfterQuestionMark() {
        let enhanced = "Is this the one?"
        let (final, _) = LTXPipeline.applyLipDubSignatureFallback(
            enhanced: enhanced, original: canonicalOriginal
        )
        #expect(final ==
            #"Is this the one? speaking in Spanish saying: "Hola a todos.""#)
    }

    // MARK: - Edge cases

    @Test func returnsEnhancedWhenOriginalLacksSignature() {
        // Pathological input: user wrote no LipDub signature at all. We have
        // nothing to fall back to, so we return whatever the VLM produced.
        let enhanced = "A static portrait shot."
        let original = "Some scene"  // no "speaking in"
        let (final, reappended) = LTXPipeline.applyLipDubSignatureFallback(
            enhanced: enhanced, original: original
        )
        #expect(final == enhanced)
        #expect(reappended == nil)
    }

    @Test func detectsSignatureCaseInsensitively() {
        let original = #"SPEAKING IN French saying: "Bonjour.""#
        let enhanced = "An outdoor scene with no speaking-verb hint at all."
        let (final, reappended) = LTXPipeline.applyLipDubSignatureFallback(
            enhanced: enhanced, original: original
        )
        // The signature is taken from the original (verbatim casing).
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
        #expect(final ==
            #"A scene description with trailing whitespace. speaking in Spanish saying: "Hola a todos.""#)
    }

    @Test func emptyEnhancedFallsBackToBareSignature() {
        let (final, reappended) = LTXPipeline.applyLipDubSignatureFallback(
            enhanced: "", original: canonicalOriginal
        )
        #expect(reappended == #"speaking in Spanish saying: "Hola a todos.""#)
        #expect(final == #"speaking in Spanish saying: "Hola a todos.""#)
    }

    // MARK: - Unicode safety (the lowercased-index hazard)

    @Test func handlesNonAsciiOriginalSafely() {
        // Turkish 'İ' lowercases to 'i\u{0307}' (2 scalars). With the old
        // implementation, using a String.Index from `original.lowercased()` to
        // slice `original` could trap or misalign by one scalar. The rewrite
        // searches the original directly with .caseInsensitive, so this works.
        let original = #"İstanbul shot, SPEAKING IN Turkish saying: "Merhaba""#
        let enhanced = "An outdoor scene with no wrapper present at all."
        let (final, reappended) = LTXPipeline.applyLipDubSignatureFallback(
            enhanced: enhanced, original: original
        )
        #expect(reappended == #"SPEAKING IN Turkish saying: "Merhaba""#)
        #expect(final.hasSuffix(#"SPEAKING IN Turkish saying: "Merhaba""#))
    }

    @Test func handlesSharpSLowercaseSafely() {
        // German 'ẞ' lowercases to 'ss' (1 → 2 chars). Same hazard class.
        let original = #"WEIẞ light scene, speaking in German saying: "Guten Tag""#
        let enhanced = "Plain enhanced output."
        let (final, reappended) = LTXPipeline.applyLipDubSignatureFallback(
            enhanced: enhanced, original: original
        )
        #expect(reappended == #"speaking in German saying: "Guten Tag""#)
        #expect(final.hasSuffix(#"speaking in German saying: "Guten Tag""#))
    }
}
