// FrameCountSpecTests.swift — asking for a length in frames or seconds
// Copyright 2026

import Foundation
import Testing
@testable import LTXVideo

@Suite("Frame count spec")
struct FrameCountSpecTests {

    @Test func autoIsRecognised() throws {
        #expect(try FrameCountSpec.parse("auto").spec == .auto)
        #expect(try FrameCountSpec.parse("AUTO").spec == .auto)
        #expect(try FrameCountSpec.parse("  auto ").spec == .auto)
    }

    @Test func plainCountsPassThrough() throws {
        #expect(try FrameCountSpec.parse("121").spec == .frames(121))
        #expect(try FrameCountSpec.parse("9").spec == .frames(9))
        #expect(try FrameCountSpec.parse("481").spec == .frames(481))
    }

    @Test func offGridCountsAreRefusedWithBothNeighbours() {
        // The old message just said "must be 8n+1", leaving the reader to do the
        // arithmetic that caused the mistake.
        #expect(throws: LTXError.self) { try FrameCountSpec.parse("360") }
        do {
            _ = try FrameCountSpec.parse("360")
        } catch {
            let text = "\(error)"
            #expect(text.contains("353"))
            #expect(text.contains("361"))
        }
    }

    /// The reason this type exists: `(F - 1) / fps` is the duration, so 15 s is
    /// 361 frames and not 360. Nobody should have to derive that.
    @Test(arguments: [
        (15.0, 361), (10.0, 241), (5.0, 121), (1.0, 25), (20.0, 481),
    ])
    func wholeSecondsLandExactlyOnTheGrid(seconds: Double, expected: Int) throws {
        let (frames, exact) = FrameCountSpec.frames(forSeconds: seconds)
        #expect(frames == expected)
        #expect(exact, "\(seconds)s should need no adjustment")
        #expect((frames - 1) % 8 == 0)
        #expect(Double(frames - 1) / 24.0 == seconds)

        let parsed = try FrameCountSpec.parse("\(Int(seconds))s")
        #expect(parsed.spec == .frames(expected))
        #expect(parsed.note == nil, "exact durations must not emit a note")
    }

    @Test func offGridDurationsSnapAndSaySo() throws {
        // 2.5 s -> 61 frames, which is off the grid; 57 and 65 are equidistant,
        // and the note has to appear or the caller silently gets another length.
        let parsed = try FrameCountSpec.parse("2.5s")
        guard case .frames(let count) = parsed.spec else {
            Issue.record("expected a frame count"); return
        }
        #expect((count - 1) % 8 == 0)
        #expect(parsed.note != nil, "an adjusted duration must be reported")
        #expect(parsed.note?.contains("\(count)") == true)
    }

    @Test func durationsNeverUndershootTheMinimumClip() {
        // A 9-frame clip is the shortest the config accepts; anything below has
        // to round up to it rather than produce an invalid request.
        for seconds in [0.01, 0.1, 0.3] {
            let (frames, _) = FrameCountSpec.frames(forSeconds: seconds)
            #expect(frames >= 9, "\(seconds)s produced \(frames)")
            #expect((frames - 1) % 8 == 0)
        }
    }

    @Test func nonsenseIsRefused() {
        for raw in ["banana", "", "15 s", "s", "-5s", "0s", "12.5.3s"] {
            #expect(throws: LTXError.self, "'\(raw)' should not parse") {
                try FrameCountSpec.parse(raw)
            }
        }
    }

    @Test func aDifferentFrameRateNeedNotLandOnTheGrid() {
        // 15 s at 30 fps wants 451 frames, and 450 % 8 != 0 — the grid simply has
        // no point there, so it snaps to 449 (14.93 s) and must report doing so.
        // The grid is a property of the VAE, not of the frame rate.
        let (frames, exact) = FrameCountSpec.frames(forSeconds: 15, fps: 30)
        #expect(!exact)
        #expect(frames == 449)
        #expect((frames - 1) % 8 == 0)
    }
}

@Suite("Prompt duration detection")
struct PromptDurationTests {

    @Test(arguments: [
        ("15 seconds, 16:9 landscape. A laundromat.", 15.0),
        ("A laundromat. 15 secs.", 15.0),
        ("Make it 7 sec long", 7.0),
        ("about 2.5 seconds of footage", 2.5),
        ("1 minute of rain", 60.0),
        ("1.5 minutes of rain", 90.0),
    ])
    func findsWrittenDurations(prompt: String, expected: Double) {
        #expect(PromptDuration.find(in: prompt) == expected)
    }

    @Test(arguments: [
        "A quiet laundromat with flickering lights.",
        "a few seconds of silence",          // no numeral, nothing to compare to
        "shot at 24 fps",
        "768x512 landscape",
        "16:9 aspect",
        "a 1950s diner",                     // decade, not a duration
        "an 80s music video",                // ditto
        "A laundromat. 15s.",                // bare 's' is ambiguous with decades
    ])
    func ignoresEverythingElse(prompt: String) {
        #expect(PromptDuration.find(in: prompt) == nil, "matched in: \(prompt)")
    }

    @Test func takesTheFirstOfSeveral() {
        // Several durations describe the scene rather than request a length;
        // the first is the one a person writing a header would have put there.
        #expect(PromptDuration.find(in: "15 seconds. A 3 second pause.") == 15.0)
    }
}
