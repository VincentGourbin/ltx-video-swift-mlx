// FrameCountSpecTests.swift — asking for a length in frames or seconds
// Copyright 2026

import Foundation
import Testing
@testable import LTXVideo

@Suite("Frame grid rounding")
struct FrameGridRoundingTests {

    /// The two rules answer different questions and are a full grid step apart
    /// at every whole second. Asserted deliberately, because an earlier version
    /// of this file pinned both answers in separate tests without noticing they
    /// contradicted each other.
    @Test(arguments: [
        (5.0, 121, 113), (10.0, 241, 233), (15.0, 361, 353), (20.0, 481, 473),
    ])
    func requestAndPredictionDivergeByOneStep(
        seconds: Double, requested: Int, predicted: Int
    ) {
        // What `--frames 15s` gives: the closest count, so the clip really is
        // the duration that was named.
        let ask = FrameCountSpec.frames(forSeconds: seconds, rounding: .nearest)
        #expect(ask.frames == requested)
        #expect(FrameGrid.seconds(forFrames: ask.frames, fps: 24) == seconds)

        // What `--frames auto` gives: rounded down, never exceeding the
        // prediction, matching upstream's seconds_to_clamped_num_frames.
        let head = LTXDurationHead.snapToGrid(
            seconds: Float(seconds), frameRate: 24, minSeconds: 1, maxSeconds: 20)
        #expect(head == predicted)

        #expect(requested - predicted == FrameGrid.step)
    }

    /// Upstream converts with Python's `round()`, which is banker's rounding.
    /// 5.6875 × 24 is exactly 136.5; half-away-from-zero would give 137 frames
    /// where upstream gives 129.
    @Test func theHeadRoundsHalvesTheWayPythonDoes() {
        let frames = LTXDurationHead.snapToGrid(
            seconds: 5.6875, frameRate: 24, minSeconds: 1, maxSeconds: 20)
        #expect(frames == 129, "banker's rounding expected, got \(frames)")
    }

    @Test func aClipLastsFrameCountMinusOneOverFps() {
        // The one definition. Anything printing F/fps is wrong, and the CLI
        // banner used to.
        #expect(FrameGrid.seconds(forFrames: 121, fps: 24) == 5.0)
        #expect(FrameGrid.seconds(forFrames: 481, fps: 24) == 20.0)
        #expect(FrameGrid.seconds(forFrames: 9, fps: 24) == 1.0 / 3.0)
    }
}

@Suite("Frame count spec")
struct FrameCountSpecTests {

    @Test func autoIsRecognisedRegardlessOfSurroundingSpace() throws {
        // The CLI used to detect `auto` a second time with a rule that did not
        // trim, so `--frames " auto"` fell through to the count branch and died
        // with a message naming 'auto' as valid.
        for raw in ["auto", "AUTO", "  auto "] {
            #expect(try FrameCountSpec.parse(raw).spec == .auto)
        }
    }

    @Test func plainCountsPassThrough() throws {
        #expect(try FrameCountSpec.parse("121").spec == .frames(121))
        #expect(try FrameCountSpec.parse("9").spec == .frames(9))
        #expect(try FrameCountSpec.parse("481").spec == .frames(481))
    }

    @Test func offGridCountsAreRefusedWithBothNeighbours() throws {
        do {
            _ = try FrameCountSpec.parse("360")
            Issue.record("360 should not parse")
        } catch {
            let text = "\(error)"
            #expect(text.contains("353"))
            #expect(text.contains("361"))
        }
    }

    /// Counts outside 9...481 used to be accepted here and rejected only by
    /// `config.validate()` — after `loadModels()` had pulled tens of gigabytes.
    @Test(arguments: ["489", "1", "0", "-7", "-1"])
    func countsOutsideTheModelsRangeAreRefusedUpFront(raw: String) {
        #expect(throws: LTXError.self, "'\(raw)' should not parse") {
            try FrameCountSpec.parse(raw)
        }
    }

    @Test(arguments: [
        (15.0, 361), (10.0, 241), (5.0, 121), (1.0, 25), (20.0, 481),
    ])
    func wholeSecondsLandExactlyOnTheGrid(seconds: Double, expected: Int) throws {
        let (frames, exact) = FrameCountSpec.frames(forSeconds: seconds)
        #expect(frames == expected)
        #expect(exact, "\(seconds)s should need no adjustment")
        #expect(FrameGrid.seconds(forFrames: frames, fps: 24) == seconds)

        let parsed = try FrameCountSpec.parse("\(Int(seconds))s")
        #expect(parsed.spec == .frames(expected))
        #expect(parsed.note == nil, "exact durations must not emit a note")
    }

    @Test func offGridDurationsSnapAndSaySo() throws {
        let parsed = try FrameCountSpec.parse("2.5s")
        guard case .frames(let count) = parsed.spec else {
            Issue.record("expected a frame count"); return
        }
        #expect((count - 1) % FrameGrid.step == 0)
        #expect(parsed.note != nil, "an adjusted duration must be reported")
        #expect(parsed.note?.contains("\(count)") == true)
    }

    /// The clamp to the 9-frame minimum used to return `exact: true`, so
    /// `--frames 0.05s` silently produced a 0.33 s clip with no note — the exact
    /// silent substitution this type exists to prevent.
    @Test(arguments: [0.01, 0.02, 0.05, 0.1])
    func theMinimumClampIsReported(seconds: Double) throws {
        let (frames, exact) = FrameCountSpec.frames(forSeconds: seconds)
        #expect(frames == FrameGrid.minimum)
        #expect(!exact, "\(seconds)s was clamped to \(frames) frames and called exact")

        let parsed = try FrameCountSpec.parse("\(seconds)s")
        #expect(parsed.note != nil, "\(seconds)s changed length with no note")
    }

    /// Durations beyond the model's range are refused, not silently capped —
    /// and refused *here*, not after a multi-gigabyte load.
    @Test(arguments: ["21s", "30s", "120s", "3600s"])
    func durationsBeyondTheModelAreRefused(raw: String) {
        #expect(throws: LTXError.self, "'\(raw)' should not parse") {
            try FrameCountSpec.parse(raw)
        }
    }

    /// `Double("inf")`, `Double("1e400")` and `Double("0x10")` all parse, and the
    /// first two used to abort the process inside `Int(_:)` — reachable from
    /// `--frames auto` after the checkpoint had already loaded.
    @Test(arguments: [
        "banana", "", "15 s", "s", "-5s", "0s", "12.5.3s",
        "infs", "nans", "1e400s", "1e19s", "0x10s", "  s", ".5s", "5.s",
    ])
    func nonsenseIsRefusedWithoutTrapping(raw: String) {
        #expect(throws: LTXError.self, "'\(raw)' should not parse") {
            try FrameCountSpec.parse(raw)
        }
    }

    /// `Int.max` and `Int.min` used to reach unchecked arithmetic and trap.
    @Test func extremeCountsAreRefusedWithoutTrapping() {
        for raw in ["\(Int.max)", "\(Int.min)", "99999999999999999999"] {
            #expect(throws: LTXError.self, "'\(raw)' should not parse") {
                try FrameCountSpec.parse(raw)
            }
        }
    }

    @Test func aDifferentFrameRateNeedNotLandOnTheGrid() {
        // 15 s at 30 fps wants 451 frames, and 450 % 8 != 0 — the grid has no
        // point there, so it snaps and must report doing so. The grid is a
        // property of the VAE, not of the frame rate.
        let (frames, exact) = FrameCountSpec.frames(forSeconds: 15, fps: 30)
        #expect(!exact)
        #expect(frames == 449)
        #expect((frames - 1) % FrameGrid.step == 0)
    }

    @Test func durationsAreFormattedForPeopleNotForPrintf() {
        // "%g" renders 1200 as 1.2e+03, which in a sentence about seconds reads
        // as a typo.
        #expect(FrameCountSpec.format(1200) == "1200")
        #expect(FrameCountSpec.format(15) == "15")
        #expect(FrameCountSpec.format(2.5) == "2.50")
    }
}

@Suite("Prompt duration detection")
struct PromptDurationTests {

    @Test(arguments: [
        ("15 seconds, 16:9 landscape. A laundromat.", 15.0),
        ("15 secs. A laundromat.", 15.0),
        ("7 sec long, a laundromat", 7.0),
        ("2.5 seconds of footage", 2.5),
        ("1 minute of rain", 60.0),
        ("1.5 minutes of rain", 90.0),
        // The hyphenated adjective is at least as common as the plain form, and
        // matching one but not the other made the warning look arbitrary.
        ("a 15-second clip of a cat", 15.0),
        ("a 20-minute ceremony", 1200.0),
    ])
    func findsRequestedDurations(prompt: String, expected: Double) {
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

    /// A duration deep in the prose is describing the scene, not requesting a
    /// length. Warning about these would train people to ignore the warning.
    @Test(arguments: [
        "A wide shot of a stadium at dusk, the crowd hushed, as the sprinter runs the 100 metres in 9.58 seconds.",
        "A kitchen at night, warm light, a microwave counting down from 30 seconds on its cracked display.",
        "A basketball court, sneakers squeaking, 24 seconds on the shot clock as the crowd roars.",
    ])
    func ignoresDurationsThatDescribeTheScene(prompt: String) {
        #expect(PromptDuration.find(in: prompt) == nil, "matched in: \(prompt)")
    }

    @Test(arguments: [
        "Make it 15 seconds long, a laundromat at night",
        "About 15 seconds. A laundromat.",
    ])
    func acceptsRequestsPhrasedWithAShortLeadIn(prompt: String) {
        #expect(PromptDuration.find(in: prompt) == 15.0)
    }

    @Test func readsTheOpeningClauseWhereRequestsAreWritten() {
        // People who want a length put it first — "15 seconds, 16:9 landscape…".
        // The later mention is scene content, so the first is the right answer
        // and there is no ambiguity to resolve.
        #expect(PromptDuration.find(in: "15 seconds. A 3 second pause.") == 15.0)
    }
}
