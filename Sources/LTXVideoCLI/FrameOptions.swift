// FrameOptions.swift - Resolving --frames, and reporting what the model ignored
// Copyright 2026

import ArgumentParser
import Foundation
import LTXVideo

/// Resolve a `--frames` value that may be a count or a duration (`15s`).
///
/// `generate` handles this inline because it also accepts `auto`; the commands
/// that take a fixed length share this.
func resolveFrames(_ raw: String) throws -> Int {
    let parsed: (spec: FrameCountSpec, note: String?)
    do {
        parsed = try FrameCountSpec.parse(raw)
    } catch {
        throw ValidationError("\(error.localizedDescription)")
    }
    guard case .frames(let count) = parsed.spec else {
        throw ValidationError(
            "'auto' needs the duration head; this command takes a count or a duration like 15s.")
    }
    if let note = parsed.note { print("Note: \(note)") }
    return count
}

/// Report a duration written into the prompt that nothing acted on.
///
/// Nothing in LTX reads a duration out of a prompt: with an explicit
/// `--frames` the text is never consulted, and with `--frames auto` the duration
/// head regresses a length from connector tokens and never sees the characters.
/// Either way the request is dropped, and it used to be dropped in silence.
///
/// Must be called on **every** path, not just the auto one. The person this
/// exists for is the one who typed "15 seconds, 16:9 landscape…" and left
/// `--frames` at its default — they never pass `auto`, so gating the check on it
/// misses exactly the case it was written for.
///
/// - Parameters:
///   - prompt: the prompt as the user wrote it, not an enhanced rewrite — the
///     enhancer tends to drop the duration outright.
///   - resolvedFrames: the frame count the run will actually use.
///   - predictedSeconds: what the duration head returned *before* clamping, when
///     it ran. Reporting the post-clamp count instead would attribute the clamp
///     to the model: a head that asked for 23.5 s and got capped is not a head
///     that asked for 19.7 s.
func noteIgnoredPromptDuration(
    prompt: String,
    resolvedFrames: Int,
    predictedSeconds: Double? = nil,
    fps: Double = 24.0
) {
    guard let asked = PromptDuration.find(in: prompt) else { return }
    let got = FrameGrid.seconds(forFrames: resolvedFrames, fps: fps)
    // Half a second of slack: a prompt asking for 5 s that resolved to 121
    // frames (5.0 s) got what it wanted, whatever route it took.
    guard abs(asked - got) > 0.5 else { return }

    let want = FrameCountSpec.format(asked)
    let ceiling = FrameGrid.seconds(forFrames: FrameGrid.maximum, fps: fps)

    print()
    if let predicted = predictedSeconds {
        let raw = FrameCountSpec.format(predicted)
        let clamped = predicted > got + 0.5
        print("Note: the prompt asks for \(want)s, but --frames auto predicted \(raw)s"
              + (clamped ? " (capped to \(FrameCountSpec.format(got))s)." : "."))
        print("      The duration head reads the scene, not written durations.")
    } else {
        print("Note: the prompt asks for \(want)s, but this run is "
              + "\(FrameCountSpec.format(got))s.")
        print("      Nothing reads a duration out of the prompt text.")
    }

    // Only ever suggest something the CLI will accept. A prompt saying "a 2
    // minute short film" asks for more than the model can generate at all.
    guard asked <= ceiling else {
        print("      \(want)s is beyond this model's \(FrameCountSpec.format(ceiling))s maximum "
              + "(\(FrameGrid.maximum) frames); it cannot be generated in one run.")
        return
    }
    let (wanted, exact) = FrameCountSpec.frames(forSeconds: asked, fps: fps)
    if exact {
        print("      Pass --frames \(wanted) (or \(want)s) to get \(want)s.")
    } else {
        // Saying "to get 15s" when the grid cannot express 15s would repeat, in
        // the advisory itself, the silent substitution it exists to flag.
        let actual = FrameCountSpec.format(FrameGrid.seconds(forFrames: wanted, fps: fps))
        print("      Pass --frames \(wanted) (or \(want)s) — the closest the grid allows, \(actual)s.")
    }
}
