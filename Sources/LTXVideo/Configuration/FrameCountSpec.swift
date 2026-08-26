// FrameCountSpec.swift - Asking for a clip length in frames, seconds, or neither
// Copyright 2026

import Foundation

/// How a caller expressed the clip length.
///
/// Frame counts must land on the VAE's `8k + 1` grid, so "15 seconds" is 361
/// frames and not 360 — arithmetic nobody should have to do in their head. That
/// friction is why people write the duration into the prompt instead, where
/// ``PromptDuration`` shows it is silently ignored.
public enum FrameCountSpec: Sendable, Equatable {
    /// Let the LTX-2.5 duration head pick, from the prompt's content.
    case auto
    /// An exact frame count, already validated against the grid.
    case frames(Int)

    /// Parse `auto`, a plain frame count, or a duration like `15s` / `2.5s`.
    ///
    /// - Parameter fps: frame rate used to turn seconds into frames.
    /// - Returns: the spec, plus a note when a duration had to be adjusted onto
    ///   the grid. The note is for the caller to show; adjusting silently is how
    ///   you get someone counting frames in a bug report.
    public static func parse(
        _ raw: String, fps: Double = 24.0
    ) throws -> (spec: FrameCountSpec, note: String?) {
        let text = raw.trimmingCharacters(in: .whitespaces).lowercased()

        if text == "auto" { return (.auto, nil) }

        if let count = Int(text) {
            guard (count - 1) % 8 == 0 else {
                throw LTXError.invalidConfiguration(
                    "Frame count must be 8n+1 (9, 17, …, 481). Got \(count). "
                    + "\(Self.suggestion(near: count)) Or pass a duration, e.g. "
                    + "\(String(format: "%.1f", Double(count - 1) / fps))s.")
            }
            return (.frames(count), nil)
        }

        if text.hasSuffix("s"), let seconds = Double(text.dropLast()), seconds > 0 {
            let (frames, exact) = Self.frames(forSeconds: seconds, fps: fps)
            let note = exact ? nil : String(
                format: "%.3gs is not on the 8k+1 grid at %.6g fps — using %d frames (%.3gs).",
                seconds, fps, frames, Double(frames - 1) / fps)
            return (.frames(frames), note)
        }

        throw LTXError.invalidConfiguration(
            "Frames must be a count (121), a duration (5s), or 'auto'. Got '\(raw)'.")
    }

    /// Frames for a duration, snapped to the nearest `8k + 1`.
    ///
    /// A clip of `F` frames lasts `(F - 1) / fps`, hence the `+ 1` before
    /// snapping: at 24 fps, 15 s is 361 and not 360.
    public static func frames(forSeconds seconds: Double, fps: Double = 24.0) -> (
        frames: Int, exact: Bool
    ) {
        let ideal = Int((seconds * fps).rounded()) + 1
        if (ideal - 1) % 8 == 0 { return (max(ideal, 9), true) }

        let below = ((ideal - 1) / 8) * 8 + 1
        let above = below + 8
        let nearest = (ideal - below) <= (above - ideal) ? below : above
        return (max(nearest, 9), false)
    }

    /// The two grid points either side of an off-grid count, for error messages.
    private static func suggestion(near count: Int) -> String {
        let below = max(9, ((count - 1) / 8) * 8 + 1)
        let above = below + 8
        return "Nearest valid: \(below) or \(above)."
    }
}

/// Finds a duration a person wrote into a prompt.
///
/// The LTX-2.5 duration head is a regression over connector tokens — it never
/// sees the text, so a written duration is not an instruction and is dropped
/// without a word. See
/// `docs/knowledge/pitfalls/duration-head-does-not-read-written-durations.md`.
///
/// This exists so the drop can at least be *reported*. It deliberately does not
/// override anything: `auto` means "let the model choose the natural length",
/// and quietly substituting a parsed number would change what the mode means.
public enum PromptDuration {

    /// A duration mentioned in `prompt`, in seconds, or nil.
    ///
    /// Matches `15 sec`, `15 seconds`, `1.5 minutes`. Requires a numeral:
    /// "a few seconds" carries no target to compare against. Returns the first
    /// match — prompts that name several durations are describing the scene, not
    /// requesting a length.
    ///
    /// A bare `s` suffix is deliberately **not** accepted. `1950s`, `80s` and
    /// `90s` are decades, and they turn up in creative prompts far more often
    /// than someone writing `15s` in prose would. `--frames 15s` covers the
    /// abbreviated form where it actually gets typed: the command line.
    public static func find(in prompt: String) -> Double? {
        let pattern = #"(?<![\w.])(\d+(?:\.\d+)?)\s*(seconds?|secs?|minutes?|mins?)(?![\w])"#
        guard let regex = try? NSRegularExpression(pattern: pattern, options: .caseInsensitive)
        else { return nil }

        let range = NSRange(prompt.startIndex..., in: prompt)
        guard let match = regex.firstMatch(in: prompt, range: range),
              let valueRange = Range(match.range(at: 1), in: prompt),
              let unitRange = Range(match.range(at: 2), in: prompt),
              let value = Double(prompt[valueRange])
        else { return nil }

        let unit = prompt[unitRange].lowercased()
        return unit.hasPrefix("m") ? value * 60 : value
    }
}
