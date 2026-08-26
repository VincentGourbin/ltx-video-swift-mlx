// FrameCountSpec.swift - Asking for a clip length in frames, seconds, or neither
// Copyright 2026

import Foundation

/// The VAE's causal temporal grid: valid frame counts are `8k + 1`.
///
/// One place for this arithmetic, because there are two callers that round
/// **differently on purpose** and the difference must be legible rather than
/// discovered. See ``GridRounding``.
public enum FrameGrid {
    /// Spacing of the grid. A frame count is valid iff `(F - 1) % step == 0`.
    public static let step = 8

    /// Smallest and largest counts ``LTXVideoGenerationConfig`` accepts.
    public static let minimum = 9
    public static let maximum = 481

    /// A clip of `F` frames lasts `(F - 1) / fps` — the first frame is `t = 0`.
    /// This is *the* definition; anything printing `F / fps` is wrong.
    public static func seconds(forFrames frames: Int, fps: Double) -> Double {
        Double(frames - 1) / fps
    }

    /// Move `frames` onto the grid.
    public static func snap(_ frames: Int, rounding: GridRounding) -> Int {
        let below = ((frames - 1) / Self.step) * Self.step + 1
        switch rounding {
        case .down:
            return below
        case .nearest:
            let above = below + Self.step
            return (frames - below) <= (above - frames) ? below : above
        }
    }
}

/// Which way to move an off-grid frame count.
///
/// The two rules answer different questions, and conflating them means one
/// duration produces two different clips:
///
/// - ``down`` — "give me a count that does not exceed this". What the duration
///   head does with a *prediction*, matching upstream's
///   `seconds_to_clamped_num_frames`. At 24 fps a predicted 15 s becomes 353
///   frames (14.67 s).
/// - ``nearest`` — "give me the count closest to what was asked". What
///   ``FrameCountSpec`` does with an explicit `--frames 15s`, because the user
///   named a duration rather than had one estimated: 361 frames, exactly 15 s.
///
/// Both are correct for their own question. `FrameGridRoundingTests` asserts the
/// gap deliberately so it can never again be two green tests that disagree.
public enum GridRounding: Sendable {
    case down
    case nearest
}

/// How a caller expressed the clip length.
///
/// Frame counts must land on the `8k + 1` grid, so "15 seconds" is 361 frames
/// and not 360 — arithmetic nobody should have to do in their head. That
/// friction is why people write the duration into the prompt instead, where
/// ``PromptDuration`` shows it is silently ignored.
public enum FrameCountSpec: Sendable, Equatable {
    /// Let the LTX-2.5 duration head pick, from the prompt's content.
    case auto
    /// A frame count on the grid and within ``FrameGrid/minimum``...``FrameGrid/maximum``.
    case frames(Int)

    /// Parse `auto`, a frame count, or a duration like `15s` / `2.5s`.
    ///
    /// Every rejection happens here, before a caller spends minutes loading
    /// tens of gigabytes only to fail validation afterwards.
    ///
    /// - Returns: the spec, plus a note when a duration had to be adjusted onto
    ///   the grid. The note is for the caller to show; adjusting silently is how
    ///   you get someone counting frames in a bug report.
    public static func parse(
        _ raw: String, fps: Double = 24.0
    ) throws -> (spec: FrameCountSpec, note: String?) {
        let text = raw.trimmingCharacters(in: .whitespaces).lowercased()

        if text == "auto" { return (.auto, nil) }
        guard !text.isEmpty else { throw Self.malformed(raw) }

        if text.hasSuffix("s") {
            let digits = String(text.dropLast())
            // `Double("0x10")` is 16, and `Double("inf")`/`Double("1e400")` are
            // infinite — all three would sail past a bare `> 0` check, and the
            // last two used to abort the process inside `Int(_:)`.
            guard Self.isPlainDecimal(digits), let seconds = Double(digits),
                  seconds.isFinite, seconds > 0
            else { throw Self.malformed(raw) }
            return try Self.fromSeconds(seconds, fps: fps, original: raw)
        }

        guard Self.isPlainDecimal(text), !text.contains("."), let count = Int(text) else {
            throw Self.malformed(raw)
        }
        try Self.validate(count, fps: fps)
        return (.frames(count), nil)
    }

    /// Frames for a duration, snapped to the nearest grid point.
    ///
    /// - Returns: the count, and whether it is exactly the duration asked for.
    ///   `exact` is false whenever anything moved the value — including the
    ///   clamp to ``FrameGrid/minimum``, which used to report `true` and change
    ///   the length in silence.
    public static func frames(
        forSeconds seconds: Double, fps: Double = 24.0, rounding: GridRounding = .nearest
    ) -> (frames: Int, exact: Bool) {
        guard seconds.isFinite, fps.isFinite, fps > 0 else {
            return (FrameGrid.minimum, false)
        }
        // Clamp before converting: a duration of 1e30 s would otherwise overflow
        // Int on the way in.
        let capped = min(max(seconds, 0), FrameGrid.seconds(forFrames: FrameGrid.maximum, fps: fps))
        // Half-away-from-zero here, unlike the duration head's banker's rounding:
        // this converts a duration a person named, where "round 0.5 up" is what
        // they expect. The head matches Python's round() because it is
        // reproducing upstream's arithmetic on a prediction.
        let ideal = Int((capped * fps).rounded(.toNearestOrAwayFromZero)) + 1

        let snapped = (ideal - 1) % FrameGrid.step == 0
            ? ideal : FrameGrid.snap(ideal, rounding: rounding)
        let bounded = min(max(snapped, FrameGrid.minimum), FrameGrid.maximum)
        return (bounded, bounded == ideal && capped == seconds)
    }

    // MARK: - Validation

    private static func validate(_ count: Int, fps: Double) throws {
        guard count >= FrameGrid.minimum, count <= FrameGrid.maximum else {
            throw LTXError.invalidConfiguration(
                "Frame count must be between \(FrameGrid.minimum) and \(FrameGrid.maximum) "
                + "(\(Self.format(FrameGrid.seconds(forFrames: FrameGrid.maximum, fps: fps))) s at "
                + "\(Self.format(fps)) fps — the model's RoPE range). Got \(count).")
        }
        guard (count - 1) % FrameGrid.step == 0 else {
            let below = FrameGrid.snap(count, rounding: .down)
            let above = min(below + FrameGrid.step, FrameGrid.maximum)
            throw LTXError.invalidConfiguration(
                "Frame count must be \(FrameGrid.step)n+1 (9, 17, …, \(FrameGrid.maximum)). "
                + "Got \(count). Nearest valid: \(below) or \(above). Or pass a duration, e.g. "
                + "\(Self.format(FrameGrid.seconds(forFrames: count, fps: fps)))s.")
        }
    }

    private static func fromSeconds(
        _ seconds: Double, fps: Double, original: String
    ) throws -> (spec: FrameCountSpec, note: String?) {
        let ceiling = FrameGrid.seconds(forFrames: FrameGrid.maximum, fps: fps)
        guard seconds <= ceiling else {
            throw LTXError.invalidConfiguration(
                "\(Self.format(seconds))s is longer than this model can generate "
                + "(\(Self.format(ceiling))s, \(FrameGrid.maximum) frames at \(Self.format(fps)) fps). "
                + "Got '\(original)'.")
        }
        let (frames, exact) = Self.frames(forSeconds: seconds, fps: fps)
        let note = exact ? nil : "\(Self.format(seconds))s is not on the "
            + "\(FrameGrid.step)k+1 grid at \(Self.format(fps)) fps — using \(frames) frames "
            + "(\(Self.format(FrameGrid.seconds(forFrames: frames, fps: fps)))s)."
        return (.frames(frames), note)
    }

    private static func malformed(_ raw: String) -> LTXError {
        .invalidConfiguration(
            "Frames must be a count (121), a duration (15s), or 'auto'. Got '\(raw)'.")
    }

    /// Rejects hex (`0x10`), infinities, exponents and stray signs, all of which
    /// `Double(_:)` and `Int(_:)` accept in ways nobody typing `--frames` means.
    private static func isPlainDecimal(_ text: String) -> Bool {
        guard !text.isEmpty, text.count <= 12 else { return false }
        var seenDot = false
        for character in text {
            if character == "." {
                if seenDot { return false }
                seenDot = true
            } else if !character.isASCII || !character.isNumber {
                return false
            }
        }
        return text.first != "." && text.last != "."
    }

    /// Plain decimal, never scientific notation — `%g` renders 1200 as `1.2e+03`,
    /// which in a message about durations reads as noise.
    public static func format(_ value: Double) -> String {
        if value == value.rounded() && abs(value) < 1e9 {
            return String(Int(value))
        }
        return String(format: "%.2f", value)
    }
}

/// Finds a duration a person wrote into a prompt.
///
/// Nothing in LTX reads one. With an explicit `--frames` the prompt text is
/// never consulted; with `--frames auto` the duration head regresses a length
/// from connector tokens and never sees the characters. See
/// `docs/knowledge/pitfalls/duration-head-does-not-read-written-durations.md`.
///
/// This exists so the drop can be *reported*. It deliberately does not override
/// anything: `auto` means "let the model choose the natural length", and quietly
/// substituting a parsed number would change what the mode means.
public enum PromptDuration {

    /// How far into the prompt a duration may *start* and still count as a
    /// request.
    ///
    /// A request is stated up front — "15 seconds, 16:9 landscape…", "a
    /// 15-second clip", "Make it 15 seconds". Past that opening the duration is
    /// describing the scene:
    ///
    ///   "…as the sprinter runs the 100 metres in 9.58 seconds"
    ///   "…a microwave counting down from 30 seconds"
    ///   "…24 seconds on the shot clock as the crowd roars"
    ///
    /// None of those ask for anything, and warning about them would teach people
    /// to ignore the warning — which costs more than the missed case of someone
    /// burying a real request mid-sentence.
    static let requestWindow = 20

    private static let regex: NSRegularExpression? = {
        // The hyphen matters: "a 15-second clip" is at least as common as
        // "15 seconds", and matching one but not the other made the warning look
        // arbitrary. A bare `s` suffix is excluded — `1950s`, `80s` and `90s`
        // are decades, and `--frames 15s` covers the abbreviated form where it
        // is actually typed.
        let pattern = #"(?<![\w.])(\d{1,4}(?:\.\d+)?)[\s-]*(seconds?|secs?|minutes?|mins?)(?![\w])"#
        return try? NSRegularExpression(pattern: pattern, options: .caseInsensitive)
    }()

    /// A duration requested in the opening of `prompt`, in seconds, or nil.
    public static func find(in prompt: String) -> Double? {
        // A nil regex here would be a programming error in the literal above,
        // not a "no duration found" — but returning nil is the safe failure for
        // an advisory, so it is only ever a missing note.
        guard let regex else { return nil }

        // Search a slightly longer slice than the window so a match that merely
        // *ends* past it is still found, then judge it by where it starts.
        let head = String(prompt.prefix(Self.requestWindow + 24))
        let range = NSRange(head.startIndex..., in: head)
        guard let match = regex.firstMatch(in: head, range: range),
              match.range.location <= Self.requestWindow,
              let valueRange = Range(match.range(at: 1), in: head),
              let unitRange = Range(match.range(at: 2), in: head),
              let value = Double(head[valueRange])
        else { return nil }

        let unit = head[unitRange].lowercased()
        return unit.hasPrefix("m") ? value * 60 : value
    }
}
