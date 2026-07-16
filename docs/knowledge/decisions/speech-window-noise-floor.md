---
type: Decision
title: Speech-window detection thresholds — absolute floor + credible noise-floor offset
description: max(-35 dBFS, 10th-percentile frame RMS + 10 dB), the floor trusted only when ≥15 dB below the loudest frame. Peak-relative thresholds were tried and rejected.
tags: [audio, silence-detection, lipdub, alignment]
timestamp: 2026-07-16T00:00:00Z
---

`AudioPreprocessor.detectSpeechWindow` classifies 10 ms frames against the
**stricter** of:

1. an absolute floor: `-35 dBFS` (the historical behavior), and
2. the clip's own noise floor (10th-percentile frame RMS) `+ 10 dB` —
   **only when that floor is credible**, i.e. sits ≥ 15 dB below the loudest
   frame's RMS.

# Why each piece exists

- **Absolute alone failed on enrolled voices**: Voxtral voice enrollment
  reproduces the *reference recording's* noise floor. Measured case:
  "silences" at **-32.5 dB** — above -35, so every frame counted as speech,
  no window was found, and lip-sync drifted (Fluxforge asks B2/A1).
- **Peak-relative (peak − 25 dB) was tried and rejected**: the measured
  peak-to-floor gap can be under 25 dB (synthetic reproduction: 19 dB), so
  the threshold lands below the floor and changes nothing.
- **The 15 dB credibility gate exists because of a review-caught regression**:
  on tightly-trimmed clips (<10% silence frames) the 10th percentile lands on
  quiet SPEECH; a threshold derived from it either clips soft onsets or
  rejects every frame (full-clip fallback) — regressing exactly the clean
  recordings the absolute threshold handled. A real noise floor is always far
  below the loudest frame; quiet speech is not.

# Examples

Regression tests pin all three regimes
(`AudioPreprocessorTests`): the -32.5 dB enrolled case, the quiet-noise-only
clip, and the <10%-silence tightly-trimmed clip. A real LipDub run with a
fabricated -33.4 dB-floor target audio (ffmpeg pink noise mix) reproduced the
field failure and validated the fix end-to-end — recipe in the
[validation protocol](/docs/testing/PR36-validation-protocol.md).

# Citations

[1] Fluxforge asks B2/A1 (ffmpeg astats/silencedetect measurements on two
LipDub outputs), resolved in PR #36.
