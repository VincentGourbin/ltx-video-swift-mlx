---
type: Pitfall
title: The duration head ignores durations written in the prompt
description: '`--frames auto` regresses a clip length from connector tokens, not from text — a written duration never moves the prediction to what was asked. Measured August 2026 (video tokens only) and re-measured 2026-09-05 (video + audio tokens).'
tags: [ltx25, duration-head, frames, prompting]
timestamp: 2026-08-26T00:00:00Z
---

`--frames auto` does **not** honour a duration written in the prompt, and the
failure is silent: it returns a plausible number that simply is not the one
asked for.

Measured August 2026 on LTX-2.5 distilled, 24 fps, same scene each time
(`A quiet late-night laundromat with flickering fluorescent lights.`), **with
video connector tokens only** — before
[the duration-head-audio-tokens fix](duration-head-needs-audio-tokens.md) gave
the head the audio connector tokens too:

| Prompt | Raw prediction | Frames |
| --- | --- | --- |
| the scene, no duration | 23.5 s | 473 (clamped) |
| `3 seconds. ` + the scene | **23.5 s** | 473 (clamped) |
| `15 seconds. ` + the scene | 16.88 s | 401 |
| `20 seconds. ` + the scene | 19.5 s | 465 |

The `3 seconds.` prefix was the decisive row: **byte-identical** to no prefix.
`15` and `20` did move the value, but to 16.9 s and 19.5 s — learned
correlation with what those tokens changed in the scene representation, not a
reading of the numeral.

Re-measured 2026-09-05 with **video AND audio connector tokens**, matching
upstream (the fix above):

| Prompt | Raw prediction | Frames |
| --- | --- | --- |
| the scene, no duration | 4.09 s | 97 |
| `3 seconds. ` + the scene | 3.59 s | 81 |
| `15 seconds. ` + the scene | 4.22 s | 97 |
| `20 seconds. ` + the scene | 3.59 s | 81 |

Two things changed, and the conclusion is not the same as before:

- **The `3 seconds.` prefix is no longer byte-identical** — it now moves the
  prediction by ~0.5 s. The head is not byte-blind to the prefix once it also
  sees the audio connector's tokens.
- **It still does not read the numeral.** The four rows are not ordered by the
  requested duration at all (`20 seconds.` predicts *less* than `15 seconds.`,
  and both predict less than the no-duration baseline) — a literal reading
  would put `3 seconds.` near 3 s and `20 seconds.` near 20 s. Instead every
  value now sits in a narrow 3.6–4.2 s band, regardless of what duration is
  asked for.

The mechanism (a single learned query pooling the connector's output — see
below) hasn't changed; what changed is which connector outputs the head
receives, and that measurably shifted both the absolute predictions and how
much a written duration nudges them.

# Why

The head never sees text. Per upstream's own docstring it is a *"small
regression head that predicts shot duration from frozen Connector outputs"* —
`ltx_core/duration_head/duration_head.py`. It reads the connector's output, a
semantic conditioning signal built for the diffusion transformer. That signal is
then pooled by a **single learned query** (`query_tokens [1, 256]`, in the
checkpoint) into one 256-dim vector, and a two-layer MLP regresses it to one
scalar. There is no path by which a numeral survives as a number.

What the head does weigh is shot content. A style-dense prompt saturated with
close-framing language ("one-handed phone-camera feel", "delayed autofocus at
close range") predicted **5.16 s → 121 frames** (video tokens only, August
2026) while asking for 15 seconds; re-measured 2026-09-05 with video + audio
tokens, the same prompt predicts **6.69 s → 153 frames** — still nowhere near
what was asked for. The prompt enhancer made it worse in the obvious way: its
rewrite *dropped the "15 seconds" entirely* while keeping "16:9 landscape
aspect ratio". With `--enhance-prompt` — upstream's ordering — the request
never reaches the head at all. The enhanced prompt predicted 5.28 s, the same
121 frames (video-tokens-only measurement, not re-verified under the
video+audio fix).

# Consequences

- **Want a specific length? Pass `--frames`.** 15 s at 24 fps is `--frames 361`.
- **Two ceilings, and they differ on purpose.** An explicit `--frames` reaches
  481 (20 s). `--frames auto` tops out at **473**: it rounds a *prediction*
  down to the grid rather than to the nearest point, matching upstream's
  `seconds_to_clamped_num_frames`. Same reason `--frames 15s` gives 361 while a
  predicted 15 s gives 353 — a request and an estimate are different questions,
  and `GridRounding` names both rules.
- **`wasClamped` distinguishes the two failure shapes.** "The model asked for
  23.5 s and got capped" is not "the model asked for 5 s".
- A `"15 seconds, 16:9 landscape"` preamble is a prompt shape borrowed from
  other tools. LTX parses neither: duration is `--frames`, aspect is `-w`/`-h`.

# Guarded by

`DurationPromptE2ETests` (gated on `LTX25_CACHE_ROOT` — a cache *root* with per-component subdirectories, deliberately not the flat `LTX25_MODELS_DIR` four other suites read) asserts that a `3 seconds.`
prefix does not land on a literal 3.0 s reading, so if a future checkpoint
*does* learn to read durations, the test fails and says the premise needs
revisiting. Its ceiling-demonstrating test was retired once the scene it used
stopped clamping under the video+audio fix — see
[duration-head-needs-audio-tokens](duration-head-needs-audio-tokens.md) — the
473-frame ceiling itself is still covered on synthetic input by the pure
`DurationGridSnap` suite. See also
[the LTX-2.5 checkpoint diff](/docs/knowledge/investigations/ltx-2.5-checkpoint-diff-2026-08.md).
