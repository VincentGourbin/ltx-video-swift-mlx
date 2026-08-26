---
type: Pitfall
title: The duration head ignores durations written in the prompt
description: '`--frames auto` regresses a clip length from connector tokens, not from text. A "3 seconds." prefix returns the same value as no prefix at all. Measured August 2026.'
tags: [ltx25, duration-head, frames, prompting]
timestamp: 2026-08-26T00:00:00Z
---

`--frames auto` does **not** honour a duration written in the prompt, and the
failure is silent: it returns a plausible number that simply is not the one
asked for.

Measured on LTX-2.5 distilled, 24 fps, same scene each time
(`A quiet late-night laundromat with flickering fluorescent lights.`):

| Prompt | Raw prediction | Frames |
| --- | --- | --- |
| the scene, no duration | 23.5 s | 473 (clamped) |
| `3 seconds. ` + the scene | **23.5 s** | 473 (clamped) |
| `15 seconds. ` + the scene | 16.88 s | 401 |
| `20 seconds. ` + the scene | 19.5 s | 465 |

The `3 seconds.` prefix is the decisive row: **byte-identical** to no prefix.
`15` and `20` do move the value, but to 16.9 s and 19.5 s — the head responds to
what those tokens change in the scene representation, by learned correlation,
not by reading them as quantities. If it parsed numerals, `3 seconds` would
return 3 s.

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
close range") predicted **5.16 s → 121 frames** while asking for 15 seconds. The
prompt enhancer made it worse in the obvious way: its rewrite *dropped the
"15 seconds" entirely* while keeping "16:9 landscape aspect ratio". With
`--enhance-prompt` — upstream's ordering — the request never reaches the head at
all. The enhanced prompt predicted 5.28 s, the same 121 frames.

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

`DurationPromptE2ETests` (gated on `LTX25_MODELS_DIR`) asserts the `3 seconds.`
equivalence directly, so if a future checkpoint *does* learn to read durations,
the test fails and says the premise needs revisiting. See also
[the LTX-2.5 checkpoint diff](/docs/knowledge/investigations/ltx-2.5-checkpoint-diff-2026-08.md).
