---
type: Decision
title: A retake picks its stream with a modality, not with a strength of zero
description: Freezing a stream is a σ = 0 timestep, not a re-noise level — so .videoOnly/.both/.audioOnly, and .audioOnly re-muxes the source picture instead of decoding it.
tags: [retake, audio, api, dual-stream]
timestamp: 2026-08-30T00:00:00Z
---

The app asked for per-modality retake control and proposed two shapes: (a) two
strengths, where `retakeStrength = 0` freezes a stream, or (b) an explicit
modality switch. (b) was chosen, and (a) is not implementable as described.

**Freezing is a timestep, not a strength.** The dual-stream path already froze
the audio: `audioTimesteps` is `σ` when the audio is regenerated and `0` when it
is not, and the frozen stream stays in the forward pass as cross-modal context.
Nothing about that mechanism is expressible as a re-noise level. Independently,
`LTXVideoGenerationConfig.validate()` rejects `retakeStrength == 0` — and
`retakeStrength` is *unused* by `generateRetake` anyway (regenerated frames
always start from pure noise, matching the Lightricks reference), so the
strength the ask wanted to overload carries no meaning to overload.

Hence `RetakeModality`:

| Modality | Picture | Sound |
| --- | --- | --- |
| `.videoOnly` (default) | denoised | source, passed through |
| `.both` | denoised | denoised from noise |
| `.audioOnly` | source, re-muxed untouched | denoised from noise |

`.videoOnly` and `.both` are the pre-existing `regenerateAudio == false / true`,
which stays as a deprecated two-value view of the property. Only `.audioOnly` is
new behaviour.

# Two consequences worth knowing

**`.audioOnly` does not decode the picture.** The video latent is never noised
and never stepped, so decoding it would only add a VAE round-trip's losses to
frames that already exist. The result carries the source frames, read back and
re-muxed: bit-identical output, and no decode paid. That also makes an
audio-only retake far cheaper than a video one — the transformer passes remain,
the decode does not.

**The audio Euler step moved out of the transformer helper.** It used to be
applied inside the per-pass closure, which the dev path calls two or three times
per step (CFG, then STG) — the audio latent was advanced once per pass, each
time from the already-advanced value, so a dev-model `regenerateAudio` run took
2–3× the intended step per schedule step. It is now applied once per step, from
the conditioned pass's velocity (audio is not CFG-guided). The distilled path
makes exactly one pass per step and is unchanged, bit for bit.

# `audioRetakeStrength`, and why it is audio-only

Starting the audio from pure noise gives a new soundtrack with no relation to
the source. `audioRetakeStrength < 1` instead enters the schedule lower, from
the source track: `x_σ = σ·noise + (1 − σ)·x₀`, the same renoise the temporal
refinement pass uses. Rhythm and ambience survive in proportion, and fewer steps
run.

Two properties are deliberate:

- **Only the trained sigmas are reachable.** The schedule is filtered
  (`σ <= strength`, plus the terminal `0`), never interpolated, so on the
  distilled model a strength snaps to its 9-value schedule: `0.9` → `0.909375`
  (the level stage 2 renoises to), `0.8` → `0.725`, `0.5` → `0.421875`. Below
  `0.421875` no step remains and the call throws rather than running an empty
  schedule.
- **It is refused outside `.audioOnly`.** Video and audio share one sigma
  schedule, so truncating it for `.both` would truncate the picture's schedule
  too — a silently different generation. Rejecting the combination is what keeps
  the knob honest.

# Rejected

- **Two strengths (option a).** See above: no strength participates in freezing,
  and `0` is already a validation error. `audioRetakeStrength` is a schedule
  entry point, not the frozen/regenerated switch the ask conflated it with —
  which is why both exist rather than one.

# Citations

[1] Fluxforge asks §4 (2026-08-26), answered 2026-08-30.
[2] `RetakeModalityTests` (config-level), `RetakeAudioOnlyE2ETests` (gated,
    `LTX_E2E_RETAKE=1`): bit-identity of the returned picture, passthrough not
    armed for a regenerated track, and the missing-audio-models refusal.
