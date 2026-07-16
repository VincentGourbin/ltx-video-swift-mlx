---
type: Pitfall
title: Keyframe conditioning must append guide tokens, never overwrite latent slots
description: Slot-injection is structurally wrong for any keyframe past frame 0 — one VAE-encoded frame cannot stand in for a latent slot representing 8 pixel frames of motion.
tags: [keyframes, conditioning, rope, i2v]
timestamp: 2026-07-16T00:00:00Z
---

The legacy "overwrite the latent slot" conditioning only ever worked at slot 0
(the `+1` frame of the `8n+1` layout). At slot N≥1 a single-frame VAE encoding
is structurally wrong for a slot that represents 8 pixel frames of motion —
the cause of the grainy artifacts at middle/last keyframes (issue #21).

# The defense

The current primitive (`AppendedGuideTokens.swift`): each keyframe is
VAE-encoded, patchified, and **appended** to the video token sequence with
RoPE temporal position `(pixelFrameIndex + 0.5) / fps` and timestep 0 (clean
reference). The velocity is cropped back to the original token count before
the scheduler step — appended tokens never enter the final latent, they only
steer attention. The old `injectKeyframeLatents`/`buildKeyframeMask` helpers
are deleted; never reintroduce the pattern. Cost: ~+50% generation time at
small frame counts (more attention tokens).

The same primitive backs the LipDub audio reference (appended audio latent at
negative RoPE positions) and is the natural basis for the segment-continuation
proposal (issue #35).

# Citations

[1] Issue #21 (grainy artifacts at non-zero keyframes), fixed May 2026.
[2] Lightricks `keyframe_cond.py` (`AudioConditionByReferenceLatent`).
