---
type: Pitfall
title: A smoothness metric will clear a seam where the subject changes
description: Mean inter-frame difference measures how gradual a clip is, not whether it still shows the same thing. It cleared a tile seam (z = +0.32) that the eye saw as a jump, and stayed blind to a second car appearing. Per-frame fidelity against the source found both instantly.
tags: [measurement, tiling, temporal, method, root-cause]
timestamp: 2026-08-20T00:00:00Z
---

Tiled temporal interpolation was checked with the obvious instrument: the
mean absolute difference between consecutive frames, looking for a spike at
each seam. It reported one seam perfectly clean (z = +0.32) and blamed the
other on content, since the source had its own peak at the same instant.

Both conclusions were wrong. The user saw a jump at the "clean" seam and a
*different car* appearing after the other one. Measuring each output frame
against the source frame it derives from showed it at once:

| t | seam | identity vs source |
|---|---|---|
| 5.00 s | 1 | 22.5 dB |
| 10.00 s | 2 | **13.4 dB** |
| elsewhere | — | 25-28 dB |

A smoothness metric answers "does this change abruptly?". A seam where each
tile has *gradually* drifted to a different subject is smooth and wrong. The
metric cannot see it by construction, and no amount of staring at its output
would have revealed that — it took a human watching the video.

# The defense

- For anything that re-renders existing content — interpolation, upscaling,
  retake, tiling — measure **fidelity to the source, per frame**, not
  smoothness. Smoothness is a secondary check, useful only once identity is
  established.
- When a measurement clears something a person reports as broken, the
  measurement is the first suspect, not the person.
- Root cause once measured properly: tiles renoised independently at sigma
  0.975 each rebuilt their own subject. Tiled runs now default to 0.725 with
  dense anchoring (worst-case identity 13.4 → 24.3 dB).

# Postscript: the fix that lost its own bake-off

Carrying the previous tile's denoised output into the next tile's lead-in —
upstream's `_merge_carry_forward_keyframes` — was expected to restore the
single-window noise level for tiled runs. It does repair the failure it was
written for (seam identity 13.4 → 23.5 dB) but still loses to the plain tiled
defaults:

| tiled config | worst identity | seam spike | wall time |
|---|---|---|---|
| 0.975, no carry | 13.4 dB | 35.5 | 43 min |
| **0.725 + dense anchors** | **24.3 dB** | **13.1** | **35 min** |
| 0.975 + carry-forward | 23.5 dB | 18.3 | 72 min |

Our lead-in is two latent frames: too little carried signal to pay for
doubling the anchor count. Upstream carries far more, across multiple rounds.
Shipped off by default, kept behind a flag — a mechanism that works but does
not earn its place is worth keeping only where it is honest about that.

# Citations

[1] Bench runs 2026-08-20, `interpolate` on the 337-frame 2CV clip, 3 tiles.
[2] Related: [[renoise-level-needs-its-anchor]].
