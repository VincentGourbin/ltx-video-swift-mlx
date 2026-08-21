---
type: Pitfall
title: A temporally densified clip must be positioned at its new rate, not the source's
description: A temporal round doubles the frames at constant duration, so the refined clip runs at 2x fps. Positioning it at the source's fps makes the model read a 4.9 s clip as 9.8 s — half-speed motion, and coordinates past the 20 s RoPE range on a long canvas.
tags: [rope, temporal, dfr, interpolation, fps, root-cause]
timestamp: 2026-08-21T00:00:00Z
---

RoPE temporal coordinates are **seconds**, not frame indices: the position grid
divides a latent frame's pixel-span midpoint by `fps`. Everything downstream —
how far apart two frames feel, whether a clip fits the model's 20 s range —
follows from that division.

A temporal round doubles the latent frame count and doubles the frame rate, so
the clip's *duration* is unchanged. Port only the first half of that and the
grid still divides by 24 while the clip now holds 2n−1 frames:

| | last latent frame | fps | coordinate |
|---|---|---|---|
| source, 16 latent frames | 15 | 24 | 4.88 s |
| densified, 31 latent frames | 30 | 48 | 4.94 s ✓ |
| densified, positioned at 24 | 30 | 24 | 9.88 s ✗ |

The failure is quiet, which is what makes it worth a note. The base grid and the
anchors were both computed at 24, so they agreed with each other; nothing was
misaligned, the output was plausible, and the clip simply read to the model as
one running at half speed. Two consequences follow:

* motion is interpreted at half its real velocity, which is exactly the wrong
  prior for a round whose job is to *invent* intermediate motion;
* on a long canvas the coordinates run past `maxPos[0] = 20 s`, where the RoPE
  range is no longer defined by anything the model saw in training.

Upstream conditions its temporal rounds at `min(2 × fps, 60)` — the cap matters
from the second round on, where 24 → 48 → 96 would otherwise exceed it.

**The check that catches it**: the last frame of the refined clip must land at
the same second as the last frame of the source, to within one latent frame's
span. Pinned in `DensifiedClipRateTests`.

## What it does *not* change, measured

A/B on the bench clip (121 source frames, 5 s, single window, seed 3), the
refined 241-frame output against its source:

| positioning | fidelity mean | worst | inter-frame delta | max z |
|---|---|---|---|---|
| 48 fps (correct) | 23.45 dB | 15.68 dB | 0.0109 | 5.21 |
| 24 fps (the bug) | 23.86 dB | 16.98 dB | 0.0114 | 5.10 |

A tie, inside single-seed noise, and if anything the wrong one scores slightly
better on fidelity. That is worth stating plainly: the fix is right because it
is what the model was trained on and what upstream does, not because this clip
improved.

The case where it must bite is the one this clip cannot show — a source of 10 s
or more, where doubling the coordinates puts the tail past `maxPos[0] = 20 s`
and the RoPE range stops being defined by anything the model saw. Untested here.

See also [[renoise-level-needs-its-anchor]] — the same temporal round, the other
parameter that does not survive being copied without its context.
