---
type: Decision
title: Segment continuation anchors on the previous tail's LAST latent frame, at position 0
description: A motion-bearing tail latent replaces the still image as the frame-0 guide keyframe; measured seam PSNR improves 17.4 → 24.6 dB. Overlap-and-trim is the concatenation contract.
tags: [lipdub, segmentation, continuation, keyframes, issue-35]
timestamp: 2026-07-17T00:00:00Z
---

Image-mode LipDub segments used to re-anchor every segment on the same still
image — visible jumps at each cut (issue #35). The shipped design
(`continuationTailPath` / `--continuation-tail`):

- The caller passes a clip whose **first 9 pixel frames** are the previous
  segment's tail; its VAE encoding yields 2 latent frames and the **last one**
  (carrying 8 frames of actual motion, not just a pose) becomes the frame-0
  guide keyframe — the existing appended-guide-token machinery
  (`EncodedKeyframe` at slot 0, σ=0, RoPE `(0+0.5)/fps`) is reused unchanged.
- **Position 0, not negative positions**: the first output frame reproduces
  the anchor, and the app drops one frame at concatenation
  (overlap-and-trim). Chosen over negative RoPE positions because it reuses
  the keyframe path verbatim and gives a deterministic trim contract instead
  of unanchored extrapolation.
- Video-reference mode rejects the parameter — its continuity comes from
  segmenting the source video itself.

# Measured (July 2026, 2× 57-frame segments, 384×256, same seeds)

| Seam (last frame of seg 1 vs first frame of seg 2) | PSNR |
|---|---|
| Baseline: segment 2 re-anchored on the still image | 17.4 dB (visible jump) |
| Continuation: anchored on segment 1's tail latent | **24.6 dB** |

Residual gap to identity is the VAE + denoise round-trip of the anchor frame,
not a positioning error. Known open point: identity drift over MANY chained
segments (the original still is no longer an anchor once chaining starts) —
if it bites, re-introduce the still as a secondary keyframe or re-anchor
every N segments.

# Citations

[1] Issue #35; implementation in `generateLipDub(continuationTailPath:)`.
[2] Builds on [the keyframe append pitfall](/docs/knowledge/pitfalls/keyframes-append-not-inject.md).
