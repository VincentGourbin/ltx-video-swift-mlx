---
type: Decision
title: The frame cap is 481 because that is the RoPE positional range, not a preference
description: Temporal RoPE coordinates are seconds normalized by maxPos[0] = 20 s; 481 frames = 20 s at 24 fps. The old 257 was an invented bound.
tags: [rope, validation, frames, config]
timestamp: 2026-07-16T00:00:00Z
---

`LTXVideoGenerationConfig.validate()` accepts `numFrames` up to **481**. The
derivation, so it is never re-litigated:

- Temporal RoPE coordinates are **seconds**: pixel-frame middle / fps
  (`pixelCoordsToSeconds` in `LTXRoPE.swift`).
- They are normalized by `LTXTransformerConfig.maxPos[0] = 20` (seconds).
- 20 s × 24 fps + 1 = **481** frames. Beyond that, fractional positions
  exceed 1.0 — outside the embedding's designed range.
- The audio side is linear in duration (no cap); the VAE decode is
  temporally tiled. No other constraint binds.

The previous cap of 257 (~10.7 s) dated from the initial commit as
"reasonable bounds" — no upstream source. It was the blocking constraint for
dubbing (real user case: a 46 s line) and its removal halves the number of
segment cuts.

# Caveats

- Training clips are typically ~10 s: expect some quality softening on very
  long videos even within the bound. A 481f vs 241f same-seed comparison was
  generated for the PR #36 validation — judge visually per use case.
- If `maxPos` ever changes (longer-context variant) or output fps becomes
  configurable, the 481 in `validate()` must follow — the review flagged that
  the bound is hardcoded rather than derived; acceptable while both constants
  live in `LTXConfig.swift`.

# Citations

[1] Fluxforge asks item B1, resolved in PR #36.
