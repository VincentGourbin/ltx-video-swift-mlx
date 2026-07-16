---
type: Pitfall
title: The LipDub IC-LoRA fusion is destructive — re-fusing applies the delta twice
description: No pristine originals are kept for the 22B; any path that fuses onto an already-fused transformer, or exports/unfuses around it, corrupts silently.
tags: [lora, lipdub, fusion, corruption]
timestamp: 2026-07-16T00:00:00Z
---

`generateLipDub` fuses the IC-LoRA with `W' = W + delta` and deliberately
discards the originals (keeping pristine copies of 1344 layer-pairs of a 22B
model costs ~10+ GB). Consequences, all of the "no error, wrong output" kind:

- **Re-fusing the same LoRA** on a surviving transformer applies the delta
  twice (burned/saturated output).
- **`generateVideo`/`generateRetake`** on fused weights generate with the
  LipDub delta baked in.
- **`exportQuantizedTransformer`** would persist the delta to disk as base
  weights — every future run loading that file is corrupted.
- **`fuseLoRA()` (generic) then `generateLipDub` then `unfuseLoRA()`**: the
  unfuse restores pre-both originals on overlapping layers, partially wiping
  the IC-LoRA delta while the state still claims it is fused.
- **`LTX_LIPDUB_SKIP_LORA=1`** on a still-fused transformer would "skip
  fusion" but run fused anyway, invalidating the A/B diagnostic.

# The defense

All five are guarded since PR #36: the pipeline tracks a `LipDubFusionRecord`
(canonical path + mtime) and throws on every misuse; consecutive
`generateLipDub` calls with the same, unchanged file legitimately reuse the
fused transformer (see [the reuse decision](/docs/knowledge/decisions/lipdub-fusion-reuse-policy.md)).
When adding a NEW entry point that touches transformer weights, call
`ensureNoLipDubLoRAFused(wouldCorrupt:)` at its top — the review found the
export path missing exactly that guard.

Known limitation: the guard is checked at entry; actor reentrancy across
`await` points means two concurrent generations on one pipeline could still
interleave. Concurrent generations are unsupported anyway.

# Citations

[1] PR #36 review findings 3, 4, 5 — see
[the investigation](/docs/knowledge/investigations/lipdub-segmentation-asks-2026-07.md).
