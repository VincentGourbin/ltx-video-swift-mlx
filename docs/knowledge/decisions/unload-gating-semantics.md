---
type: Decision
title: unloadAfterUse gates ALL mid-run unloads — .disabled means keep everything
description: Gemma and the transformer follow the same preset flag; the cost is ~7.5 GB extra at decode peak with .disabled, traded for reusable state across consecutive runs.
tags: [memory, presets, gemma, pipeline]
timestamp: 2026-07-16T00:00:00Z
---

Historically the pipelines unloaded Gemma (+tokenizer) unconditionally after
text encoding, and the transformer per the `unloadAfterUse` preset flag. The
unconditional Gemma unload broke consecutive-run scenarios: the second
`generateLipDub` in one process failed with "Tokenizer not loaded" before
even reaching the fusion-reuse path (caught by the gated E2E, not by any unit
test).

Decision: **every mid-run unload follows `memoryOptimization.unloadAfterUse`**
— Gemma via `unloadGemmaIfConfigured()`, the transformer via the existing
gated blocks (which also clear the LipDub fusion record). The `.disabled`
preset now truly means "keep everything resident".

# Trade-offs accepted

- With `.disabled`, Gemma (~7.5 GB) stays resident through denoising and VAE
  decode — a peak-memory increase vs pre-change for `.disabled` users. The
  preset targets 96 GB+ machines (`recommended(forRAMGB:)` only returns it
  above 96), where this is acceptable.
- The rejected alternative (keep unloading, lazily reload Gemma at the next
  run's start) would save the resident 7.5 GB at the cost of a per-segment
  reload and more code; revisit if 48 GB-class machines need the reuse path.
- Any `loadModels()` call that assumed "Gemma was unloaded above" must gate
  on `gemmaModel == nil` — the retake dev-model negative-prompt path had
  exactly that stale assumption (review finding, fixed).
- `generateLipDub` clears the MLX workspace cache before returning: with
  `.disabled` nothing else does, and decode buffers accumulating across
  segments degraded later runs.

# Citations

[1] PR #36 (E2E-caught fix + review findings).
