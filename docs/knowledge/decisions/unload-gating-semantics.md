---
type: Decision
title: Mid-run unloads are gated per component — unloadAfterUse is only the default
description: Gemma and the transformer followed one flag, which made fusion reuse reachable only at .disabled; LTX-2.5's 26 GB encoder broke that bargain, so each now has its own override.
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

# Revised 2026-08-30 — per-component gating

The single flag made **fusion reuse reachable only at `.disabled`**:
`recommended(forRAMGB:)` returns `.light` for 65–96 GB, `.light` unloads, and
the transformer unload clears the LipDub fusion record with it. On a 96 GB
machine — the class the app ships on — every chained segment therefore paid a
full `loadModels()` **plus** a re-fusion of the 22B. The other horn was no
better: `.disabled` keeps the reuse but holds the prompt encoder resident, and
in LTX-2.5 that encoder is **26 GB**, not the 7.5 GB this decision was costed
against. Neither branch was affordable.

What changed:

- `MemoryOptimizationConfig.unloadTextEncoderAfterUse` and
  `unloadTransformerAfterUse` (both `Bool?`, `nil` = follow `unloadAfterUse`,
  so every existing preset and caller behaves exactly as before), read through
  `unloadsTextEncoder` / `unloadsTransformer`.
- `.keepingTransformer()` / `.keepingTextEncoder()` derive a preset without
  spelling the flags out.
- `LTXPipeline.ensureTextEncoderLoaded()` rebuilds the prompt encoder **alone**,
  from the source `loadModels()` used (remembered at load time). Every
  text-encoding entry point calls it, so dropping the encoder is now cheap to
  undo — that is what makes the split safe rather than just possible.

The intended setting for chained segments is `.recommended(forRAMGB:)`
`.keepingTransformer()`: encoder freed after each encode, transformer and its
fusion kept, no re-fusion from segment 2 on.

The original decision still holds where it was aimed: `unloadAfterUse` remains
the one switch a caller needs, and `.disabled` still means keep everything. It
is now a default rather than the whole story.

# Citations

[1] PR #36 (E2E-caught fix + review findings).
[2] Fluxforge asks §7 (2026-08-30): ~96 GB RSS over 3–4 LipDub segments,
    reported as a missing `Memory.clearCache()` — which was already there. The
    growth was the reload-and-re-fuse cycle above.
[3] `LipDubReuseE2ETests.encoderReloadsAloneWhileTheFusionSurvives` covers the
    split end to end (gated, `LTX_E2E_LIPDUB=1`).
