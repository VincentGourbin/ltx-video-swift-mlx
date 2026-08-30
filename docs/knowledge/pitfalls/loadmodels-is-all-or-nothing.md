---
type: Pitfall
title: loadModels() rebuilds everything — never call it to recover one component
description: It is not gated per component, so recovering an unloaded prompt encoder that way rebuilds the 22B and both VAEs mid-run, and discards any fused LoRA with the transformer it was fused into.
tags: [memory, pipeline, gemma, lora, fusion]
timestamp: 2026-08-30T00:00:00Z
---

`LTXPipeline.loadModels()` reads as if it were idempotent and cheap when things
are already loaded. It is neither: it re-resolves the checkpoint and
*unconditionally* rebuilds the prompt encoder, the transformer, the VAE decoder
and the connector, whatever is currently resident. Two costs follow, and both
have been paid in this repo:

1. **Minutes, and a doubled peak.** The dev-model retake unloaded Gemma after
   encoding the positive prompt, then called `loadModels()` to get an encoder
   back for the negative prompt — rebuilding the 22B transformer and both VAEs
   in the middle of a generation, with the old transformer still referenced
   until ARC caught up.
2. **A fused LoRA silently disappears.** The new transformer is pristine, so a
   LipDub IC-LoRA fusion (or any `fuseLoRA()`) is gone. The fusion record is
   cleared with it, so nothing reports the loss — the next segment simply
   re-fuses, at full cost.

**Use `ensureTextEncoderLoaded()`** instead: it rebuilds the prompt encoder
alone, from the source `loadModels()` was pointed at (remembered at load time),
and touches nothing else. Every text-encoding entry point in the pipeline calls
it, so a dropped encoder repairs itself.

It deliberately does nothing when `loadModels()` was never called — the
text-side connector survives every mid-run unload, so its absence means "nothing
was ever loaded", where the right answer is still the "not loaded" error the
caller expects rather than a surprise 42 GB download.

# Examples

```swift
// Wrong: rebuilds the world, and drops any fused adapter.
if !(await pipeline.isGemmaLoaded) { try await pipeline.loadModels() }

// Right: the encoder comes back on its own.
try await pipeline.ensureTextEncoderLoaded()
```

If a *host* needs the same guarantee across its own segment loop, it does not
need to do anything: the pipeline repairs the encoder at the next encode.

# Citations

[1] The retake dev path carried exactly this call, found while splitting the
    unload gating — see
    [the unload-gating decision](/docs/knowledge/decisions/unload-gating-semantics.md).
[2] `predictFrameCount` had the same shape and now only falls back to
    `loadModels()` when nothing at all is loaded.
