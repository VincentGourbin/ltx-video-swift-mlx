---
type: Pitfall
title: A quantized transformer load first materialised the entire bf16 checkpoint
description: 'loadModels() evaluated the whole bf16 transformer before quantizing it, then evaluated the whole quantized model again — issue #86''s GPU timeout on a 36 GB machine. Fixed by evaluating per-block and dropping the source weight dict early.'
tags: [ltx25, quantization, memory, transformer, oom]
timestamp: 2026-09-05T00:00:00Z
---

Issue #86: a 36 GB M3 Max hit `kIOGPUCommandBufferCallbackErrorTimeout` while
"Loading transformer" at 50%, using `--transformer-quant int4`.

# Root cause

`LTXPipeline.loadModels()` (`Sources/LTXVideo/Pipeline/LTXPipeline.swift`), on
the quantized-load path, did:

```swift
transformer = LTXTransformer(config: transformerConfig, ...)
try LTXWeightLoader.applyTransformerWeights(transformerWeights, to: transformer!)
eval(transformer!.parameters())          // (1) materialize the WHOLE bf16 model
quantize(model: transformer!, ...)       // (2) quantize it
eval(transformer!.parameters())          // (3) materialize the WHOLE quantized model
```

Step (1) forces MLX to evaluate every block's bf16 weights *together*, in one
combined command buffer — the checkpoint is mmap'd and lazy up to that point,
so this is where the full ~42 GB (2.5-distilled bf16) actually gets read into
memory, before quantization even has a chance to shrink anything. Measured
peak on a 96 GB machine: **54519 MB** for a run that ends at int4 (~12 GB
transformer resident).

`transformerWeights` (the dictionary handed to `applyTransformerWeights`) was
also still alive throughout — a `let` local kept in scope for the rest of
`loadModels()` — so even after quantization replaced `transformer`'s own
parameter references with quantized arrays, the original bf16 arrays stayed
reachable (and thus resident) through that second reference.

# Fix

`fix/quantized-load-per-block`:

1. `transformerWeights` becomes a `var`, explicitly `.removeAll()`'d
   immediately after `applyTransformerWeights` — nothing keeps the original
   bf16 dictionary alive once the transformer has its own copies.
2. The pre-quantization full `eval()` only runs when *no* quantization is
   configured (bf16-only path) — and even then, through the new
   `evalParametersPerBlock` helper (below), not one combined call.
3. `evalParametersPerBlock(_ model: Module)` — for `LTXTransformer` or
   `LTX2Transformer`, evaluates each `transformerBlocks[i].parameters()`
   individually, then the small remaining top-level parameters — used both
   before quantization (bf16-only case) and after (all cases), replacing
   every `eval(transformer!.parameters())` / `eval(ltx2.parameters())` call
   in `loadModels()` and `loadAudioModels()`.
4. `LTXMemoryManager.logMemoryState("after transformer load")` (and an
   equivalent for the LTX2 audio path) added permanently — this class of bug
   is otherwise invisible until someone happens to run on a memory-constrained
   machine.

Measured after the fix, same int4 run: peak **37180 MB** — down from 54519 MB
(≈17 GB less transient overhead). `VAE raw output` is byte-identical
before/after on a plain (non-quantized) generation, confirming the fix changes
only *when* memory is touched, not the numerics.

# What this fix does not cover

The transformer is not the whole story for issue #86 on a 36 GB machine. The
LTX-2.5 Gemma 4 text encoder, loaded *before* the transformer in the same
`loadModels()` call, independently costs far more than expected when
quantized — see
[gemma4-quantize-does-not-release-bf16](gemma4-quantize-does-not-release-bf16.md).
Fixing the transformer alone does not get a 36 GB machine reliably under
budget.

# Guarded by

No automated regression test pins the *memory number* (peak memory isn't
something `swift test` asserts on); the fix is verified manually via
`LTXMemoryManager.logMemoryState` and the reproduction steps in the PR. The
*numerical* side (quantization still produces the same output) is guarded by
the existing `VAE raw output` byte-identity check and the full test suite.
