---
type: Pitfall
title: Quantizing the Gemma 4 text encoder adds memory instead of saving it
description: 'On-the-fly quantization of gemma4-12b-ltx-v1 (MoE, SwitchGLU) is additive, not replacing — bf16 measures ~22.7 GB, int4 measures ~29.1 GB (bf16 + ~6.2 GB of quantized weights on top). Root cause is outside this repo; verified independent of eval ordering, chunking granularity, and reference lifetime.'
tags: [ltx25, gemma4, quantization, memory, moe, mlx-swift-lm]
timestamp: 2026-09-05T00:00:00Z
---

`quantization.textEncoder` (set by `--transformer-quant` on the CLI, same
value as the video transformer's) does not shrink the LTX-2.5 Gemma 4
encoder's resident memory. It makes it bigger. This was found while
investigating issue #86 (GPU timeout loading the transformer on a 36 GB
machine) — the transformer's own version of this class of bug is fixed on
`fix/quantized-load-per-block` (see
[quantized-load-materialised-full-bf16](quantized-load-materialised-full-bf16.md)
once that PR exists), but fixing it did not get a 36 GB machine under budget,
because the *text encoder* load — which runs before the transformer, in the
same `loadModels()` call — costs more on its own than expected.

# Measured

2.5-distilled, seed 42, 256×256, 9 frames, `LTXMemoryManager.logMemoryState`
checkpoints added around `Gemma4TextEncoder.load` (`Sources/LTXVideo/Models/TextEncoder/Gemma4/Gemma4TextEncoder.swift`):

| Step | `--transformer-quant bf16` | `--transformer-quant int4` |
| --- | --- | --- |
| after `Gemma4LLMModel` construction (no weights) | 0 MB | 0 MB |
| after applying bf16 weights (still lazy) | 0 MB | 0 MB |
| after evaluating the loaded bf16 weights | 22711 MB | 22711 MB |
| after `MLXNN.quantize(model:...)` is *called* (still lazy) | n/a | 22711 MB (unchanged) |
| after evaluating post-quantize | n/a | **29100 MB** |

int4 costs **more** than bf16, not less. The delta (29100 − 22711 = 6389 MB)
is close to what a genuinely-quantized ~22.7 GB bf16 model should cost at
4-bit (÷4 plus group-64 scale/zero overhead ≈ 6.2-6.4 GB) — consistent with
quantization *succeeding* at producing a smaller representation, while the
original bf16 weights simply never get released. Both copies coexist.

This was verified independent of every lever this repo controls:

- **Eval ordering**: materializing the bf16 weights before calling `quantize`
  (so it can't be blamed for forcing eager materialization) made no
  difference — the number after the subsequent eval was identical to calling
  `quantize` on the still-lazy weights directly.
- **Eval granularity**: replacing the single `eval(model.parameters())` with
  either (a) fixed-size chunks of the flattened parameter list, or (b) one
  `eval()` per decoder layer (via `Gemma4LLMModel.loraLayers`, the only public
  way to get per-layer granularity — `Gemma4TextModel.layers` itself is
  `internal` to `Gemma4Swift`) — changed nothing. Chunk (a) was tried first
  and ruled out for a specific reason: this checkpoint is MoE
  (`Gemma4Experts` / `SwitchGLU`, an upstream comment in `Gemma4Experts.swift`
  names the reference architecture "26B-A4B"), and a `SwitchLinear`'s weight
  is one `[numExperts, out, in]` tensor covering every expert at once — a
  count-based chunk boundary landing inside it still materializes the whole
  tensor regardless of the chunk size. Layer-granularity chunking (b) isolates
  each layer's own such tensor from every other layer's, and *still* made no
  difference to the final number — ruling out granularity as the axis
  entirely, not just the count-based version of it.
- **Reference lifetime**: dropping every Swift-side reference this repo holds
  to the source bf16 dictionary as early as possible (`LTX25TextEncoderAssets`
  parsed once and threaded through `LTXPipeline.loadModels()` /
  `loadTextEncoderModels()`, its projection weights extracted *before* the
  Gemma load instead of read a second time afterward at transformer-loading
  time; the per-call weights dict explicitly `.removeAll()`'d right after
  `Module.update(parameters:)` applies it) — made no difference either.

# Where this likely lives

Not in `ltx-video-swift-mlx`: none of this repo's own reference-lifetime or
eval-scheduling choices move the number, at all, in either direction — ruling
out this repo's calling code as the cause. Not conclusively `gemma-4-swift-mlx`
either: `Gemma4LLMModel`/`Gemma4TextModel` just declare the module tree: the
`SwitchLinear` type performing the actual quantized matmul (`SwitchGLU`'s
`gate_proj`/`up_proj`/`down_proj`) is defined in **`mlx-swift-lm`**
(`Libraries/MLXLMCommon/SwitchLayers.swift`) and conforms to `Quantizable`
there. `MLXNN.quantize(model:)` itself is `mlx-swift`. The likely candidates,
in rough order of suspicion, are:

1. `SwitchLinear`'s `Quantizable` conformance (in `mlx-swift-lm`) not actually
   dropping its original weight when replaced — quantizing correctly
   *computes* a smaller array but the pre-quantization `SwitchLinear` instance
   stays reachable from somewhere.
2. `MLXNN.quantize(model:)`'s submodule-replacement mechanism (in `mlx-swift`,
   `Source/MLXNN/Quantized.swift` and `Module.update(modules:)`) not fully
   detaching the module it replaces, in a way that only manifests for
   MoE/`SwitchLinear`-shaped submodules rather than plain `Linear`.
3. Some interaction with how this repo populates the model before quantizing
   — via `Module.update(parameters:)`, which this repo's own
   [Module.update mutates in place](module-update-mutates-in-place.md) pitfall
   already documents as *not* a clean array replacement for a different reason
   (LoRA fuse/unfuse snapshots silently aliasing the live state). That pitfall
   is about parameter arrays specifically, not submodule swaps, so it may be
   unrelated — flagged as a lead, not a confirmed mechanism.

None of these were fixed here: (1) and (2) are upstream dependencies this repo
pins by version/branch, not something to patch in place, and (3) needs
instrumentation inside `mlx-swift`/`mlx-swift-lm` to confirm, which is out of
this investigation's scope.

# Consequences

- Every LTX-2.5 generation that passes a non-`bf16` `--transformer-quant`
  currently pays a **larger** text-encoder memory cost than passing `bf16`,
  not a smaller one — the opposite of what the flag promises for the text
  encoder specifically (it still shrinks the video transformer correctly).
- On issue #86's class of machine (36 GB), this alone — before the
  transformer is even loaded — leaves very little headroom: ~29.1 GB active
  for Gemma plus whatever the transformer needs on top, per
  [quantized-load-materialised-full-bf16](quantized-load-materialised-full-bf16.md).
- Until this is root-caused upstream, quantizing the text encoder should not
  be assumed to help memory-constrained machines; `bf16` for `--transformer-quant`'s
  text-encoder effect is *smaller* on this specific checkpoint than any
  quantized level measured.

# Guarded by

Nothing yet — this needs an upstream fix (or a confirmed root cause) before a
regression test can pin the *correct* behavior. The measurement protocol
above (`LTXMemoryManager.logMemoryState` around `Gemma4TextEncoder.load`,
compared across `--transformer-quant bf16` vs `int4`) is the fastest way to
re-check after any `gemma-4-swift-mlx` / `mlx-swift-lm` / `mlx-swift` version
bump.
