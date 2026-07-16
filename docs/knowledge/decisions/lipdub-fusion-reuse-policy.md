---
type: Decision
title: LipDub fusion reuse — identity by canonical path + mtime, guards instead of unfuse
description: Pristine originals of the 22B are too big to keep, so reuse is allowed only for the same unchanged file, and every other weight-touching path throws while fused.
tags: [lora, lipdub, fusion, segmentation, memory]
timestamp: 2026-07-16T00:00:00Z
---

App-side segmentation of long dialogues runs N consecutive `generateLipDub`
calls with the same IC-LoRA. Reloading the 22B per segment costs minutes;
keeping pristine originals for a clean unfuse costs ~10+ GB. The chosen
policy:

- **No originals are kept** for the LipDub fusion (unlike the generic
  `fuseLoRA()`/`unfuseLoRA()` pair, which keeps them).
- **Reuse is keyed on `LipDubFusionRecord`**: canonical path (symlinks
  resolved, standardized — different spellings of the same file must not
  force a reload) **plus the file's mtime at fusion time** (a file
  overwritten in place must not be silently reused). The mtime is recorded
  *before* fusing so an overwrite racing the fusion reads as "changed".
- **Everything else throws while fused** rather than corrupting silently —
  see [the double-delta pitfall](/docs/knowledge/pitfalls/lora-refusion-double-delta.md).
- **Precondition**: the transformer must survive between runs —
  `MemoryOptimizationConfig.disabled` (`unloadAfterUse == false`). Host apps
  check `pipeline.fusedLipDubLoRAPath` to decide reuse vs reload.

# Measured

The gated E2E (`LipDubReuseE2ETests`, `LTX_E2E_LIPDUB=1`) drives two
consecutive runs in-process and asserts the full state machine. It caught two
real blockers on first execution (unconditional Gemma unload; MLX cache
accumulation) — see
[the investigation](/docs/knowledge/investigations/lipdub-segmentation-asks-2026-07.md).
Run 2 consistently measures ~2× run 1 wall-clock (67→131 s and 234→522 s in
two independent sessions) even after the cache fix — thermal throttling on
back-to-back runs is the working hypothesis; the reuse win is the avoided
reload+refusion, to be measured in-app.

# Citations

[1] Fluxforge asks item B6, resolved in PR #36.
