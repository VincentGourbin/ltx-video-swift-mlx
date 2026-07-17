---
type: Decision
title: LoRA training should run on a quantized frozen base (QLoRA) — bf16 barely fits 96 GB
description: qint8 halves peak memory (84.3 → 43.6 GB) with near-exact loss parity and ~20× the throughput of the swapping bf16 run; int4 works too with noisier loss.
tags: [training, lora, qlora, memory, quantization]
timestamp: 2026-07-16T00:00:00Z
---

Measured July 2026 on M3 Max 96 GB — distilled base, 3-clip dataset,
256×256×9, rank 16, lr 2e-4 cosine, seed 42, identical everything except
`--transformer-quant`:

| Base | Peak GPU | Pace | Loss (step 10, same seed) |
|---|---|---|---|
| bf16 | **84.3 GB** | ~110 s/step (**swapping**) | 0.5924 |
| qint8 | **43.6 GB** (−48%) | ~5 s/step | 0.5923 — near-exact parity |
| int4 | **37.5 GB** (−55%) | ~5 s/step | 0.7513 — noisier, still learns |

- The bf16 run is not "slow", it is **thrashing**: 84.3 GB of a 96 GB machine
  leaves nothing for the workspace, and every step pays swap. qint8 restored
  ~20× throughput (200 steps + load + on-the-fly quantization in 21 min).
- Gradients through the frozen `QuantizedLinear` base are correct — the
  step-10 loss matches bf16 to 4 decimals on the same seed.
- **qint8 IS the training default** (`LoRATrainingConfig`/`--transformer-quant`,
  since PR #38); bf16 remains available explicitly for >96 GB machines; int4
  for ~48 GB machines (accept the extra quantization noise). `nvfp4`/`mxfp8`
  are rejected at validate() — training always quantizes on the fly, which
  upstream doesn't support for those modes (mlx-swift #285).
- Related behavior notes pinned by the same PR: AdamW bias correction is
  always on (loss curves are NOT comparable to pre-July-2026 runs, even with
  `--lr-schedule constant`), and resume refuses checkpoints trained under a
  different LR schedule (pre-PR checkpoints = `constant`).
- Gradient checkpointing (remat) is NOT exposed by mlx-swift (verified
  0.31.6) — quantizing the frozen base is the memory lever, activations must
  be contained via resolution/frames presets.
- The load+cache phase peaks at ~40 GB regardless (Gemma + VAE encoding,
  both unloaded before the loop).

# Citations

[1] Runs recorded in [the training baselines](/docs/knowledge/benchmarks/lora-training-baselines-m3max.md);
dataset: Wild-Heart/Disney-VideoGeneration-Dataset (Apache-2.0).
