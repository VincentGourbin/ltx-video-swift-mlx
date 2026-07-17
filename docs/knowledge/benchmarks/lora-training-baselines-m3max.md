---
type: Benchmark
title: LoRA training baselines on M3 Max 96 GB (July 2026)
description: Wall-clock, peak memory and loss trajectories for the 3-clip Disney overfit runs — the sanity anchors for any future training-perf claim.
tags: [training, lora, benchmarks, m3-max]
timestamp: 2026-07-16T00:00:00Z
---

Setup: `ltx-video train`, distilled base, dataset = 3 clips of
Wild-Heart/Disney-VideoGeneration-Dataset (Apache-2.0) at 256×256×9,
rank 16, alpha 16, lr 2e-4, warmup 20, cosine schedule, seed 42,
AdamW bias-correction on. Release binary, M3 Max 96 GB.

# Wall-clock and memory

| Run | Steps | Total time (load+cache+loop) | Peak GPU |
|---|---|---|---|
| qint8 | 200 | **1279 s (~21 min)** | 43.6 GB |
| int4 | 100 | 617 s (~10 min) | 37.5 GB |
| bf16 | killed at ~30 | ~110 s/step observed | 84.3 GB (swapping) |

Full style run (69 clips, dev base, qint8, rank 32, 49 frames):

| Config | Tokens/sample | Outcome | Peak GPU |
|---|---|---|---|
| 448×256×49 | 784 (7×8×14) | **jetsam-killed at step ~40** | 95.4 GB |
| **320×192×49** | 420 (7×6×10) | 1500 steps in **17097 s (~4 h 45)** | **61.2 GB, flat** |

# Activation-memory sizing rule (no remat in mlx-swift)

Peak memory ≈ quantized weights (~24 GB dev qint8) + a per-token activation
cost across the 48 blocks. Measured points: 128 tokens → 43.6 GB,
420 tokens → 61.2 GB, 784 tokens → >95 GB (death). Roughly **~60 MB per
training token** on top of the base — size `latentFrames × H/32 × W/32`
against the machine's RAM *before* launching, and keep ≥30 GB of headroom
(the 448×256×49 run died with the GPU watchdog also killing the NEXT launch
until the system settled).

Fixed overhead per run: model load + on-the-fly quantization + latent-cache
build ≈ 5-8 min; cache-build phase alone peaks at ~40 GB (Gemma + VAE
encoder, unloaded before the loop). LoRA injection: 384 layers (48 blocks).

# Loss trajectory (qint8, 200 steps, smoothed by 50-step buckets)

`0.686 → 0.673 → 0.611 → 0.564`, best single step 0.407. Flow-matching loss
never approaches 0 (irreducible floor from sigma sampling) — **the monotone
decline is the health signal**, not the absolute value. int4 follows the
same shape ~0.05-0.15 higher.

# Reading these numbers later

- A training run whose smoothed loss does NOT decline over 200 steps, or
  whose peak memory far exceeds the table at the same config, is broken —
  don't tune hyper-parameters first, find the regression.
- See [the QLoRA decision](/docs/knowledge/decisions/qlora-training-default.md)
  for why bf16 is not a usable baseline on 96 GB.
