---
type: Decision
title: The LTX-2.5 prompt enhancer is a second Gemma, and the caller may supply it
description: Enhancement needs a generative E2B-it separate from the encode-only bundled encoder. bf16 stays the default (10.2 GB, reference parity); 6bit (4.7 GB) and a caller-supplied root are opt-in.
tags: [ltx25, enhancer, gemma4, downloads, disk]
timestamp: 2026-08-26T00:00:00Z
---

`--enhance-prompt` on LTX-2.5 loads a **second** Gemma 4, on top of the 26 GB
encoder inside the checkpoint. That is not redundancy to be optimized away: the
bundled `gemma4_unified` is declared encode-only upstream and its LM head is
measurably vestigial (see the
[enhancer investigation](/docs/knowledge/investigations/enhancer-residual-defects-2026-08.md)),
so it cannot rewrite its own prompt. Upstream solves it the same way, with
`--prompt-enhancer-gemma-root` pointing at a Gemma 4 E2B-it.

``PromptEnhancerSource`` therefore offers three ways to pay for that model:

| Source | On disk | Why |
| --- | --- | --- |
| `.managed(.bf16)` — **default** | 10.24 GB | What the reference space runs |
| `.managed(.sixBit)` | 4.74 GB | 5.5 GB less; quality unmeasured |
| `.localRoot(path)` | 0 | Reuse weights the host app already ships |

Measured from the HuggingFace blob sizes, August 2026.

# Why bf16 stays the default

A quantized default would be a silent quality change for every existing caller,
and the evidence does not support it: 4-bit E2B was tried and degraded
instruction following. 6-bit sits between that and the reference and **nobody
has run it against the enhancer bench**. It is offered, documented as
unmeasured, and not chosen for anyone.

The bench to settle it with is `docs/examples/ltx-2.5/enhancer-bench` — same
prompt/image/seed, one variable at a time, as the reasoning/n-gram policy was
settled.

# Why a caller-supplied root needs no precision flag

`mlx-community/gemma-4-e2b-it-6bit` carries a standard MLX quantization block:

```json
"quantization": {"group_size": 64, "bits": 6, "mode": "affine"}
```

`MLXLMCommon`'s loader reads that and quantizes during load, so a caller's
checkpoint of any precision goes through the identical code path as bf16. The
CLI refuses `--prompt-enhancer-root` together with
`--prompt-enhancer-precision` rather than silently ignoring one.

# Caveats

- **The file list must not be hardcoded.** bf16 ships three shards, 6-bit ships
  one. The download enumerates the repo through the HuggingFace API and records
  what it wrote in a `.ltx-enhancer-manifest.json`, written *last* so a manifest
  can never certify a partial download.
- The `bf16` cache directory keeps its pre-existing name
  (`enhancer-gemma4-e2b-bf16`), so installs made before precisions existed are
  not re-downloaded.
- These are LTX-2.3 no-ops: 2.3's Gemma 3 has a real LM head and self-enhances
  through the shared VLM.
