---
type: Pitfall
title: 384 block-norm weights are absent from the checkpoint by design
description: norm1/2/3, audio_norm1/2/3 and the cross-modal norms are affine-free RMSNorms; reporting them as "missing" is noise, but the suppression list can mask a real mapping bug.
tags: [weights, rmsnorm, checkpoint, adaln]
timestamp: 2026-07-16T00:00:00Z
---

The official `ltx-2.3-22b-distilled.safetensors` ships **zero** weights for
`transformer_blocks.*.{norm1,norm2,norm3,audio_norm1,audio_norm2,audio_norm3,
audio_to_video_norm,video_to_audio_norm}` — verified by reading the
safetensors header (86 keys per block, only attention `q_norm`/`k_norm` carry
norm weights). These RMSNorms are affine-free: their scale/shift comes from
the AdaLN `scale_shift_table`, so MLXNN's default `weight = 1` is exactly
correct. 8 norms × 48 blocks = the 384 "missing" keys that used to pollute
every load report.

# The defense

`applyTransformerWeights` filters these suffixes from the missing report
(`affineFreeNormSuffixes` in `ModelDownloader.swift`) so REAL mapping holes
stand out again.

**The trade-off to remember**: if a future checkpoint variant ever ships
affine weights for these norms and a key-mapping bug keeps them from loading,
the suppression list will label the failure "expected". When bumping to a new
checkpoint generation, re-check the header for these keys before trusting the
filter. The deeper fix (an affine-free RMSNorm module that simply declares no
weight parameter) was considered and deferred.

# Citations

[1] Fluxforge asks item B4, resolved in PR #36 — see
[the investigation](/docs/knowledge/investigations/lipdub-segmentation-asks-2026-07.md).
