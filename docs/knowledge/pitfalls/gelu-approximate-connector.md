---
type: Pitfall
title: The connector must use tanh-approximate GELU, not exact GELU
description: MLXNN.gelu() in the connector produced wrong text embeddings whose downstream symptom was 94-98% sub-bass audio noise — nowhere near the text encoder.
tags: [gelu, connector, text-encoder, root-cause]
timestamp: 2026-07-16T00:00:00Z
---

Python declares `activation_fn="gelu-approximate"` for the connector's
`GELUProjection`; the Swift port initially used `MLXNN.gelu()` (exact). The
numerical difference is tiny per-activation but the downstream symptom was
spectacular and misleading: **94-98% sub-bass noise in generated audio** —
debugged for a long time in the audio stack before being traced back to text
embeddings (March 2026).

# The defense

- `GELUProjection` in `LTXTextEncoder.swift` uses `geluApproximate()` (tanh).
  The main transformer FFN (SwiGLU) does too.
- The general lesson: when porting from the Python reference, activation
  variants (`gelu` vs `gelu-approximate`), norm placement, and dtype paths are
  bit-for-bit contracts. A "close enough" activation can surface as a garbage
  symptom in a completely different subsystem after 48 blocks × 10 steps of
  accumulation.

# Citations

[1] Root-caused March 2026 during the audio pipeline bring-up.
