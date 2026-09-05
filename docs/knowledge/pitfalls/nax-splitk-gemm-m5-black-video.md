---
type: Pitfall
title: The text connector goes NaN on M5/macOS 26.2+ — an mlx NAX split-K GEMM bug, worked around, not bumped
description: 'mlx#3797: the NAX split-K GEMM templates on the accumulator''s hardcoded float32 dtype instead of the input''s, misreading bf16 buffers on gen-17+ GPUs. Fixed upstream (mlx#3810) but no mlx-swift tag vendors it, and bumping to main changes generation output broadly (see quantized-load pitfalls). Worked around with a hardware-gated float32 cast instead.'
tags: [ltx25, m5, mlx-bug, connector, nan, black-video]
timestamp: 2026-09-06T00:00:00Z
---

Issue #69 / PR #85: on Apple M5 (macOS 27), the LTX text connector produced
`mean=0.0, std=0.0` then `NaN`, ending in an all-black ~2 KB video. Every
resolution and step count reproduced it; only M5-class hardware did.

# Root cause

[ml-explore/mlx#3797](https://github.com/ml-explore/mlx/issues/3797). The
Metal NAX split-K GEMM kernel dispatches when `batch_size_out == 1`,
`M·N ≥ 2048²`, `K ≥ 10240`, `K ≥ 3·max(M,N)`, and the GPU reports
`is_nax_available()` (gen-17+ silicon, macOS ≥ 26.2 — `MLX_ENABLE_TF32`
defaults to `1`, so a bf16 or float32 input dispatches identically; TF32
doesn't gate this).

Once dispatched, `get_steel_gemm_splitk_nax_kernel`
(`mlx/backend/metal/jit_kernels.cpp`) templates the kernel on
`get_type_string(out.dtype())` — but `out` here is the split-K partial-sum
accumulator (`C_split`, `mlx/backend/metal/matmul.cpp`), which is
**hardcoded to `float32`** regardless of the actual input dtype. The template
should read the *input*'s dtype instead. For a bf16 input, the kernel is
compiled expecting float32-laid-out data but fed 2-byte bf16 values — it
misreads the buffer. A float32 input happens to already match the hardcoded
accumulator dtype, so it sidesteps the mismatch entirely — not because it
avoids the dispatch (it doesn't).

The LTX connector's feed-forward down-projection —
`[1024 tokens, 16384] @ [16384, 4096]` (the *down* projection, `ConnectorFeedForward.projectOut`;
its *up* projection, `projectIn`, is the differently-shaped `[1024, 4096] @ [4096, 16384]`
and is not the one that hits this) — sits exactly on the `M·N = 2048²`
boundary (concretely `1024 tokens · 4096 out = 4194304 = 2048²`), so it's the first thing to hit this on every LTX-2.5
generation on affected hardware. `TextEncoderConfig.default`'s connector
geometry (32 heads × 128 = 4096, 8 layers) is identical between LTX-2.3 and
LTX-2.5, so this is not 2.5-specific by shape — if 2.3 hasn't been reported
affected, that's unconfirmed, not shape-immune. The video transformer's own
feed-forward down-projection (`N=4096, K=16384`) has the same shape class and
is reachable at ≥ 1024 video tokens in bf16 on the (unbatched) dev path — not
confirmed broken, but not confirmed safe either.

Fixed upstream by [mlx#3810](https://github.com/ml-explore/mlx/pull/3810)
(2026-07-07) — the one-line fix reads `in.dtype()` instead of `out.dtype()`.

# Why this repo didn't just bump mlx-swift

No mlx-swift tag vendors the fix yet; only `main`
(`ab924c82`, "update for mlx v0.32.2", 2026-09-01) does. Bumping to `main`
changes final generation output substantially — ~20-28% relative on a plain
bf16 generation — on hardware this NAX bug **cannot** reach at all (measured
on an M3 Max, gen-15, well under the gen-17 threshold). Text-encoder output
was ~identical before/after the bump (<0.2% relative, ordinary bf16 rounding
noise); the divergence enters somewhere in the denoising loop. Bisecting the
~13 mlx-swift-level commits between `0.31.6` and `main` narrowed it to that
one vendor-sync commit as a whole; bisecting *inside* it (the ~500 underlying
mlx commits it carries) hit a structural wall — `mlx-c` (a second submodule,
pinned in lockstep with `mlx` by mlx-swift's own maintainers) needs a very
recent `mlx` snapshot to even compile, leaving almost no independently
testable range without bisecting both submodules together. Not pursued
further; see `docs/knowledge/log.md` (2026-09-05/06 entries) for the detailed
trail if someone picks this back up.

Given that, shipping the bump now would fix M5 at the cost of a silent,
unvalidated numerical change for every other machine. Not an acceptable
trade for a workaround that has a narrower, verifiable alternative.

# The workaround

`MLXNAXSplitKWorkaround` (`Sources/LTXVideo/Utils/MLXNAXSplitKWorkaround.swift`)
replicates `is_nax_available()`'s own check — same input
(`MTLDevice.architecture.name`, already public via `MLX.GPU.deviceInfo()`),
same parsing (two digits before the architecture string's last character are
the GPU generation; that last character is `'p'` for phone-class silicon,
threshold 18, anything else threshold 17) — and gates a float32 cast on the
connector's input (`Embeddings1DConnector.callAsFunction`,
`LTXTextEncoder.swift`) behind it. `LTX_NAX_WORKAROUND=1`/`0` env var forces
it on/off, for testing or in case detection ever misses a future GPU string.

Verified on an M3 Max (gen-15, unaffected):
- Gate correctly evaluates `affected=false`; connector and final `VAE raw
  output` are byte-identical to the unpatched code path (mean, min, max all
  match to the printed precision) — zero behavior change on hardware this
  bug can't reach.
- Forcing the workaround on (`LTX_NAX_WORKAROUND=1`) produces finite,
  reasonable output close to the bf16 baseline (~0.02% relative on the
  connector, small propagated drift by VAE decode) — the float32 code path
  itself is sound, even though this machine can't exercise the actual bug
  (the NAX kernel never dispatches here regardless of input dtype).
- Not independently verified against the actual M5 NaN, since this repo
  doesn't have M5 hardware — the cast technique itself is exactly what
  PR #85's author validated end-to-end on real M5 hardware (finite connector
  stages, successful 512×512 33-frame generation); this fix only narrows
  *when* it applies.

# What this does not cover

The video transformer's own feed-forward down-projection is the same shape
class and reachable at ≥ 1024 video tokens in bf16 — not patched here (PR
#85's author's own 512×512/33-frame/1280-token validation run should have
hit this dispatch condition too, per the shape math, yet reported success;
either that GEMM is unaffected for a reason not visible from the shape
analysis alone, or the corruption didn't show up in a luminance check).
Flagged, not fixed, pending a confirmed report.

# Guarded by

No automated test exercises the actual M5 kernel bug (this repo has no M5
hardware). `swift test` confirms the workaround doesn't change behavior on
non-affected hardware (`MLXNAXSplitKWorkaround.isAffectedHardware == false`
on every CI/dev machine so far) and doesn't crash when forced on.
