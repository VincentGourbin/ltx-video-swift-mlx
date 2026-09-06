---
type: Decision
title: Bumping mlx-swift to main was rejected — it changes output on hardware the NAX fix doesn't even touch
description: 'Investigated to pick up mlx#3810 (the NAX split-K GEMM fix). Measured ~20-28% relative divergence in final generation output on an M3 Max — hardware the underlying bug cannot reach at all. Shipped a hardware-gated workaround instead; the bump itself is parked, not merged.'
tags: [mlx-swift, dependency-bump, m5, nax, regression]
timestamp: 2026-09-06T00:00:00Z
---

Issue #69 needed [mlx#3810](https://github.com/ml-explore/mlx/pull/3810) (the
NAX split-K GEMM dtype fix — see
[nax-splitk-gemm-m5-black-video](../pitfalls/nax-splitk-gemm-m5-black-video.md)).
No mlx-swift tag vendors it; only `main` (`ab924c82`, "update for mlx
v0.32.2", 2026-09-01) does. Bumping was tried on
`chore/mlx-swift-main-nax-fix` and rejected.

# The measurement

Same seed, same weights, same LTX code — only the mlx-swift pin changed.
2.5-distilled, seed 42, 256×256, 9 frames, plain bf16 (no `--transformer-quant`),
measured on an **M3 Max (gen-15)** — well under the gen-17 threshold
`is_nax_available()` requires, so this hardware cannot reach the NAX bug at
all, in either direction:

| | mlx-swift `0.31.6` | mlx-swift `main` (`ab924c82`) |
| --- | --- | --- |
| Text-encoder connector output | `mean=0.013376623` | `mean=0.013375904` (~0.005% relative — ordinary bf16 rounding) |
| Final VAE decode output | `mean=-0.23723084` | `mean=-0.16997448` (**~28% relative**) |

Text encoding survives the bump essentially untouched; the divergence enters
somewhere in the denoising loop and compounds. This is not the NAX bug
resurfacing under a different name — it's a separate, apparently-unrelated
numerical shift somewhere in the ~500 mlx commits the bump carries between
`0.31.6` and `main`.

# Why the exact commit wasn't pinned down

Bisecting the 13 mlx-swift-level commits between `0.31.6` and `main`
correctly narrowed the divergence to the single vendor-sync commit
`ab924c82` itself — every other commit (a deadlock fix, a `Stream`-argument
fix, CI/build changes) reproduced the pre-bump output exactly.

Going further — into the ~500 underlying `mlx` commits that one vendor-sync
commit carries — hit a structural wall, not just a slow one:

- The files actually compiled (`Source/Cmlx/mlx-generated/*`) are committed
  directly into mlx-swift's own repo, not regenerated from the `mlx`
  submodule at SPM build time. Swapping the submodule's checked-out commit
  alone silently compiles the same fixed kernel set regardless — confirmed
  by every test in that first attempt returning byte-identical output,
  including the one that should have been the pre-bump baseline.
- Redone properly with mlx-swift's own `update-mlx.sh` (which does
  regenerate those files from the submodule via `cmake`) at each bisection
  point, `Source/Cmlx/mlx-c` — a **second** submodule, pinned in lockstep
  with `mlx` by mlx-swift's own maintainers for this exact bump — turned out
  to need a very recent `mlx` snapshot just to compile. Going back more than
  a handful of commits broke the build on missing/changed mlx-c APIs
  (`mlx::core::detail::CompileCacheWeakPtr` not existing yet,
  `mlx::core::fast::cross_entropy` missing, `scaled_dot_product_attention`'s
  signature differing) — each step back surfaced a *new* incompatibility,
  suggesting mlx-c's pin sits very close to `main` itself and leaves almost
  no independently-testable range for `mlx` alone.

Pinning the exact commit would require bisecting both submodules together —
finding a compatible `mlx`/`mlx-c` pair at each candidate point, effectively
redoing mlx-swift's own historical update procedure several times over. Not
attempted; flagged as the next step if someone picks this back up.

# Decision

Shipped
[a hardware-gated workaround](../pitfalls/nax-splitk-gemm-m5-black-video.md)
instead of the bump: it fixes the one confirmed-affected op
(the connector's feed-forward down-projection) only on hardware the NAX bug
can actually reach, leaving every other machine's code path — and this
divergence — untouched. The bump itself stays parked on
`chore/mlx-swift-main-nax-fix` (uncommitted `Package.swift`/`Package.resolved`
changes only, nothing merged) until either a tagged mlx-swift release
carries the NAX fix without this side effect, or someone re-bisects with
both submodules varying.

# Revisit when

- A new mlx-swift tag ships that's closer to `0.31.6` than to `main` and
  carries mlx#3810 — check whether the divergence is still present at that
  tag before assuming it's fixed.
- Someone has time to bisect `mlx` and `mlx-c` together across the full
  range; the per-cycle cost once `update-mlx.sh` is wired in is small
  (~4-5 minutes per candidate on this hardware) — the blocker was the
  submodule pairing, not raw build time.
