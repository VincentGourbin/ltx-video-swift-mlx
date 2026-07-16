---
type: Investigation
title: LipDub app-integration campaign — asks, E2E-caught blockers, review findings (July 2026)
description: How four app asks (PR #36) turned into two E2E-caught bugs and ten review findings — and what each validation layer caught that the others missed.
tags: [lipdub, segmentation, e2e, code-review, pr36]
timestamp: 2026-07-16T00:00:00Z
---

Fluxforge Studio integrated LipDub for dubbing and filed measured asks
(`FRAMEWORK_ASKS_VOICE_LIPDUB.md`): raise the frame cap (B1), fix
speech-window detection on enrolled voices (B2), explain 384 missing norms
(B4), stop reloading the 22B per segment (B6). PR #36 resolved all four —
see the [frame-cap](/docs/knowledge/decisions/frame-cap-481-rope-range.md),
[speech-window](/docs/knowledge/decisions/speech-window-noise-floor.md),
[fusion-reuse](/docs/knowledge/decisions/lipdub-fusion-reuse-policy.md) and
[unload-gating](/docs/knowledge/decisions/unload-gating-semantics.md)
decisions. The meta-lesson is which validation layer caught what:

# What each layer caught

- **Unit tests** caught nothing the implementation didn't already know —
  they pinned the intended behavior (thresholds, bounds).
- **The gated in-process E2E** (`LipDubReuseE2ETests`) caught, on its first
  real execution, two blockers invisible to both unit tests and one-shot CLI
  runs, because only it runs two generations in one process:
  1. Gemma/tokenizer unloaded unconditionally after text encoding → segment 2
     died with "Tokenizer not loaded" before reaching the reuse path.
  2. MLX workspace buffers accumulating across segments (nothing clears the
     cache with `.disabled`).
- **The 8-angle code review** then found 10 findings the E2E could NOT see,
  because they live on paths the test doesn't drive: `exportQuantizedTransformer`
  persisting fused weights to disk, the `fuseLoRA()`/LipDub state machines
  corrupting each other, the retake dev-path reloading a resident stack
  mid-run, and a regression in the new threshold logic on clean audio
  (tightly-trimmed clips) that the enrolled-voice test could not expose.
- **Real reruns after the review fixes** confirmed byte-identical speech
  windows on the enrolled case and a green E2E — the fixes changed exactly
  the intended behavior and nothing else.

# Unresolved

Run 2 of the E2E is consistently ~2× run 1 (67→131 s, 234→522 s in separate
sessions) even after the cache fix, with identical work. Working hypothesis:
thermal throttling on back-to-back 22B runs. Not blocking (the reuse win is
the avoided reload), but measure in-app before quoting per-segment gains.

# Reusable artifacts

- [Validation protocol with PASS/FAIL criteria](/docs/testing/PR36-validation-protocol.md)
- `LipDubReuseE2ETests` — see
  [the Release-tests pitfall](/docs/knowledge/pitfalls/release-tests-need-testability.md)
  for how to run it.
- Segment-continuation design for image mode: issue #35.
