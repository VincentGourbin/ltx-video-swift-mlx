---
okf_version: "0.1"
---

# Engineering Knowledge Base

Durable, measured engineering knowledge about this framework: benchmarks,
decisions with their rationale, verified pitfalls, investigation records and
diagnostic playbooks. Structured as an
[OKF](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md)
knowledge bundle — plain markdown with YAML frontmatter, readable by humans
and agents alike. This is the **knowledge layer**: what is true and why. The
user docs in [`docs/`](../) explain how to use the framework; concepts here
record what was measured, decided and root-caused so it is never re-derived
or re-litigated.

# Benchmarks

* [Generation baselines on M3 Max 96 GB](benchmarks/generation-baselines-m3max.md) - healthy wall-clock numbers measured July 2026; anything far above them is contention/thermal/misconfiguration, not the engine
* [LoRA training baselines on M3 Max 96 GB](benchmarks/lora-training-baselines-m3max.md) - wall-clock, peak memory and loss trajectories for the Disney overfit runs

# Decisions

* [Frame cap = 481, derived from the RoPE range](decisions/frame-cap-481-rope-range.md) - temporal RoPE coordinates are seconds normalized by maxPos[0] = 20 s; 481 frames = 20 s at 24 fps
* [Speech-window thresholds](decisions/speech-window-noise-floor.md) - absolute floor + credible noise-floor offset; peak-relative was tried and rejected
* [LipDub fusion reuse policy](decisions/lipdub-fusion-reuse-policy.md) - identity by canonical path + mtime, guards instead of unfuse
* [unloadAfterUse gates all mid-run unloads](decisions/unload-gating-semantics.md) - .disabled means keep everything; the trade-offs that buys
* [QLoRA is the training default on ≤96 GB](decisions/qlora-training-default.md) - qint8 halves peak memory with near-exact loss parity; bf16 swaps
* [LipDub segment continuation anchors on the tail latent](decisions/lipdub-continuation-anchor.md) - position 0 + overlap-and-trim; measured seam PSNR 17.4 → 24.6 dB

# Pitfalls

* [swift build binaries crash at MLX runtime](pitfalls/spm-binary-no-metallib.md) - the metallib only resolves from xcodebuild products
* [Release tests need ENABLE_TESTABILITY=YES](pitfalls/release-tests-need-testability.md) - plus a dedicated derived-data path and TEST_RUNNER_-prefixed env vars
* [LipDub fusion is destructive — double-delta trap](pitfalls/lora-refusion-double-delta.md) - five silent-corruption paths, all guarded since PR #36
* [384 block-norm weights are absent by design](pitfalls/affine-free-norms-expected-missing.md) - affine-free RMSNorms; and how the suppression list could mask a future mapping bug
* [Fake-stereo audio breaks the AudioVAE](pitfalls/audio-must-stay-stereo.md) - mono-downmixed references make mouths move in wrong directions
* [LipDub prompts must contain the dialogue](pitfalls/lipdub-prompt-needs-dialogue.md) - the trained prompt format, and the VLM-enhancement repair
* [Keyframes: append guide tokens, never slot-inject](pitfalls/keyframes-append-not-inject.md) - why slot overwrite is structurally wrong past frame 0
* [Connector GELU must be tanh-approximate](pitfalls/gelu-approximate-connector.md) - exact GELU surfaced as 94-98% sub-bass audio noise
* [VAE decoder D2S blocks: residual=false](pitfalls/decoder-d2s-residual-false.md) - the grid-artifact root cause, plus decoder facts worth not re-deriving

# Investigations

* [Cross-modal AdaLN sigma swap (May 2026)](investigations/crossmodal-adaln-sigma-swap-2026-05.md) - the LipDub mouth-modulation root cause, two expensively-refuted hypotheses, and the audio-anchored vs pose-anchored trade-off
* [LipDub segmentation campaign (July 2026)](investigations/lipdub-segmentation-asks-2026-07.md) - what unit tests, the in-process E2E, the code review and real reruns each caught that the others missed

# Playbooks

* [Diagnosing a lip-sync offset](playbooks/lipsync-offset-diagnosis.md) - the ordered checklist (prompt → channels → windows → fusion) before suspecting the model
