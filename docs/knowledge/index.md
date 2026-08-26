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
* [LTX-2.5 against MiniMax-H3 on a 10 s case](benchmarks/ltx25-vs-h3-starship-2026-08.md) - same machine and seed: phase table, per-step cost, and the 2.8× thermal spread that bounds any cost claim

# Decisions

* [Frame cap = 481, derived from the RoPE range](decisions/frame-cap-481-rope-range.md) - temporal RoPE coordinates are seconds normalized by maxPos[0] = 20 s; 481 frames = 20 s at 24 fps
* [Speech-window thresholds](decisions/speech-window-noise-floor.md) - absolute floor + credible noise-floor offset; peak-relative was tried and rejected
* [LipDub fusion reuse policy](decisions/lipdub-fusion-reuse-policy.md) - identity by canonical path + mtime, guards instead of unfuse
* [unloadAfterUse gates all mid-run unloads](decisions/unload-gating-semantics.md) - .disabled means keep everything; the trade-offs that buys
* [QLoRA is the training default on ≤96 GB](decisions/qlora-training-default.md) - qint8 halves peak memory with near-exact loss parity; bf16 swaps
* [Generated keyframe slots are appended, denoised and marked](decisions/generated-keyframe-slots.md) - DFR's own anchors: denoised with the video, marked, one pixel frame of RoPE span; and why the 2.5 detailing LoRA needed no work
* [IC-LoRA stage 2 keeps adapter and reference](decisions/iclora-stage2-keeps-adapter-and-reference.md) - measured 7-run matrix: identity survives the inter-stage renoise only with both active; deliberate divergence from ic_lora.py
* [LipDub segment continuation anchors on the tail latent](decisions/lipdub-continuation-anchor.md) - position 0 + overlap-and-trim; measured seam PSNR 17.4 → 24.6 dB
* [The LTX-2.5 prompt enhancer is a second Gemma, and the caller may supply it](decisions/prompt-enhancer-source.md) - encode-only bundled encoder forces a separate E2B-it; bf16 default (10.24 GB), 6-bit and a caller-supplied root opt-in

# Pitfalls

* [swift build binaries crash at MLX runtime](pitfalls/spm-binary-no-metallib.md) - the metallib only resolves from xcodebuild products
* [Release tests need ENABLE_TESTABILITY=YES](pitfalls/release-tests-need-testability.md) - plus a dedicated derived-data path and TEST_RUNNER_-prefixed env vars
* [LipDub fusion is destructive — double-delta trap](pitfalls/lora-refusion-double-delta.md) - five silent-corruption paths, all guarded since PR #36; superseded in part by the in-place-update pitfall below
* [Module.update mutates in place — snapshots lie](pitfalls/module-update-mutates-in-place.md) - unfuseLoRA never restored anything: captured originals aliased the fused values; copy at capture, materialise before updating
* [384 block-norm weights are absent by design](pitfalls/affine-free-norms-expected-missing.md) - affine-free RMSNorms; and how the suppression list could mask a future mapping bug
* [Fake-stereo audio breaks the AudioVAE](pitfalls/audio-must-stay-stereo.md) - mono-downmixed references make mouths move in wrong directions
* [LipDub prompts must contain the dialogue](pitfalls/lipdub-prompt-needs-dialogue.md) - the trained prompt format, and the VLM-enhancement repair
* [Keyframes: append guide tokens, never slot-inject](pitfalls/keyframes-append-not-inject.md) - why slot overwrite is structurally wrong past frame 0
* [Connector GELU must be tanh-approximate](pitfalls/gelu-approximate-connector.md) - exact GELU surfaced as 94-98% sub-bass audio noise
* [VAE decoder D2S blocks: residual=false](pitfalls/decoder-d2s-residual-false.md) - the grid-artifact root cause, plus decoder facts worth not re-deriving
* [The LipDub audio contract](pitfalls/lipdub-audio-contract.md) - ship the generated track, feed a clean reference; six rules with the measurements behind them
* [LipDub segments cap at ~233 frames](pitfalls/lipdub-segment-bound-233.md) - the negative-position audio reference doubles the RoPE span; 481 is only for generate/retake
* [Split-checkpoint lookups fail silently](pitfalls/split-checkpoint-silent-empty-load.md) - a wrong-file prefix returns zero keys, not an error; the VAE encoder ran randomly initialised through a whole generation
* [LipDub overwrites anything crossing the mouth](pitfalls/lipdub-overwrites-objects-crossing-the-mouth.md) - no occlusion modelling: a headset band or hand in front of the lips is painted over; visible in the repo's own teaser
* [CFG against an empty negative erases the prompt](pitfalls/empty-cfg-negative-erases-the-prompt.md) - the dev paths inherited "" from the MLX port; one A/B apart, the official negative restored a 14-second choreography
* [no_repeat_ngram bans quoting the prompt](pitfalls/ngram-blocking-mangles-prompt-quoting.md) - enhancer timestamps mangled, duration over-predicted ~5 s; reference-space limitation, fix pending in gemma-4-swift-mlx
* [Smoothness metrics miss identity drift](pitfalls/smoothness-metrics-miss-identity-drift.md) - a seam where each tile drifted gradually to a different subject reads as perfectly smooth
* [Cross-attention's q_norm is not the block's pre-norm](pitfalls/cross-attention-prenorm.md) - found by the transformer parity harness; the legacy block skipped the RMS norm before cross-attention
* [A densified clip must be positioned at its new rate](pitfalls/densified-clip-keeps-its-rate.md) - a temporal round doubles frames and fps; keeping the source's fps reads as twice the duration
* [A renoise level needs its anchoring](pitfalls/renoise-level-needs-its-anchor.md) - upstream's sigma 0.975 redraws the subject without the keyframe seams that make it viable there
* [Tiled-attention mask caches need the whole window pattern](pitfalls/na-tile-mask-cache-key.md) - border and interior tiles collide on a summary key; 8% error in one stage, invisible without a reference
* [Dotted parameter names never load](pitfalls/dotted-parameter-names-never-load.md) - unflattened() reads "." as a module boundary; the update lands nowhere and strict key checks miss it
* [The wrong vocoder cost the top octave](pitfalls/wrong-vocoder-lost-the-top-octave.md) - LTX-2's vocoder decoded 2.3/2.5 latents plausibly; +18 dB at 12-16 kHz once corrected, nothing below 8 kHz — and how a two-variable A/B first got this badly wrong
* [Don't validate the LTX Gemma by generating text](pitfalls/ltx-gemma-head-is-vestigial.md) - its tied head is saturated by design; check parameter coverage, scale band and meaning instead
* [The continuation-tail clip must be re-encoded](pitfalls/continuation-tail-clip-encoding.md) - an input seek leaves frame 0 off t=0 and the zero-tolerance extractor refuses it
* [URLSession's per-task delegate never reports download progress](pitfalls/urlsession-task-delegate-has-no-download-progress.md) - download(for:delegate:) calls didWriteData zero times; needs a session delegate on an explicit downloadTask, which then puts error pages on disk
* [The duration head ignores durations written in the prompt](pitfalls/duration-head-does-not-read-written-durations.md) - a "3 seconds." prefix returns byte-identical output to no prefix; it regresses from connector tokens and never sees the text

# Investigations

* [The enhancer's residual defects are the reference's](investigations/enhancer-residual-defects-2026-08.md) - measured against the Space's own inference across four prompts
* [What LTX-2.5 actually changes (August 2026)](investigations/ltx-2.5-checkpoint-diff-2026-08.md) - tensor-level diff against 2.3: the DiT moves by two flags, the VAEs and upscaler are unchanged, the cost is the Gemma 4 encoder
* [Cross-modal AdaLN sigma swap (May 2026)](investigations/crossmodal-adaln-sigma-swap-2026-05.md) - the LipDub mouth-modulation root cause, two expensively-refuted hypotheses, and the audio-anchored vs pose-anchored trade-off
* [LipDub segmentation campaign (July 2026)](investigations/lipdub-segmentation-asks-2026-07.md) - what unit tests, the in-process E2E, the code review and real reruns each caught that the others missed
* [Custom-voice timbre chain (July 2026)](investigations/custom-voice-timbre-chain-2026-07.md) - attributing a bad custom-voice LipDub across Voxtral enrollment and LTX; the decoder was innocent

# Playbooks

* [Diagnosing a lip-sync offset](playbooks/lipsync-offset-diagnosis.md) - the ordered checklist (transcript → segment length → prompt → channels → windows → fusion → timbre) before suspecting the model
