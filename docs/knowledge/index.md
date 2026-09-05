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
* [Vectorized RGBA pixel conversion vs. the scalar loop it replaced](benchmarks/pixel-conversion-vectorization-2026-08.md) - loadVideo on a 121-frame clip: ~8.15s to ~1.0s (~8x); the remaining time is AVFoundation decode, not conversion

# Decisions

* [Frame cap = 481, derived from the RoPE range](decisions/frame-cap-481-rope-range.md) - temporal RoPE coordinates are seconds normalized by maxPos[0] = 20 s; 481 frames = 20 s at 24 fps
* [Speech-window thresholds](decisions/speech-window-noise-floor.md) - absolute floor + credible noise-floor offset; peak-relative was tried and rejected
* [LipDub fusion reuse policy](decisions/lipdub-fusion-reuse-policy.md) - identity by canonical path + mtime, guards instead of unfuse
* [Mid-run unloads are gated per component](decisions/unload-gating-semantics.md) - unloadAfterUse is the default, not the whole story: LTX-2.5's 26 GB encoder made one flag for encoder and transformer unaffordable
* [QLoRA is the training default on ≤96 GB](decisions/qlora-training-default.md) - qint8 halves peak memory with near-exact loss parity; bf16 swaps
* [Generated keyframe slots are appended, denoised and marked](decisions/generated-keyframe-slots.md) - DFR's own anchors: denoised with the video, marked, one pixel frame of RoPE span; and why the 2.5 detailing LoRA needed no work
* [IC-LoRA stage 2 keeps adapter and reference](decisions/iclora-stage2-keeps-adapter-and-reference.md) - measured 7-run matrix: identity survives the inter-stage renoise only with both active; deliberate divergence from ic_lora.py
* [LipDub segment continuation anchors on the tail latent](decisions/lipdub-continuation-anchor.md) - position 0 + overlap-and-trim; measured seam PSNR 17.4 → 24.6 dB
* [The LTX-2.5 prompt enhancer is a second Gemma, and the caller may supply it](decisions/prompt-enhancer-source.md) - encode-only bundled encoder forces a separate E2B-it; bf16 default (10.24 GB), 6-bit and a caller-supplied root opt-in
* [A retake picks its stream with a modality](decisions/retake-modality-frozen-stream.md) - freezing is a σ = 0 timestep, not a strength of zero; .audioOnly re-muxes the source picture instead of decoding it

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
* [loadModels() rebuilds everything](pitfalls/loadmodels-is-all-or-nothing.md) - recovering one unloaded component that way rebuilds the 22B mid-run and silently drops any fused LoRA
* [Don't validate the LTX Gemma by generating text](pitfalls/ltx-gemma-head-is-vestigial.md) - its tied head is saturated by design; check parameter coverage, scale band and meaning instead
* [The continuation-tail clip must be re-encoded](pitfalls/continuation-tail-clip-encoding.md) - an input seek leaves frame 0 off t=0 and the zero-tolerance extractor refuses it
* [URLSession's per-task delegate never reports download progress](pitfalls/urlsession-task-delegate-has-no-download-progress.md) - download(for:delegate:) calls didWriteData zero times; needs a session delegate on an explicit downloadTask, which then puts error pages on disk
* [The duration head ignores durations written in the prompt](pitfalls/duration-head-does-not-read-written-durations.md) - a "3 seconds." prefix returns byte-identical output to no prefix; it regresses from connector tokens and never sees the text
* [The conv VAE decoder padded every conv with reflect instead of zeros](pitfalls/conv-decoder-wrong-spatial-padding.md) - every clip's default decode path; 17-27% relative error against the reference, ~1e-6 once fixed; found by the new element-wise parity harness (issue #57)
* [The text connector's register replacement reordered tokens instead of substituting in place](pitfalls/connector-register-replacement-reorders-tokens.md) - every real (left-padded) prompt hit this; 135% relative error against the reference, 0.15% once fixed
* [The vocoder's float32 policy only ever cast the runtime input, never its checkpoint weights](pitfalls/vocoder-weights-stayed-bf16.md) - every LipDub/audio generation ran BigVGAN's ~108-conv chain on bf16 weights; 2-9% relative error, ~1e-5/1e-4 once the loader casts the parameters too
* [A quantized transformer load first materialised the entire bf16 checkpoint](pitfalls/quantized-load-materialised-full-bf16.md) - issue #86's GPU timeout; one combined eval() forced ~54 GB peak before quantization could shrink anything, fixed by per-block eval + dropping the source dict early (~37 GB after)
* [Quantizing the Gemma 4 text encoder adds memory instead of saving it](pitfalls/gemma4-quantize-does-not-release-bf16.md) - measured additive not replacing (bf16 ~22.7 GB, int4 ~29.1 GB); independent of eval ordering, chunking granularity, or reference lifetime — root cause is outside this repo (mlx-swift-lm's SwitchLinear or mlx-swift's quantize())

# Investigations

* [The enhancer's residual defects are the reference's](investigations/enhancer-residual-defects-2026-08.md) - measured against the Space's own inference across four prompts
* [What LTX-2.5 actually changes (August 2026)](investigations/ltx-2.5-checkpoint-diff-2026-08.md) - tensor-level diff against 2.3: the DiT moves by two flags, the VAEs and upscaler are unchanged, the cost is the Gemma 4 encoder
* [Cross-modal AdaLN sigma swap (May 2026, corrected Aug 2026)](investigations/crossmodal-adaln-sigma-swap-2026-05.md) - the LipDub mouth-modulation root cause, two expensively-refuted hypotheses, the audio-anchored vs pose-anchored trade-off, and two follow-up parity-harness findings: the May fix itself half-backwards (scale/shift wants OWN sigma, only the gate wants the OTHER's), and scale/shift wrongly collapsed from per-token to one broadcast value (affects every keyframe/IC-LoRA/LipDub-reference generation)
* [LipDub segmentation campaign (July 2026)](investigations/lipdub-segmentation-asks-2026-07.md) - what unit tests, the in-process E2E, the code review and real reruns each caught that the others missed
* [Custom-voice timbre chain (July 2026)](investigations/custom-voice-timbre-chain-2026-07.md) - attributing a bad custom-voice LipDub across Voxtral enrollment and LTX; the decoder was innocent
* [LipDub on LTX-2.5 — attributing "worse lip tracking"](investigations/lipdub-25-quality-attribution-2026-08.md) - seed variance dominated; the Gemma 4 enhancer drops the language and adds a soundtrack; quantization and LoRA coverage cleared; a single-seed scale recommendation retracted

# Playbooks

* [Diagnosing a lip-sync offset](playbooks/lipsync-offset-diagnosis.md) - the ordered checklist (transcript → segment length → prompt → channels → windows → fusion → timbre) before suspecting the model
