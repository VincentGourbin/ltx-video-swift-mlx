# Directory Update Log

## 2026-09-06

* **Pitfall**: [The text connector goes NaN on M5/macOS 26.2+](/docs/knowledge/pitfalls/nax-splitk-gemm-m5-black-video.md)
  — issue #69 / PR #85. Root-caused to
  [mlx#3797](https://github.com/ml-explore/mlx/issues/3797): the NAX split-K
  GEMM kernel templates on its accumulator's hardcoded `float32` dtype
  instead of the actual input dtype, misreading bf16 buffers on gen-17+ GPUs
  — the connector's `[1024,4096]@[4096,16384]ᵀ` feed-forward down-projection
  sits exactly on the dispatch threshold. Fixed upstream (mlx#3810,
  2026-07-07) but no mlx-swift tag vendors it. Reviewed PR #85's blanket f32
  cast (posted to the PR after correcting two errors in the draft: the TF32
  explanation, and a false claim that LTX-2.3's connector has a different,
  unaffected shape — it's identical to 2.5's) and requested changes rather
  than merging: it doesn't cover the video transformer's own same-shape-class
  exposure, and it changes the connector's output dtype on every machine, not
  just affected ones. Investigated bumping mlx-swift to `main` instead
  (`chore/mlx-swift-main-nax-fix`, not merged) — real, reproducible ~20-28%
  output divergence on non-M5 hardware, bisection blocked by `mlx-c` (a
  second submodule pinned in lockstep with `mlx`) requiring a very recent
  `mlx` snapshot to compile at all, leaving almost no independently-testable
  range. Shipped a narrower fix instead: `MLXNAXSplitKWorkaround` replicates
  mlx's own `is_nax_available()` hardware check (same
  `MTLDevice.architecture.name` input, same parsing) and gates the same f32
  cast behind it — verified byte-identical connector/VAE output on an M3 Max
  (unaffected hardware) before/after, and a finite/reasonable output when the
  workaround is forced on via `LTX_NAX_WORKAROUND=1` (can't verify against
  the actual M5 NaN — no M5 hardware in this repo).

## 2026-09-05

* **Pitfall**: [The duration head must see audio connector tokens, not just video](/docs/knowledge/pitfalls/duration-head-needs-audio-tokens.md)
  — `predictFrameCount` fed the LTX-2.5 duration head video connector tokens
  only, while upstream always builds the audio connector and gives the head
  both streams. On the reference scene (seed 42, 2.5-distilled) this was a
  27.0 s → 473 frames (clamped) vs 4.09375 s → 97 frames difference, over 6x,
  on the identical prompt. Fixed in `fix/duration-head-audio-tokens` by
  building a throwaway video+audio encoder for the one call that feeds the
  head, never touching `self.textEncoder` — the actual generation path
  (`videoEncoding`) is verified byte-identical before/after. Re-measured the
  before/after tables in
  [duration-head-does-not-read-written-durations](/docs/knowledge/pitfalls/duration-head-does-not-read-written-durations.md)
  and updated `DurationPromptE2ETests`' assertions to the new values (not a
  regression — the old assertions were pinned to the wrong inputs).
* **Pitfall**: [A quantized transformer load first materialised the entire bf16 checkpoint](/docs/knowledge/pitfalls/quantized-load-materialised-full-bf16.md)
  — issue #86 (GPU timeout on a 36 GB M3 Max loading the transformer at
  50%). `loadModels()` evaluated the whole bf16 transformer in one combined
  command buffer before quantizing (54519 MB peak measured on int4), then
  evaluated the whole quantized model again in another. Fixed on
  `fix/quantized-load-per-block`: `evalParametersPerBlock` evaluates each
  transformer block individually (both for the bf16-only path and after
  quantization), and the source weight dictionary is explicitly cleared
  right after it's applied instead of staying alive for the rest of the
  function. Peak dropped to 37180 MB (~17 GB less transient overhead);
  `VAE raw output` byte-identical before/after on a plain generation.
* **Pitfall**: [Quantizing the Gemma 4 text encoder adds memory instead of saving it](/docs/knowledge/pitfalls/gemma4-quantize-does-not-release-bf16.md)
  — found while chasing the rest of issue #86's budget: the transformer fix
  above didn't get a 36 GB machine under budget, because
  `Gemma4TextEncoder.load` costs *more* quantized (int4: ~29.1 GB active)
  than unquantized (bf16: ~22.7 GB active) — additive, not replacing. Tested
  every lever this repo controls (eval ordering relative to `quantize()`,
  eval granularity — whole-model, fixed-count chunks, and true per-layer via
  `Gemma4LLMModel.loraLayers` — and Swift-side reference lifetime of the
  source bf16 dictionary) and none of it moved the number by even 1 MB.
  Root cause is outside this repo: likely `mlx-swift-lm`'s `SwitchLinear`
  (the MoE layer type `gemma4-12b-ltx-v1` uses, per `Gemma4Experts.swift`'s
  own "26B-A4B" comment) not releasing on quantization, or `mlx-swift`'s
  `quantize(model:)` submodule-replacement mechanism not fully detaching
  what it replaces for that shape of module. Documented rather than fixed;
  the transformer-side chantier C fix ships regardless since it's a real,
  independent improvement.

## 2026-08-31 (4)

* **Validation**: Sub-task 6 of issue #57 — the last of the six —
  `SpatialUpscalerParityTests` pins `SpatialUpscaler` (the second stage of
  every two-stage generation, LipDub, and the IC-LoRA v2v path) against
  Lightricks' own config-driven `LatentUpsampler`
  (`scripts/spatial_upscaler_reference.py`). This was the component the
  plan flagged as most likely to hide a sub-task-1-style dimension-order
  bug (the pixel-shuffle resampler's H/W-vs-shuffle-factor pairing can be
  wrong without the output *shape* changing at all) — clean on the real
  run: every stage (initial_conv, all 8 res blocks, the resampler, final
  output) held at 7e-7-4e-6 against the reference. The only bug this
  sub-task found was in the test itself, not the port: the first written
  version of `bisectFirstDivergence` compared its `initial_conv` tap
  *after* `initial_norm` + SiLU against a reference hook that fires on the
  bare `Conv3d` alone (73% relative error) — a reminder that a bisection
  harness's own tap placement needs the same scrutiny as the port it's
  checking. A `/code-review` pass on the PR (no bugs found in the port
  itself — three angles independently re-derived every transpose) hardened
  the harness before merge: `final_conv` gained its own tap (previously
  exercised only via the aggregate output number), the fixture moved to
  batch=2 (batch=1 made the resampler's batch/frame fold order
  undiscriminating — any fold order looks identical at batch=1), the
  threshold tightened from 2% to 2e-4 (matching the lesson already on
  record in this file's DualStreamAudioParityTests entries), and the
  output-shape check gained a guard against a crash on mismatch instead of
  a clean failure.

## 2026-08-31 (3)

* **Correction (2nd)**: [Cross-modal AdaLN sigma swap](/docs/knowledge/investigations/crossmodal-adaln-sigma-swap-2026-05.md) —
  a `/code-review` gap-sweep on the PR carrying the sub-task 5 fix above
  caught a second, independent defect in the same lines, verified against
  `ltx_core`'s `_prepare_cross_attention_timestep`: the cross-modal
  scale/shift AdaLN input was collapsed to one value per modality via
  `.max(axis: 1)` and broadcast to every token, when the reference keeps it
  genuinely per-token. Only the gate is a true scalar. Matters wherever
  `AppendedGuideTokens.swift`'s `buildExtendedTimestep` already builds
  non-uniform per-token timesteps in production — every keyframe, IC-LoRA and
  LipDub-audio-reference generation. Required splitting
  `AudioTransformerArgs.crossVideoScaleShift`/`crossAudioScaleShift` (now
  per-token `(B,T,4,D)`) from new `crossVideoGate`/`crossAudioGate` fields
  (scalar-broadcast `(B,1,1,D)`) — the two could no longer share one fused
  `(B,1,5,D)` tensor. The parity harness's fixture (both Python and Swift)
  was updated to non-uniform per-token timesteps to actually exercise this;
  reverting the fix locally confirmed the new fixture catches it (clean
  ~1e-6 → 1.1e-3/5.4e-3 collapsed), which also revealed the suite's original
  2% threshold was too loose to have caught either regression — tightened to
  2e-4, matching `TransformerParityTests`'s video-only precision.

## 2026-08-31 (2)

* **Correction**: [Cross-modal AdaLN sigma swap](/docs/knowledge/investigations/crossmodal-adaln-sigma-swap-2026-05.md) —
  sub-task 5 of issue #57's breakdown. The first element-wise reference for
  `LTX2Transformer`'s dual video/audio blocks (`DualStreamAudioParityTests`,
  extending `scripts/transformer_reference.py` with an "av" variant, small
  dims, *deliberately different* sigmas per stream — equal sigmas make a
  sigma swap a no-op) found the May 2026 fix for this exact file was itself
  half-backwards: only the cross-modal GATE AdaLNs want the opposite
  modality's sigma; the SCALE/SHIFT AdaLNs want their own. The May fix
  pointed both the same way, correctly fixing the gate but breaking
  scale/shift. Per-module isolation: scale/shift own-sigma error 3e-6/8e-7
  vs cross-sigma 0.55/0.26; gate cross-sigma error 1.5e-7/3.4e-8 vs own-sigma
  0.066/0.015. Full output error 3.6e-3/8.2e-3 → 2.1e-6/1.2e-6 — both already
  under the 2% pass/fail threshold even with the bug present, confirming the
  plan's warning that output-only thresholds aren't sensitive enough for this
  component. Real end-to-end check (`retake --modality audio`, before/after,
  same seed): RMS 0.045 → 0.0097, 0-2 kHz band down 16.6 dB.

## 2026-08-31

* **Creation**: [The vocoder's float32 policy only ever cast the runtime
  input, never its checkpoint weights](/docs/knowledge/pitfalls/vocoder-weights-stayed-bf16.md) —
  sub-task 4 of issue #57's breakdown. `AudioVAEVocoderParityTests`, the
  first element-wise reference for the real audio decode chain
  (`AudioVAE.decode` → mel → `LTXVocoderWithBWE`, against Lightricks' own
  `AudioDecoder` + `VocoderWithBWE`), found the BigVGAN vocoder's `Conv1d`/
  `ConvTransposed1d` weights still at the checkpoint's native bf16 despite a
  comment claiming float32 execution throughout — only the runtime
  activation was ever cast. 2.0-8.8% relative error on the vocoder/BWE taps
  and the final waveform, collapsing to ~1e-5/1e-4 once
  `BigVGANWeightLoader.load` casts every loaded parameter to float32. The
  AudioVAE decoder upstream (sub-task's other half) was clean on the first
  run: all six taps ~1e-7 against the reference.

## 2026-08-30

* **Creation**: [The conv VAE decoder padded every conv with reflect instead
  of zeros](/docs/knowledge/pitfalls/conv-decoder-wrong-spatial-padding.md) —
  found by a new element-wise parity harness against Lightricks' own
  `ConvVideoDecoder` (issue #57), the first of six planned sub-tasks. 17-27%
  relative error on the default decode path used by every clip this repo has
  ever produced, collapsing to ~1e-6 once the five conv sites in
  `VideoDecoder.swift` got the checkpoint's actual `spatial_padding_mode:
  "zeros"` instead of the framework's fallback default. Corrects a wrong
  claim in [the D2S-residual pitfall](/docs/knowledge/pitfalls/decoder-d2s-residual-false.md),
  which had the padding modes backwards.

* **Validation**: Sub-task 2 of issue #57 — `ConvVAEEncoderParityTests` pins the
  conv VAE encoder (every retake, i2v conditioning image, and LipDub video
  reference goes through it) against Lightricks' own `VideoEncoder`
  (`scripts/conv_video_encoder_reference.py`). Unlike the decoder (sub-task 1,
  PR #76), this one was clean on the first run: raw means and the fully
  normalized output both match to ~3e-6. The encoder's `_res`-suffixed
  space-to-depth downsamplers and `.zeros` padding, previously only
  documented, are now verified rather than assumed.

* **Creation**: [The text connector's register replacement reordered tokens
  instead of substituting in place](/docs/knowledge/pitfalls/connector-register-replacement-reorders-tokens.md)
  — sub-task 3 of issue #57's parity-harness breakdown (sub-task 1: conv VAE
  decoder padding, PR #76; sub-task 2: conv VAE encoder, clean, PR #77). The
  8-block transformer (RoPE, gated attention, feed-forward) was cleared first
  via bisection — matches the reference to ~1e-5 — which localized the
  defect to `replacePaddedWithLearnableRegisters`. Since every real prompt is
  left-padded (`Gemma4TextEncoder.encode`) and far shorter than the
  1024-token window, this was live on essentially every generation this repo
  has produced: 135% relative error on the connector output, 0.15% after the
  fix.

* **Creation**: [Vectorized RGBA pixel conversion vs. the scalar loop it
  replaced](/docs/knowledge/benchmarks/pixel-conversion-vectorization-2026-08.md)
  — `loadVideo` on a 121-frame 768x512 clip: ~8.15s to ~1.0s (~8x). The
  remaining ~1s is AVFoundation decode overhead, not pixel conversion —
  documented as the next lever if `loadVideo` needs to be faster still.

* **Update**: Split the mid-run unload gating per component in
  [the unload-gating decision](/docs/knowledge/decisions/unload-gating-semantics.md).
  One flag for the prompt encoder and the transformer made LipDub fusion reuse
  reachable only at `.disabled` — and `.disabled` holds LTX-2.5's 26 GB encoder
  resident, against the 7.5 GB the original trade-off was costed on. Reported by
  Fluxforge (§7) as a missing `Memory.clearCache()`, which was already there.

* **Creation**: [loadModels() rebuilds everything](/docs/knowledge/pitfalls/loadmodels-is-all-or-nothing.md)
  — the trap that made the split worth doing: recovering one component through
  `loadModels()` rebuilds the 22B mid-run and drops any fused LoRA with it. The
  dev retake path did exactly that for its negative prompt.

* **Creation**: [A retake picks its stream with a modality](/docs/knowledge/decisions/retake-modality-frozen-stream.md)
  — `.videoOnly` / `.both` / `.audioOnly`, why the proposed "strength of zero"
  shape could not work, the `audioRetakeStrength` schedule entry point (trained
  sigmas only, audio-only by construction), and the CFG-multiplied audio Euler
  step found while implementing it.

## 2026-08-29

* **Creation**: [LipDub on LTX-2.5 — attribution campaign](/docs/knowledge/investigations/lipdub-25-quality-attribution-2026-08.md)
  — a reported 2.3→2.5 lip-tracking drop, reproduced from the app's own
  SwiftData store and attributed by matrix runs. Seed variance dominated (the
  app's rendered seed was a bad draw; another seed beat the 2.3 reference);
  the one systematic defect is the Gemma 4 enhancer dropping the language from
  the trained wrapper and appending background music. Cleared: LoRA coverage
  (1344/1344 both), the audio→video pathway (gate 1.46× STRONGER in 2.5),
  quantization (qint8 ≈ bf16). Retracted: a single-seed `--lora-scale 1.3`
  recommendation — variance alone covered the gap. The signature fallback now
  requires a capitalized language after `speaking in`; "speaks in a clear
  voice" no longer satisfies it.

## 2026-08-26

* **Creation**: [The duration head ignores durations written in the prompt](/docs/knowledge/pitfalls/duration-head-does-not-read-written-durations.md)
  — `--frames auto` regresses a length from connector tokens, not from text: a
  `3 seconds.` prefix returns byte-identical output to no prefix (23.5 s both
  times). A style-dense prompt asking for 15 s predicted 5.16 s / 121 frames,
  and the enhancer's rewrite dropped the "15 seconds" outright. Also records
  that auto-duration's ceiling is 473 frames while an explicit `--frames`
  reaches 481 — the head rounds a *prediction* down.

* **Validation**: `scripts/duration_head_reference.py` replaces the NumPy
  re-implementation `DurationHeadE2ETests` used to pin against — it runs
  upstream's `ltx_core.duration_head.DurationHead`, built through upstream's own
  `DurationHeadConfigurator.from_metadata` so the head count comes from the
  checkpoint rather than a literal (float64: 11.402804284 s). A 5.6875 s row
  exposes the one language-semantics gap: Python's `round()` is banker's,
  Swift's `.rounded()` is half-away-from-zero — 129 frames against 137. The port
  now matches.

* **Creation**: [URLSession's per-task delegate has no download progress](/docs/knowledge/pitfalls/urlsession-task-delegate-has-no-download-progress.md)
  — `download(for:delegate:)` accepts a `URLSessionDownloadDelegate` and calls
  `didWriteData` zero times (measured: 32 MB transfer, `Content-Length`
  present, 0 delegate calls). Byte progress needs a session-level delegate on
  an explicit `downloadTask`, which then puts the response body on disk before
  the status is checked — so a 404 page can be cached as a checkpoint unless
  the destination is deleted on non-200. Answers ask 1 of the Fluxforge
  LTX-2.5 asks, along with the non-monotonic aggregate in `downloadCheckpoint`.

* **Creation**: [Prompt enhancer source](/docs/knowledge/decisions/prompt-enhancer-source.md)
  — LTX-2.5 enhancement needs a generative E2B-it separate from the encode-only
  bundled encoder, so `--enhance-prompt` puts a second Gemma on disk. bf16
  (10.24 GB) stays the default for reference parity; 6-bit (4.74 GB, quality
  unmeasured) and a caller-supplied root are opt-in. Records why the file list
  must be enumerated rather than hardcoded — bf16 ships three shards, 6-bit
  one — and that a quantized checkpoint's `quantization` block is applied by
  the loader, so a caller's root needs no precision flag. Answers ask 2 of the
  Fluxforge LTX-2.5 asks.

## 2026-08-13

* **Creation**: [IC-LoRA stage 2 keeps adapter and reference](/docs/knowledge/decisions/iclora-stage2-keeps-adapter-and-reference.md)
  — the upscale pipeline's refinement stage deliberately diverges from
  ic_lora.py: seven runs show subject identity surviving the σ-0.909 renoise
  only when the adapter and the reference are both active in the final stage.
  Also records that centroid/scale/HF metrics were all blind to the failure —
  the discriminating check was "same car between stage 1 and final", one glance
  at the `--stage-one` export.

* **Creation**: [Module.update mutates in place](/docs/knowledge/pitfalls/module-update-mutates-in-place.md)
  — `unfuseLoRA` had never restored a single weight: the originals captured for
  restore were bare references into the module, and MLXNN's in-place update made
  them track the fused values. Surfaced by a bit-identical output between an
  unfused and a fused stage-2 run; pinned by a 2-layer round-trip test and a
  capture-purity probe. Fix is copy-at-capture, materialised before the update.
  Partially supersedes July's double-delta entry: unfuse restored contaminated
  weights in *every* case, not just the LipDub-then-LoRA one.

## 2026-08-20

* **Creation**: [Smoothness metrics miss identity drift](/docs/knowledge/pitfalls/smoothness-metrics-miss-identity-drift.md)
  — the tiling seams measured clean and were not; per-frame fidelity against
  the source found both defects the user had reported.


* **Creation**: [A renoise level needs its anchoring](/docs/knowledge/pitfalls/renoise-level-needs-its-anchor.md)
  — temporal interpolation shipped; upstream's sigma redrew the subject until
  the refinement was started lower (identity 14.4 → 20.1 dB).

## 2026-08-21

* **Creation**: [LTX-2.5 against MiniMax-H3 on a 10 s case](/docs/knowledge/benchmarks/ltx25-vs-h3-starship-2026-08.md)
  — 50 min 50 against 3 h 55 for the same work, one full-resolution step 4.9×
  cheaper, and a 2.8× spread between two identical LTX runs that dwarfs most of
  what it measures. Each model wins half the timecode criterion; prompt
  enhancement moved the event *away* from the requested time.

* **Creation**: [The enhancer's residual defects are the reference's](/docs/knowledge/investigations/enhancer-residual-defects-2026-08.md)
  — four prompts against the Space's own inference: viewpoint stacking appears
  in both (verbatim the same phrase on one prompt), detached adverbs three times
  more often upstream, and only the Space breaks spacing inside quoted dialogue.
  Nothing here to fix; what is left is a product choice.

* **Creation**: [Cross-attention's q_norm is not the block's pre-norm](/docs/knowledge/pitfalls/cross-attention-prenorm.md)
  — the first bug the new *transformer* parity harness found, on its first run:
  the legacy 6-value block fed cross-attention the raw residual (1.1e-2 relative
  error, 6.2e-6 after). No shipped generation used that path; both real block
  variants now pin at ~5e-6 against upstream.

* **Creation**: [Generated keyframe slots are appended, denoised and marked](/docs/knowledge/decisions/generated-keyframe-slots.md)
  — DFR's last missing primitive, and the audit finding that came with it: the
  2.5 "detailing LoRA" is the pixel spatial upscaler this package already
  drives, so only the slots needed porting. Records the three properties that
  separate a slot from an appended guide token, and the parity check against
  Lightricks' own `_slot_positions`.

* **Creation**: [A densified clip must be positioned at its new rate](/docs/knowledge/pitfalls/densified-clip-keeps-its-rate.md)
  — the temporal round doubles frames *and* fps; positioning at the source's
  rate made the model read a 4.9 s clip as 9.8 s. Quiet because the grid and the
  anchors agreed with each other.

## 2026-08-19

* **Creation**: [Tiled-attention mask caches need the whole window pattern](/docs/knowledge/pitfalls/na-tile-mask-cache-key.md)
  — the last bug between the DiffVAE port and element-wise parity; found by the
  new reference harness, which now pins the whole decoder at ~1e-6.


* **Creation**: [Dotted parameter names never load](/docs/knowledge/pitfalls/dotted-parameter-names-never-load.md)
  — found while bringing up the LTX-2.5 diffusion video decoder: the latent
  statistics silently kept mean 0 / std 1, washing out every decode.

## 2026-08-17

* **Update**: archived the distilled-vs-dev quality series as the LTX-2.5
  benchmark anchors ([docs/examples/ltx-2.5](/docs/examples/ltx-2.5/README.md),
  `series-25-*`) — same seed/prompt/337 frames across distilled two-stage,
  dev+LoRA450, no-LoRA control, and the 30-step dev ceiling; wall times and
  memory peaks recorded in the README table.

* **Update**: [no_repeat_ngram bans quoting the prompt](/docs/knowledge/pitfalls/ngram-blocking-mangles-prompt-quoting.md)
  — fixed via gemma-4-swift-mlx 1.2.0 `includePromptInWindow: false`;
  duration prediction on the enhanced 2CV prompt fell 19.04 s → 14.71 s.

## 2026-08-16

* **Creation**: [no_repeat_ngram bans quoting the prompt](/docs/knowledge/pitfalls/ngram-blocking-mangles-prompt-quoting.md)
  — ngram on/off A/B: verbatim timestamps and 14.04 s predicted without the ban,
  mangled timestamps and 19.04 s with it; deviation ask handed to gemma-4-swift-mlx.


* **Creation**: [CFG against an empty negative erases the prompt](/docs/knowledge/pitfalls/empty-cfg-negative-erases-the-prompt.md)
  — root-caused on the 2CV bench: dev single-stage (30 steps, cfg 3.0) lost the
  entire choreography with the port-inherited "" negative and recovered it with
  upstream's DEFAULT_NEGATIVE_PROMPT, single variable flipped (p5-d vs p5-d2).

## 2026-08-12

* **Creation**: [LipDub overwrites anything crossing the mouth](/docs/knowledge/pitfalls/lipdub-overwrites-objects-crossing-the-mouth.md)
  — the IC-LoRA repaints the mouth region without modelling occlusion, so a
  headset band, hand or microphone in front of the lips is painted over.
  Demonstrated on the repo's own teaser (frames 52–64), which predates the
  vocoder work, so it is a property of the method rather than a regression. Also
  records the attribution lesson: colour and occlusion differences between two
  LipDub outputs were both blamed on a vocoder that only runs after video
  decoding.

* **Update**: The audio decode stage was running the wrong vocoder — recorded in
  [the vocoder pitfall](/docs/knowledge/pitfalls/wrong-vocoder-lost-the-top-octave.md).
  LTX-2.3 and LTX-2.5 bundle a BigVGAN v2 + bandwidth-extension pair (667+557
  tensors, byte-identical between generations, 48 kHz); this package loaded
  LTX-2's 194-tensor 24 kHz vocoder, which shares no key with them. A same-seed
  A/B puts the cost at **+18 dB in 12–16 kHz** once corrected, plus a 16–24 kHz
  band that did not exist, and **nothing below 8 kHz** — the top octave, not the
  midrange. Confirmed on speech via a 2.3 LipDub run. The entry also records the
  measurement trap that first produced a wrong 40 dB claim: the initial A/B
  compared generations of different lengths, so content differences were read as
  vocoder differences. The July timbre investigation's conclusions therefore
  **stand**; the caveat added to it earlier that day has been withdrawn.

* **Update**: LTX-2.5 now runs (text/image-to-video). Two pitfalls recorded from
  the port: [split-checkpoint lookups fail silently](/docs/knowledge/pitfalls/split-checkpoint-silent-empty-load.md)
  — the VAE encoder was read from the transformer file, matched zero keys, kept
  its random initialisation and encoded every conditioning image to noise, while
  the run still produced coherent video of the wrong car — and
  [the LTX Gemma head is vestigial](/docs/knowledge/pitfalls/ltx-gemma-head-is-vestigial.md)
  — greedy decoding emits single capital letters on any prompt because the
  encoder fine-tune let the final-norm scale drift 2.5x above stock, saturating
  the logit softcap; norm statistics compared against
  mlx-community/gemma-4-e4b-it-4bit confirm the conventions match.

* **Creation**: [What LTX-2.5 actually changes](/docs/knowledge/investigations/ltx-2.5-checkpoint-diff-2026-08.md)
  — every 2.5 component's safetensors header read by HTTP range request and
  diffed against 2.3. The DiT differs by two config keys (`ff_bias: false`,
  `use_keyframes_abs_pos_embedding: true`), 96 dropped FFN biases and one new
  `[1, 4096]` marker; the conv video VAE, the audio VAE + vocoder and the
  latent spatial upscaler are tensor-for-tensor identical; the sigma schedules
  are unchanged. The port cost is the `gemma4-12b-ltx-v1` text encoder, which
  exists only inside the 26 GB LTX file. Also records two dead download URLs
  found live (LipDub → DubIt rename, spatial upscaler 1.0 withdrawn) and the
  new `reference_spatial_scale_factor` IC-LoRA metadata key.

## 2026-07-27

* **Update**: Voxtral closed out the custom-voice loose ends, and two of their
  findings correct ours. **q6 beats bf16 on cloned voices** (99.4 % vs 96.5 %
  coverage, RTF 1.47 vs 3.44, 3.5 GB vs 8) — the opposite of what a single
  observation of ours had suggested, withdrawn upstream; the **exact digital
  zeros come from the codec** and vary 3.4 %–10.5 % *between generations*, so
  they are not an enrollment artefact and no consumer may assume a natural
  floor; and the **residual ~5 dB of fundamental is inherent** to generation,
  with no fix pending. Recorded as rules 7 and 8 of
  [the audio contract](/docs/knowledge/pitfalls/lipdub-audio-contract.md)
  (rule 7 also pins why `detectSpeechWindow`'s credibility guard must survive
  any rewrite) and in the
  [investigation](/docs/knowledge/investigations/custom-voice-timbre-chain-2026-07.md),
  which now carries the process lesson: an n=1 result was filed as a
  recommendation on another team's tracker and pointed the wrong way.

## 2026-07-26

* **Creation**: Custom-voice LipDub attribution campaign (23–26 July).
  [Segment bound ~233 frames](/docs/knowledge/pitfalls/lipdub-segment-bound-233.md)
  — the negative-position audio reference doubles the RoPE span, so 481 does
  not apply to LipDub (constant 0.75 s lag measured at 377 frames, in sync at
  233); [continuation-tail clip encoding](/docs/knowledge/pitfalls/continuation-tail-clip-encoding.md)
  — the `-sseof` recipe shipped with PR #40 produces a clip the extractor
  refuses; and the [custom-voice timbre chain](/docs/knowledge/investigations/custom-voice-timbre-chain-2026-07.md)
  — the LTX audio decoder cleared by three measurements, the real losses being
  upstream in Voxtral enrollment (mlx-voxtral-swift#44).

* **Update**: The continuation anchor reads the tail **natively** — the
  [tail-clip pitfall](/docs/knowledge/pitfalls/continuation-tail-clip-encoding.md)
  is now historical (marked as such, kept because it explains why an API that
  asks callers to hand-cut a clip is a trap), and
  [the continuation decision](/docs/knowledge/decisions/lipdub-continuation-anchor.md)
  records the withdrawn contract. No ffmpeg mention remains anywhere in
  `Sources/`.

* **Creation**: [The LipDub audio contract](/docs/knowledge/pitfalls/lipdub-audio-contract.md)
  — one place for what an integrator must respect on the audio side (ship the
  generated track, verbatim transcript, ≤233-frame segments, reference quality,
  why post-hoc normalisation does not repair one, and F0-vs-H2 as the timbre
  metric), each rule carrying the measurement that established it.

* **Update**: [The prompt pitfall](/docs/knowledge/pitfalls/lipdub-prompt-needs-dialogue.md)
  now states the stronger rule — the dialogue must be the *verbatim
  transcript* of the target audio, because the generated speech follows the
  prompt (a mismatch voided a day of timing measurements);
  [the 481 decision](/docs/knowledge/decisions/frame-cap-481-rope-range.md)
  carries the LipDub caveat; [the lip-sync playbook](/docs/knowledge/playbooks/lipsync-offset-diagnosis.md)
  gains a transcript check and a segment-length check ahead of everything
  else, plus a timbre section (F0 vs H2).

## 2026-07-17

* **Creation**: [LipDub continuation-anchor decision](/docs/knowledge/decisions/lipdub-continuation-anchor.md)
  (issue #35 implemented and measured: seam PSNR 17.4 → 24.6 dB).

* **Update**: Corrected the activation-memory sizing rule in the
  [training baselines](/docs/knowledge/benchmarks/lora-training-baselines-m3max.md)
  (per-regime marginal costs instead of one averaged slope that
  under-predicted at the OOM threshold) and aligned the
  [QLoRA decision](/docs/knowledge/decisions/qlora-training-default.md) with
  the code: qint8 is now the actual training default (PR #38 review).

## 2026-07-16

* **Update**: LoRA-training validation campaign (issue #1 revival):
  added [training baselines](/docs/knowledge/benchmarks/lora-training-baselines-m3max.md)
  and the [QLoRA training decision](/docs/knowledge/decisions/qlora-training-default.md)
  (bf16 84.3 GB swapping vs qint8 43.6 GB / int4 37.5 GB, near-exact loss
  parity through the frozen quantized base). Note: this PR also materially
  adds the [generation baselines](/docs/knowledge/benchmarks/generation-baselines-m3max.md)
  concept — it was listed in the bootstrap entry below but a `benchmarks/`
  gitignore rule had silently kept it out of PR #37.

* **Creation**: Bootstrapped the knowledge bundle after the LipDub
  app-integration campaign (PR #36) and the RuntimeBeacon work (PR #34).
  Initial concepts: the M3 Max [generation baselines](/docs/knowledge/benchmarks/generation-baselines-m3max.md),
  four decisions ([frame cap](/docs/knowledge/decisions/frame-cap-481-rope-range.md),
  [speech-window thresholds](/docs/knowledge/decisions/speech-window-noise-floor.md),
  [fusion reuse](/docs/knowledge/decisions/lipdub-fusion-reuse-policy.md),
  [unload gating](/docs/knowledge/decisions/unload-gating-semantics.md)),
  nine verified pitfalls (build/test tooling, fusion corruption paths,
  audio/prompt/keyframe contracts, two historical root causes), the
  [May 2026 AdaLN investigation](/docs/knowledge/investigations/crossmodal-adaln-sigma-swap-2026-05.md),
  the [July 2026 campaign record](/docs/knowledge/investigations/lipdub-segmentation-asks-2026-07.md)
  and the [lip-sync diagnosis playbook](/docs/knowledge/playbooks/lipsync-offset-diagnosis.md).
