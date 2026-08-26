# Directory Update Log

## 2026-08-26

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
