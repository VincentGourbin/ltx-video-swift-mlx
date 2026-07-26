# Directory Update Log

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
