# Directory Update Log

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
