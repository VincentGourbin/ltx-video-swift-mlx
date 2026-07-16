---
type: Investigation
title: LipDub mouth-modulation failure — cross-modal AdaLN sigma swap (May 2026)
description: Each cross-modal AdaLN was fed its OWN modality's sigma instead of the OTHER's, and the gate input was scaled 1000× wrong; two plausible hypotheses (RoPE negatives, LoRA delta) were numerically refuted along the way.
tags: [lipdub, adaln, cross-modal, debugging, root-cause]
timestamp: 2026-07-16T00:00:00Z
---

LipDub produced wrong mouth modulation despite a provably correct LoRA fusion
and video anchoring. The campaign that root-caused it is worth keeping
because two *very* plausible hypotheses were expensively refuted first.

# The bug (two-fold, in LTX2Transformer.forward)

1. **Wrong sigma source**: each cross-modal AdaLN
   (`avCrossAttn{Video|Audio}{ScaleShift|Gate}`) was fed its own modality's
   timesteps. The Python reference feeds the **opposite** modality's scalar
   sigma (`video_preprocessor.prepare(video, audio)` /
   `audio_preprocessor.prepare(audio, video)` — `ltx-core model.py:402-403`).
2. **Missing `av_ca_factor` on the gate**: Python scales the GATE AdaLN input
   by `av_ca_timestep_scale_multiplier / timestep_scale_multiplier` (= 1/1000
   with defaults). We passed `sigma × 1000`; correct is `sigma × 1`.

Why only LipDub expressed it: T2V+audio has equal sigmas (swap is a no-op);
I2V+audio's frame-0 AdaLN(0) lands on a latent that conditioning overwrites
anyway. LipDub has BOTH per-token timesteps and σ=0 reference tokens — the
bug fully expressed there.

# Hypotheses refuted (with parity harnesses)

- **RoPE on negative positions**: `precomputeFreqsCisDoublePrecision`
  matches Python to 3e-7 on the extended audio grid ([-2.0, +1.95]) —
  `RoPENegativePositionTests`.
- **LoRA delta direction**: Swift `getDelta` matches PEFT `(B @ A)`
  byte-for-byte (max abs diff 9.1e-7) — `LoRADeltaParityTests`.

# Quantitative validation of the fix

Pearson correlation of audio envelope vs mouth openness over 121 frames on
the Lightricks teaser: source (ground truth) +0.165/+0.298 (lagged);
Lightricks-FR −0.047/+0.056; **ours with the fix +0.140/+0.148**. On
end-of-clip silence our pipeline closes the mouth (openness 4.84 vs their
6.47 still open). Residual trade-off, understood and accepted: our output is
*audio-anchored*, Lightricks is *pose-anchored* — the remaining gap is pose
preservation, not lip-sync, and is numerical accumulation, not a code bug.

# Related

The [stereo pitfall](/docs/knowledge/pitfalls/audio-must-stay-stereo.md) was
root-caused in the same campaign.
