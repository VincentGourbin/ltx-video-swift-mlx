---
type: Pitfall
title: Fake-stereo audio breaks the AudioVAE encoder features
description: Downmixing to mono then duplicating L=R feeds the stereo-trained encoder statistically impossible input — LipDub mouths move in wrong directions with no error anywhere.
tags: [audio, audiovae, lipdub, stereo]
timestamp: 2026-07-16T00:00:00Z
---

The AudioVAE encoder (`in_channels=2`) was trained on real stereo where L≠R
carry phase/timing differences it learned to use as features. Forcing
`AVNumberOfChannelsKey: 1` in `loadAudio()` and duplicating mono into "fake
stereo" at the mel stage produces input the encoder never saw. It outputs
garbage features; for LipDub those feed `audio_to_video_attn`, which then
modulates the mouth in WRONG directions (wide smile instead of French
phoneme shapes) — while LoRA fusion, RoPE and video anchoring are all
provably correct. Measured symptom at the time: drift uniformly ~3× above the
Lightricks reference even in source-anchored regions.

# The defense

- `AudioProcessor.loadAudio()` extracts stereo when the source has it;
  `melSpectrogram()` mels each channel independently. Returns `(samples,)`
  for mono, `(2, samples)` for stereo — callers must handle both
  (sample count is `dim(ndim - 1)`).
- When preparing test inputs, do NOT `ffmpeg -ac 1` a reference video's
  audio. Stay close to what the Lightricks pipeline consumes.

# Citations

[1] Root-caused during the May 2026 LipDub debugging campaign — see
[the cross-modal AdaLN investigation](/docs/knowledge/investigations/crossmodal-adaln-sigma-swap-2026-05.md).
