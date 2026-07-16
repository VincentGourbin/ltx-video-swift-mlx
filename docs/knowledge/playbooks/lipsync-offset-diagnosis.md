---
type: Playbook
title: Diagnosing a lip-sync offset or wrong mouth shapes
description: The ordered checklist that attributes a bad LipDub to its real cause — prompt, audio channels, speech windows, or alignment — before suspecting the model.
tags: [lipdub, diagnosis, lip-sync, playbook]
timestamp: 2026-07-16T00:00:00Z
---

Bad LipDub output has four known causes with distinct signatures. Check in
this order — each step is cheaper than the next.

# 1. Prompt format (wrong mouth SHAPES, sync irrelevant)

The prompt must contain `speaking in <LANG> saying: "<literal dialogue>"`.
Scene-only prompts produce structurally wrong poses (wide smile on neutral
speech). See [the prompt pitfall](/docs/knowledge/pitfalls/lipdub-prompt-needs-dialogue.md).
Also check language: a prompt language mismatching the audio degrades sync
(user-verified).

# 2. Audio channel handling (mouth moves in WRONG DIRECTIONS)

If the reference audio went through any mono downmix (`ffmpeg -ac 1`, forced
`AVNumberOfChannelsKey: 1`), the AudioVAE features are garbage. See
[the stereo pitfall](/docs/knowledge/pitfalls/audio-must-stay-stereo.md).

# 3. Speech-window detection (constant OFFSET / late attack)

Run with `--debug` and read the alignment log:

```
[lipdub] source speech window: 0.200s..4.580s (4.380s)
[lipdub] target speech window: 0.290s..4.050s (3.760s)
[lipdub] time-stretch rate=0.858 (pitch preserved)
```

Red flags: a window spanning the whole clip (detection found no boundaries),
or a `rate` far from `target speech / source speech`. Cross-check the audio
with ffmpeg:

```bash
ffmpeg -i target.wav -af silencedetect=n=-35dB:d=0.2 -f null -   # boundaries?
ffmpeg -i target.wav -af "atrim=0:0.3,astats=metadata=1" -f null - 2>&1 | grep "RMS level"  # noise floor
```

A noise floor above -35 dB (enrolled voices) is handled since PR #36 by the
floor-relative threshold — see
[the thresholds decision](/docs/knowledge/decisions/speech-window-noise-floor.md).
If the windows are wrong anyway, tune `thresholdDB`/`noiseFloorMarginDB` on
`alignTargetToSource` and file what you learned here.

# 4. Fusion state (everything looks right but output is degraded)

On consecutive runs in one process, verify the fusion log says either
`LoRA fused: 1344 / 1344 layer-pairs (100.0%)` or
`LoRA already fused (same file) — reusing fused transformer`. A doubled
delta or a stale file should be impossible since PR #36 (guards throw), but
if you see burned/saturated output, check
[the double-delta pitfall](/docs/knowledge/pitfalls/lora-refusion-double-delta.md).

# When all four pass

The residual is the known audio-anchored vs pose-anchored trade-off — see
[the AdaLN investigation](/docs/knowledge/investigations/crossmodal-adaln-sigma-swap-2026-05.md)
for the quantitative comparison method (Pearson audio-envelope vs
mouth-openness) before concluding anything.
