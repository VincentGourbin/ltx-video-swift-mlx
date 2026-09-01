---
type: Investigation
title: LipDub on LTX-2.5 — attributing "worse lip tracking" across four suspects (August 2026)
description: A reported 2.3→2.5 quality drop in an integrating app was attributed by matrix runs. Seed variance dominated; the enhancer's language-dropping rewrite was the one systematic defect; quantization and LoRA coverage were cleared; a single-seed scale recommendation was retracted.
tags: [lipdub, ltx25, attribution, seed-variance, enhancer, lora]
timestamp: 2026-08-29T00:00:00Z
---

> **Measured before the cross-modal fixes of 2026-08-31 (PR #82).** Every run
> below predates two proven defects in `LTX2Transformer`'s cross-modal AdaLN: the
> scale/shift pair was fed the wrong modality's sigma, and its output was
> collapsed from per-token to a single broadcast value. Both express themselves
> only when the two streams carry *divergent* sigmas — which is exactly the
> LipDub regime, and PR #82 measured a −4.6× RMS change on a real clip because of
> it. The conclusions that rest on comparing runs *against each other* (seed
> variance dominating, the enhancer's language-dropping rewrite, quantization and
> LoRA coverage cleared) are unaffected: every run carried the same defect. The
> one conclusion to re-verify is the **absolute** claim about the audio→video
> pathway ("gate 1.46× stronger in 2.5, pathway cleared") — that measurement was
> taken on the buggy path. See
> [the May investigation's 2026-08-31 updates](/docs/knowledge/investigations/crossmodal-adaln-sigma-swap-2026-05.md).

Reported symptom: the Fluxforge Studio Storyboard (image-mode LipDub, Voxtral
TTS, qint8, VLM enhancement on) produced worse lip tracking on LTX-2.5 than the
same chain had on 2.3 — "ça fonctionnait bien en 2.3, je peux le certifier."

The chain was reproduced headlessly, parameter for parameter, out of the app's
own SwiftData store: real start image, real dialogue, the rendered seed, the
app's exact prompt (`A person speaking on camera, speaking in French saying:
"…"` — its subject sheet was empty), Voxtral 6-bit `fr_female` TTS, 704×448,
frame count from the app's own formula. Then a 6-run matrix isolated checkpoint
× quantization × enhancement, followed by targeted controls.

# What was cleared, with the evidence

- **LoRA coverage.** 1344/1344 layer-pairs fuse on BOTH checkpoints, in every
  one of 8+ runs. The 2.3-only DubIt IC-LoRA loses nothing mechanically on 2.5.
- **The audio→video pathway.** Static comparison of the two transformers: the
  same 576 `audio_to_video_attn` tensors exist in both, q/k/v RMS within 6 %,
  and the gate logits are actually STRONGER in 2.5 (1.46×). The channel that
  drives the mouth is not weakened.
- **Quantization.** qint8 ≈ bf16 on both checkpoints, visually and by the
  face-motion metric. (`lipdub --transformer-quant` was added to make this
  testable from the CLI at all — it was hardcoded bf16.)

# What was found

## 1. Seed variance dominates (the retraction that matters)

The app's rendered seed produced a bad draw on the short line under 2.5
(face-motion L1 8.0 vs the 2.3 reference's 4.2). The same everything with a
different seed: **3.5 — better than the 2.3 reference**, stable across the
clip, and confirmed clean by ear. A longer line under the same "bad" seed was
also fine (different geometry → different initial noise → different draw).

An intermediate conclusion — "`--lora-scale 1.3` brings 2.5 back toward the
2.3 regime" — was measured on ONE seed and looked convincing (8.0 → 5.96,
monotone in scale). The seed control showed variance alone covers that entire
gap. **The recommendation was retracted.** Single-seed comparisons of
generation quality are not findings; the sample size belongs in the finding or
the finding does not get filed (same lesson as the July custom-voice
campaign, learned again).

Practical consequence for integrators: a **re-roll-the-seed affordance** on a
LipDub step is worth more than any scale tuning.

## 2. The 2.5 enhancer drops the language and adds a soundtrack (systematic)

Two runs out of two, deterministic (greedy, fixed seed):

| | Rewrite of `speaking in French saying:` |
|---|---|
| 2.3 (Gemma 3 VLM) | "speaking in French with a clear, professional voice" — language survives |
| 2.5 (Gemma 4) | "speaks in a clear, slightly resonant voice" — **no language anywhere** |

The Gemma 4 rewrite also appends "a subtle, uplifting ambient background music
track", birds and breeze — and an app that muxes the GENERATED audio
(`useGeneratedAudio`) ships those in the clip's soundtrack.

The signature fallback did not catch the language loss: it accepted any
speaking verb + whole-word `in`, and `in a clear voice` satisfies that.
Hardened the same day — the wrapper now requires a capitalized word (a
language, as `DubLanguage.englishName` always produces) after `in`; a false
negative is safe because the fallback re-appends the original signature.

Integration guidance: **disable VLM enhancement on LipDub steps** (the app's
own AutoDub already does, for multi-segment runs).

## 3. The residual 2.5 offset is real but small

Every attention weight in the 2.5 transformer sits ~6 % RMS from its 2.3
counterpart, so the DubIt deltas — trained against 2.3 exactly, and its FFN
biases — apply slightly off-base. Perceptible effect: smaller than one seed
draw. The from-the-root fix, if parity ever matters, is a short fine-tune of
the DubIt LoRA on the 2.5 base with this repo's `train` pipeline; **no DubIt
2.5 exists upstream** (checked 28/08: `Lightricks/LTX-2.3-22b-IC-LoRA-DubIt`
is the only one published).

# Incidental findings, fixed along the way

- `warnOnGenerationMismatch` never fired for LipDub — its only caller was
  `LTXPipeline.fuseLoRA`, and LipDub fuses on the module directly. The one
  adapter published for 2.3 only was the one adapter that never warned. Wired
  into the LipDub path (v0.3.3).
- In image mode, `targetAudioPath` is consumed **directly — no alignment**;
  the silence-aware time-stretch only exists for video mode.
- The `targetAudioPath` path is mono end-to-end: the aligned audio replaces
  the reference as `(samples,)` and `melSpectrogram` duplicates it L=R into
  the stereo-trained encoder — the fake-stereo shape
  [the stereo pitfall](/docs/knowledge/pitfalls/audio-must-stay-stereo.md)
  warns about. A stereo target WAV is downmixed regardless (verified: same
  output to 50.9 dB PSNR). Not implicated in this symptom — the chain worked
  on 2.3 with the same mono path — but it is unfixable from the caller's side
  and worth revisiting.

# Method notes worth keeping

- **Reproduce from the app's own store.** The decisive inputs (empty subject
  sheet → generic "A person" prompt, enhancement enabled, the rendered seed)
  came from SwiftData, not from assumptions — and two of them contradicted
  what a reasonable reconstruction would have guessed.
- **Face-motion L1 measures quantity, not correctness.** It caught the
  restless draws but ranked a user-validated clean clip as the most agitated
  of its triplet. The ear stayed the judge; the metric only screened.
- **Every conclusion needs its seed control.** The scale sweep looked
  monotone and mechanistically plausible, and was still just variance.
