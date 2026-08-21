# LTX-2.5 bring-up — validated results (August 2026)

Every clip here was generated on-device by this package and user-validated.
All runs: 768×512, bf16 unless noted, conditioning image `conditioning-frame.png`
(frame 0 of an LTX-2.5 reference generation). The measured findings behind these
clips live in [`docs/knowledge/`](../../knowledge/index.md).

| File | What it evidences | Settings |
|---|---|---|
| `generate-25-i2v-audio-337f.mp4` | LTX-2.5 i2v two-stage with audio, at the duration the checkpoint's own head predicts (14.04 s vs 13.375 s from the reference service) | 2.5-distilled, 337f, seed 42, BigVGAN vocoder 48 kHz |
| `generate-23-control.mp4` | LTX-2.3 two-stage unchanged after the 2.5 work — same prompt/image/seed as the 2.5 run | distilled (2.3), 121f, seed 42 |
| `lora23-arcshot-on-25.mp4` | A 2.3-trained camera LoRA (attention-only, 384 modules) produces its arc on the 2.5 checkpoint; prompt requested no motion | 2.5-distilled, 121f, seed 7 |
| `transition-23.mp4` / `transition-25.mp4` | A 2.3-trained transition LoRA (attention + FFN, 576 modules) works on both generations — same keyframes, prompt, seed; only the checkpoint differs. FFN is the one layer where 2.5 diverges structurally (bias-free), so this closes the cross-generation LoRA question behaviourally | joyfox/LTX-2.3-Transition-LORA, keyframes `transition-video-A` last frame → `transition-video-B` first frame, trigger `zhuanchang`, seed 300 |
| `transition-video-A.mp4` / `transition-video-B.mp4` | The two source clips for the transition | distilled (2.3), seeds 100 / 200 |
| `transition-compare-strip.png` | Frames 0/30/60/90/120, top 2.3 / bottom 2.5 |  |
| `upscale-source-384x256.mp4` → `upscale-25-stage1.mp4` → `upscale-25-final-768x512.mp4` | The pixel spatial upscaler chain on 2.5: 8-step stage 1 with the IC-LoRA at source resolution, latent upscale, 3-step refinement. Subject identity holds end-to-end — see the [stage-2 decision](../../knowledge/decisions/iclora-stage2-keeps-adapter-and-reference.md) for why adapter and reference both stay active | 2.5-distilled, x2 1.0 adapter, seed 42 |
| `lipdub-23-bigvgan-vocoder.mp4` | LipDub (2.3) through the checkpoint's real vocoder — 48 kHz, BigVGAN + bandwidth extension | distilled (2.3), 121f, seed 42 |
| [`keyframe-slots/`](keyframe-slots/) | Generated keyframe slots: the clip, a slot decoded on its own, the matching frame, and the one-frame VAE round trip that is the control for reading them | 2.5-distilled, 121f, seed 42, slots at 40 and 80 |

## Distilled vs dev quality series (August 15-16)

Four runs, one comparison: same seed (42), same conditioning image, same
enhanced prompt (the reference service's enhancement of the 2CV take-off
scene), same 337 frames (14.04 s, auto-predicted). Only the checkpoint and
sampling recipe change. Full protocol notes in
[`docs/knowledge/`](../../knowledge/index.md).

| File | Config | Wall time | What it evidences |
|---|---|---|---|
| `series-25-distilled-two-stage.mp4` | 2.5-distilled, two-stage 8 steps, audio | 31 min | The production path, and a 41 dB reproduction of the July validated run |
| `series-25-dev-lora450.mp4` | 2.5-dev + distilled LoRA 450, two-stage, audio | 26 min | The LoRA turns dev into distilled (1660 layer-pairs incl. audio) |
| `series-25-dev-no-lora-control.mp4` | 2.5-dev, LoRA at scale 0, two-stage, audio | 26 min | Negative control: what the 8-step schedule produces without distillation |
| `series-25-dev-full-30steps.mp4` | 2.5-dev single-stage, 30 steps, CFG 3.0 + STG [28] + rescale 0.7, video-only | 5 h 10 | The dev checkpoint's quality ceiling — and the run that surfaced the [empty-negative pitfall](../../knowledge/pitfalls/empty-cfg-negative-erases-the-prompt.md): with the port-inherited "" negative this exact run lost the whole choreography |

## Convolutional vs diffusion decoder (August 20)

Same latent, seed 42, 121 frames at 768x512, one variable: which decoder
renders the pixels. The prompt is `bench-prompt-2cv.txt`, the conditioning
image `conditioning-frame.png` — both versioned here so the comparison is
reproducible.

| File | Decoder | Wall time | Sharpness | HF energy | Contrast |
|---|---|---|---|---|---|
| `decoder-conv-121f.mp4` | convolutional (default) | **195 s** | 3.51 | 1.56 | 53.2 |
| `decoder-diffvae-121f.mp4` | diffusion (`--diffvae`) | 395 s | 3.22 | 1.43 | 51.6 |

PSNR between them is 32 dB: globally alike, differing only in fine detail —
and the diffusion decoder measures marginally *lower* on every sharpness
proxy, for twice the time. Colour differs by ~1%: it lifts blue and green
slightly (so it is fractionally *cooler*, not warmer) and desaturates the
car's red by 0.01. At this resolution and duration it buys nothing
perceptible, which is why it stays opt-in.

The port itself is faithful — pinned element-wise at ~1e-6 against the
reference implementation, stage by stage — so this is a property of the
model at this scale, not of the implementation.

## Temporal interpolation (August 20)

`interpolate-2cv-48fps.mp4` — the 121-frame bench clip densified to **241
frames at 48 fps**, same 5.02 s duration, via the latent temporal upsampler
plus an ancestral refinement anchored on the source's own frames
(`ltx-video interpolate`, seed 42, defaults). 663 s.

Motion spreads as intended: mean inter-frame difference drops to 0.60 of the
source's (0.50 would be a perfect split), and sharpness *rises* (3.83 → 4.90)
rather than falling the way an averaging interpolation would.

Anchoring is what makes upstream's 0.975 noise level usable: without it the
same run redraws the car (identity 14.4 dB against 26.7 dB anchored), and
lowering the level instead only reaches 20.1 dB with less motion and less
sharpness — see the
[renoise pitfall](../../knowledge/pitfalls/renoise-level-needs-its-anchor.md).

### Tiled: 337 → 673 frames

`interpolate-tiled-673f.mp4` — the 14 s series clip at **673 frames / 48 fps**,
denoised in 3 overlapping tiles (35 min). Identity against the source holds at
27-28 dB across the clip, dipping to 24.3 dB only where the content itself
moves fastest.

The tiled defaults differ from the single-window ones for a reason worth
keeping: tiles renoise independently, so at the single-window level (0.975)
each rebuilds its own subject and the seams stop agreeing — 13.4 dB identity
at a seam, and a visibly different car afterwards. Tiled runs therefore
default to 0.725 with dense anchoring. See
[smoothness metrics miss identity drift](../../knowledge/pitfalls/smoothness-metrics-miss-identity-drift.md),
which is also how that defect was *missed* by the first round of measurement.

Known cosmetic caveats, deliberate (they document real behaviour):
- The transition clips open on a dark blurred close-up: that is genuinely the
  last frame of video A, which drifts at its end. Keyframes anchor only the
  endpoints — check hinge frames before using them as anchors.
- `transition-video-A`'s car is not held mid-transition for the same reason:
  the frame-0 anchor was unreadable, so mid-clip content came from the prompt.
