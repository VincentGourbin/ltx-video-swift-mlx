# Generated keyframe slots — validation artefacts

`ltx-video generate --model 2.5-distilled -w 768 -h 512 -f 121 --seed 42 \
    --keyframe-slot 40 --keyframe-slot 80 --slots-out slots.safetensors`

The run wrote `[1, 128, 2, 16, 24]` — two slots, one latent frame each, at the
target's spatial grid — which is the shape the layout predicts.

| file | what it is |
|---|---|
| `generate-with-slots-121f.mp4` | the clip the run produced |
| `slot-40-decoded.png` | the slot at pixel frame 40, decoded on its own |
| `clip-frame-40.png` | frame 40 of that clip, for comparison |
| `control-frame-40-roundtrip.png` | **the control**: `clip-frame-40.png` encoded and decoded as a one-frame clip |

## Reading them

The slot holds the scene at its own frame: same car, same side-profile framing,
same moment of the lift, dust under the wheels.

It is also visibly softer and darker than the clip's frame — and so is the
control, which is the *real* frame put through the same one-latent-frame round
trip. That is the VAE's behaviour on a latent spanning one pixel frame instead
of eight, not the slot's. Quantitatively:

| comparison | PSNR |
|---|---|
| slot 40 vs clip frame 40 | 16.18 dB |
| **control** (frame 40 round-tripped) vs clip frame 40 | **16.01 dB** |
| slot 40 vs clip frame 80 | 12.45 dB |

The slot is as close to the frame as the VAE's own round trip of that frame is,
and clearly further from a frame two seconds away. Slots are consumed as latents
by later stages and never decoded in normal use; the decode exists to make them
inspectable.

Cost, three runs at 768x512 / 121 frames on an M3 Max: 187.6 s and 229.0 s
without slots, 242.0 s with two. Two slots add 12.5% tokens, and the spread
between two identical baselines (41 s) is as large as the difference — the
honest statement is that the cost is small and not separable from run-to-run
variance at this sample size.
