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

## Fed back as anchors

`ltx-video interpolate --anchors slots.safetensors` reuses them in the temporal
round, which is what they are for. Against the same run without them (seed 3,
121 → 241 frames, single window):

| | source anchors + slots | source anchors only | slots only |
|---|---|---|---|
| worst inter-frame spike | z 2.94 | z 5.76 | **z 2.01** |
| mean fidelity vs source | **24.86 dB** | 23.98 dB | 22.64 dB |
| worst frame | **18.01 dB** | 16.97 dB | 16.65 dB |
| mean inter-frame delta | 0.0105 | 0.0105 | **0.0098** |

`interpolate-anchored-241f.mp4` is the first column.

The three columns say something the first two alone did not: **source anchors
buy fidelity, generated keyframes buy continuity.** Two slots alone (with
`--anchor-every 0`) give the smoothest result of the three — lowest delta and
lowest spike — while drifting furthest from the source, because two anchors
constrain far less than eight. Using both is what wins on fidelity, and it still
halves the spike.

Cost, three runs at 768x512 / 121 frames on an M3 Max: 187.6 s and 229.0 s
without slots, 242.0 s with two. Two slots add 12.5% tokens, and the spread
between two identical baselines (41 s) is as large as the difference — the
honest statement is that the cost is small and not separable from run-to-run
variance at this sample size.
