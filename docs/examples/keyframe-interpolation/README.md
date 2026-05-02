# Keyframe Interpolation — LTX-2.3 Distilled Two-Stage Pipeline

Constrain video generation to pass through one or more keyframe images at chosen
pixel positions. Generalizes single-frame I2V to arbitrary first / middle / last
frame conditioning, in any combination.

## How It Works

Each keyframe is VAE-encoded into a single latent slot, then placed at the latent
frame matching its pixel index (latent stride = 8, so pixel 0 → slot 0, pixels
1–8 → slot 1, …, pixels 113–120 → slot 15, etc.). At every denoising step:

1. The per-token timestep mask sets `σ = 0` at every keyframe slot, `σ = sigma`
   elsewhere — so the transformer sees keyframe slots as already-clean references
   while denoising the rest.
2. The full latent is stepped by the scheduler.
3. Keyframe slots are re-injected with the clean encoded latent, overwriting
   the wrong update applied by the scalar-sigma scheduler at those positions.

This is mathematically equivalent to the previous frame-0-only I2V path when a
single keyframe is placed at pixel 0, so the existing `--image` flag is preserved
as syntactic sugar.

## CLI

```bash
ltx-video generate "<prompt>" \
    --keyframe path/to/first.png:0 \
    --keyframe path/to/last.png:120 \
    -w 768 -h 512 -f 121
```

`--keyframe PATH:FRAME_IDX[:STRENGTH]` is repeatable. `--image PATH` is shorthand
for `--keyframe PATH:0` (the two flags are mutually exclusive).

### Constraints

- Pixel `FRAME_IDX` must be in `[0, numFrames - 1]`.
- Two keyframes within the same 8-pixel-frame group share the same latent slot
  and are rejected (e.g. `pixel 1` and `pixel 8` both map to latent slot 1).
- `STRENGTH` must be exactly `1.0` (hard injection). Values `!= 1.0` are
  rejected by the validator until soft conditioning is wired through in a
  future PR.
- Multi-keyframe combined with `--video` (retake) is not supported.

### Behavioral note vs. legacy `--image`

When `imageCondNoiseScale == 0` (the default), the new keyframe path is
mathematically equivalent to the previous frame-0-only frame-skip Euler step.

When `imageCondNoiseScale > 0`, there is one subtle divergence: the new code
unconditionally re-injects the **clean** keyframe latent at the end of every
denoising step, whereas the previous code left the noised pre-step injection
in place at the end of stage 1. The new behavior produces a cleaner final
keyframe slot — arguably more correct, but a possible visible difference for
users who relied on the old behavior with non-zero noise scale.

---

## Validated Use Cases

All examples below use the same 768×512 reference image of a red Citroën 2CV
(see `docs/examples/image-to-video/input_768x512.png`) and seed 42 on
M3 Max 96 GB.

### 1. Smoke test — single keyframe at pixel 0 (regression of `--image`)

Same image as both `--image input.png` and `--keyframe input.png:0` — the new
keyframe code path produces identical output to the legacy I2V path.

```bash
ltx-video generate "A chic woman walks towards a red vintage car..." \
    --image docs/examples/image-to-video/input_768x512.png \
    -w 768 -h 512 -f 9 --seed 42 --audio \
    -o regression-i2v-audio.mp4
```

| Parameter | Value |
|-----------|-------|
| Frames | 9 (0.4 s) |
| Audio | Yes |
| Inference time | 60.6 s (vs 58 s baseline pre-PR) |

---

### 2. Loop — same keyframe at first AND last frame

Forces the video to start AND end on the reference image. Useful for
seamless loops.

```bash
ltx-video generate \
    "The red vintage car's wheels fold up underneath, flames burst from the rear, \
     and the car lifts off the ground into the sky like in Back to the Future. \
     It soars through the clouds, then arcs back down, descending gracefully and \
     landing softly in the exact same spot where it started. Cinematic." \
    --keyframe docs/examples/image-to-video/input_768x512.png:0 \
    --keyframe docs/examples/image-to-video/input_768x512.png:120 \
    -w 768 -h 512 -f 121 --seed 42 \
    -o loop.mp4
```

| Parameter | Value |
|-----------|-------|
| Frames | 121 (5 s) |
| Keyframes | pixel 0 (slot 0) + pixel 120 (slot 15) |
| Audio | No |
| Inference time | 432 s |

---

### 3. Last-frame anchor — free start, fixed end

The model invents the opening freely, then converges to the reference image at
the final frame. Great for "this is where the story ends" framing.

```bash
ltx-video generate \
    "A red vintage Citroën 2CV soars through cloudy skies, then begins descending \
     gracefully. It glides down through the clouds, the engine roaring like a \
     lawnmower, the wheels emerging from underneath. The car touches down softly \
     on a quiet country road and rolls gently to a stop. Cinematic." \
    --keyframe docs/examples/image-to-video/input_768x512.png:120 \
    -w 768 -h 512 -f 121 --seed 42 --audio \
    -o landing.mp4
```

| Parameter | Value |
|-----------|-------|
| Frames | 121 (5 s) |
| Keyframes | pixel 120 only (last, slot 15) |
| Audio | Yes (dual video/audio denoising) |
| Inference time | 736 s |

---

### 4. Mid-keyframe — free start, fixed middle, free end

The reference image is anchored at the mid-point. The model freely generates
the lead-in (frames 0–119) and the continuation (frames 121–240). Maximum
narrative flexibility.

```bash
ltx-video generate \
    "A red vintage Citroën 2CV descends from cloudy skies above a quiet country \
     road, decelerating as it approaches the ground. It glides down gracefully \
     and lands softly, parking exactly here. After a brief moment of stillness, \
     its wheels fold up underneath, flames burst from the rear, and the car \
     lifts off again, soaring back up into the dramatic sunset clouds. Cinematic." \
    --keyframe docs/examples/image-to-video/input_768x512.png:120 \
    -w 768 -h 512 -f 241 --seed 42 --audio \
    -o mid-landing-takeoff.mp4
```

| Parameter | Value |
|-----------|-------|
| Frames | 241 (10 s) |
| Keyframes | pixel 120 only (middle, slot 15 of 31) |
| Audio | Yes |
| Inference time | 1496 s (~25 min) |

---

## Out of Scope (Future Work)

- Soft conditioning via `STRENGTH < 1.0` — currently clamped to 1.0 (hard injection).
- Combining keyframes with retake (`--video`).
- Per-keyframe CRF / per-keyframe prompt enhancement.

## Hardware

- Apple Silicon M3 Max 96 GB
- macOS 26.4 (Tahoe)
- Release build via `xcodebuild -scheme ltx-video -configuration Release`
