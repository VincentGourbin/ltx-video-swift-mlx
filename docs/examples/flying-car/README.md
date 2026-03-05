# Flying Car — Image-to-Video + Audio Benchmarks

**Input Image**: [example_input.png](../example_input.png) (Volkswagen Karmann Ghia)

All videos generated from the same input image with `--audio --seed 42 --audio-gain 0.5` at **1024x640** (3:2 ratio).

## Creative Tests (10 seconds, 241 frames)

Two-stage distilled pipeline (512x320 → upscale 2x → 1024x640), 8+3 steps.

| # | Prompt | Enhancement | Time | File |
|---|--------|-------------|------|------|
| 1 | "make this car fly" | VLM multimodal (sees image) | 1289s | [01-distilled-2stage-enhanced.mp4](01-distilled-2stage-enhanced.mp4) |
| 2 | Back to the Future style (manual) | None | 1243s | [02-distilled-2stage-bttf.mp4](02-distilled-2stage-bttf.mp4) |

<details>
<summary>Bench #1 command</summary>

```bash
ltx-video generate --image docs/examples/example_input.png --audio --enhance-prompt --seed 42 --audio-gain 0.5 \
    -m distilled --two-stage -w 1024 -h 640 -f 241 \
    -o docs/examples/flying-car/01-distilled-2stage-enhanced.mp4 \
    "make this car fly"
```
</details>

<details>
<summary>Bench #2 command & prompt</summary>

```bash
ltx-video generate --image docs/examples/example_input.png --audio --seed 42 --audio-gain 0.5 \
    -m distilled --two-stage -w 1024 -h 640 -f 241 \
    -o docs/examples/flying-car/02-distilled-2stage-bttf.mp4 \
    "The vintage car begins to hover and levitate, its wheels slowly folding underneath the chassis, then it accelerates forward with a futuristic humming sound growing louder, glowing light trails behind it as it launches into the sky like a DeLorean from Back to the Future"
```
</details>

### Video Comparison

<table>
<tr>
<td align="center"><b>1. Enhanced prompt</b> (1289s)</td>
<td align="center"><b>2. Manual BTTF prompt</b> (1243s)</td>
</tr>
<tr>
<td><video src="https://github.com/VincentGourbin/ltx-video-swift-mlx/raw/main/docs/examples/flying-car/01-distilled-2stage-enhanced.mp4" controls width="384"></video></td>
<td><video src="https://github.com/VincentGourbin/ltx-video-swift-mlx/raw/main/docs/examples/flying-car/02-distilled-2stage-bttf.mp4" controls width="384"></video></td>
</tr>
</table>

---

## Performance Benchmarks (5 seconds, 121 frames)

**Prompt**: *"The vintage car starts its engine with a soft purr, then slowly drives away down the sunlit road, tires gently rolling on the pavement, engine humming quietly as it disappears into the distance"*

All with `--audio --seed 42 --audio-gain 0.5 --profile`, 1024x640, I2V.

| # | Configuration | Model | Steps | CFG | Quant | Time | RAM | Notes |
|---|---------------|-------|-------|-----|-------|------|-----|-------|
| 3 | Dev | dev | 40 (×2 passes) | 3.0 | bf16 | **7452s** (~2h) | N/A | Dual CFG passes for audio |
| 4 | Dev qint8 | dev | 40 (×2 passes) | 3.0 | qint8 | **HANG** | — | Bug: audio model loading hangs |
| 5 | Two-stage | distilled | 8+3 | 1.0 | bf16 | **558s** (~9min) | N/A | 512x320 → 1024x640 |
| 6 | Two-stage qint8 | distilled | 8+3 | 1.0 | qint8 | **559s** (~9min) | N/A | 512x320 → 1024x640 |

> **RAM profiling**: Not yet available — `generateVideoWithAudio()` does not wire the `--profile` memory instrumentation. To be added. For reference, the video-only [beaver-dam benchmark](../beaver-dam/) shows ~27 GB mean denoising RAM for bf16 and ~15 GB for qint8 at 768x512.

> **Note on #4**: Dev model with `--transformer-quant qint8 --audio` hangs during audio model loading. This is a known bug — the quantized video transformer conflicts with the dual audio/video transformer loading. Two-stage qint8 (#6) works because it uses the distilled model. To be investigated.

> **Recommended**: Two-stage (#5/#6) offers the best speed at 1024x640. Dev single-stage (#3) produces the highest quality but takes 13x longer due to 40 steps × 2 CFG passes.

*Hardware: Apple Silicon M3 Max 96GB. Audio: mono 24kHz, muxed into MP4.*

### Video Comparison

<table>
<tr>
<td align="center"><b>3. Dev bf16</b> (7452s)</td>
<td align="center"><b>5. Two-stage bf16</b> (558s)</td>
<td align="center"><b>6. Two-stage qint8</b> (559s)</td>
</tr>
<tr>
<td><video src="https://github.com/VincentGourbin/ltx-video-swift-mlx/raw/main/docs/examples/flying-car/03-dev-bf16.mp4" controls width="256"></video></td>
<td><video src="https://github.com/VincentGourbin/ltx-video-swift-mlx/raw/main/docs/examples/flying-car/05-2stage-bf16.mp4" controls width="256"></video></td>
<td><video src="https://github.com/VincentGourbin/ltx-video-swift-mlx/raw/main/docs/examples/flying-car/06-2stage-qint8.mp4" controls width="256"></video></td>
</tr>
</table>

---

<details>
<summary>Bench #3 command (Dev bf16)</summary>

```bash
ltx-video generate --image docs/examples/example_input.png --audio --seed 42 --audio-gain 0.5 --profile \
    -m dev -w 1024 -h 640 -f 121 \
    -o docs/examples/flying-car/03-dev-bf16.mp4 \
    "The vintage car starts its engine with a soft purr, then slowly drives away down the sunlit road, tires gently rolling on the pavement, engine humming quietly as it disappears into the distance"
```
</details>

<details>
<summary>Bench #5 command (Two-stage bf16)</summary>

```bash
ltx-video generate --image docs/examples/example_input.png --audio --seed 42 --audio-gain 0.5 --profile \
    -m distilled --two-stage -w 1024 -h 640 -f 121 \
    -o docs/examples/flying-car/05-2stage-bf16.mp4 \
    "The vintage car starts its engine with a soft purr, then slowly drives away down the sunlit road, tires gently rolling on the pavement, engine humming quietly as it disappears into the distance"
```
</details>

<details>
<summary>Bench #6 command (Two-stage qint8)</summary>

```bash
ltx-video generate --image docs/examples/example_input.png --audio --seed 42 --audio-gain 0.5 --profile \
    -m distilled --two-stage --transformer-quant qint8 -w 1024 -h 640 -f 121 \
    -o docs/examples/flying-car/06-2stage-qint8.mp4 \
    "The vintage car starts its engine with a soft purr, then slowly drives away down the sunlit road, tires gently rolling on the pavement, engine humming quietly as it disappears into the distance"
```
</details>

---

## Known Issues

- **Dev + qint8 + audio**: Hangs during audio model loading. Likely conflict between quantized video transformer and dual audio/video transformer weight loading. Workaround: use bf16 for dev+audio, or use two-stage (distilled) which supports qint8+audio.
