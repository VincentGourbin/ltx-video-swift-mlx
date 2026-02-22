# Beaver Dam — 8 Configuration Comparison

**Prompt**: *"A beaver building a dam in a forest stream, detailed fur, water splashing, natural lighting"*

All videos generated with: `--enhance-prompt --seed 42 --profile --debug -f 25`

## Performance Summary

| # | Configuration | Model | Steps | CFG | Resolution | Time | Peak RAM | Mean RAM |
|---|---------------|-------|-------|-----|------------|------|----------|----------|
| 1 | Distilled | distilled | 8 | 1.0 | 768x512 | 113s | 52 GB | 27 GB |
| 2 | Dev | dev | 40 | 4.0 | 768x512 | 799s | 55 GB | 27 GB |
| 3 | Dev + LoRA | dev + distilled LoRA | 8 | 1.0 | 768x512 | 102s | 52 GB | 27 GB |
| 4 | Distilled + Upscaler | distilled | 8+3 | 1.0 | 384x256 -> 768x512 | 81s | 52 GB | 27 GB |
| 5 | Dev + Upscaler | dev | 40+3 | 4.0 | 384x256 -> 768x512 | 281s | 55 GB | 27 GB |
| 6 | **Dev + LoRA + Upscaler** | dev + distilled LoRA | 8+3 | 1.0 | 384x256 -> 768x512 | **78s** | 52 GB | 27 GB |
| 7 | Dev 1024 | dev | 40 | 4.0 | 1024x576 | 1455s | 55 GB | 27 GB |
| 8 | Distilled qint8 | distilled (8-bit) | 8 | 1.0 | 768x512 | 99s | 61 GB | **15 GB** |

> **Note on #5**: Dev + Upscaler without distilled LoRA produces visible noise artifacts. This is a model limitation, not a code bug — see [Why two-stage requires distilled LoRA](#why-two-stage-requires-distilled-lora) below.

> **Note on #8**: On-the-fly 8-bit quantization reduces mean denoising RAM by 43% (15 GB vs 27 GB) with faster per-step inference (8.3s vs 10.0s). Peak RAM is higher due to temporary bf16+qint8 coexistence during quantization.

> **Recommended**: #6 (Dev + LoRA + Upscaler) offers the best quality/speed ratio at 78s. For maximum quality at high resolution without two-stage, use #7 (Dev 1024, 1455s). For memory-constrained systems, use #8 (Distilled qint8, 15 GB mean).

*Hardware: Apple Silicon M3 Max 96GB. Peak RAM includes model loading phase.*

## Video Comparison

### Single-stage

<table>
<tr>
<td align="center"><b>1. Distilled</b> (113s)</td>
<td align="center"><b>2. Dev</b> (799s)</td>
<td align="center"><b>3. Dev + LoRA</b> (102s)</td>
</tr>
<tr>
<td><video src="https://github.com/VincentGourbin/ltx-video-swift-mlx/raw/main/docs/examples/beaver-dam/distilled.mp4" controls width="256"></video></td>
<td><video src="https://github.com/VincentGourbin/ltx-video-swift-mlx/raw/main/docs/examples/beaver-dam/dev.mp4" controls width="256"></video></td>
<td><video src="https://github.com/VincentGourbin/ltx-video-swift-mlx/raw/main/docs/examples/beaver-dam/dev-lora.mp4" controls width="256"></video></td>
</tr>
</table>

### Two-stage (with 2x spatial upscaler)

<table>
<tr>
<td align="center"><b>4. Distilled + Upscaler</b> (81s)</td>
<td align="center"><b>5. Dev + Upscaler</b> (281s) *</td>
<td align="center"><b>6. Dev + LoRA + Upscaler</b> (78s)</td>
</tr>
<tr>
<td><video src="https://github.com/VincentGourbin/ltx-video-swift-mlx/raw/main/docs/examples/beaver-dam/distilled-upscaler.mp4" controls width="256"></video></td>
<td><video src="https://github.com/VincentGourbin/ltx-video-swift-mlx/raw/main/docs/examples/beaver-dam/dev-upscaler.mp4" controls width="256"></video></td>
<td><video src="https://github.com/VincentGourbin/ltx-video-swift-mlx/raw/main/docs/examples/beaver-dam/two-stage.mp4" controls width="256"></video></td>
</tr>
</table>

\* *Video 5 exhibits noise artifacts — see [explanation below](#why-two-stage-requires-distilled-lora).*

### High resolution / Quantization

<table>
<tr>
<td align="center"><b>7. Dev 1024x576</b> (1455s)</td>
<td align="center"><b>8. Distilled qint8</b> (99s, 15 GB)</td>
</tr>
<tr>
<td><video src="https://github.com/VincentGourbin/ltx-video-swift-mlx/raw/main/docs/examples/beaver-dam/dev-1024.mp4" controls width="256"></video></td>
<td><video src="https://github.com/VincentGourbin/ltx-video-swift-mlx/raw/main/docs/examples/beaver-dam/distilled-qint8.mp4" controls width="256"></video></td>
</tr>
</table>

### Still Frame Comparison (Frame 12)

| Distilled | Dev | Dev + LoRA |
|-----------|-----|------------|
| ![](frames/distilled_frame_12.png) | ![](frames/dev_frame_12.png) | ![](frames/dev-lora_frame_12.png) |

| Distilled + Upscaler | Dev + Upscaler * | Dev + LoRA + Upscaler |
|----------------------|------------------|------------------------|
| ![](frames/distilled-upscaler_frame_12.png) | ![](frames/dev-upscaler_frame_12.png) | ![](frames/two-stage_frame_12.png) |

| Dev 1024x576 | Distilled qint8 |
|--------------|-----------------|
| ![](frames/dev-1024_frame_12.png) | ![](frames/distilled-qint8_frame_12.png) |

---

## Why Two-Stage Requires Distilled LoRA

The two-stage pipeline's stage 2 uses **distilled refinement sigmas** `[0.909, 0.725, 0.422, 0.0]` — a 3-step schedule designed for the distilled model's learned denoising behavior.

The **dev model** (trained for 40-step CFG denoising with continuous sigma schedules) cannot properly denoise in just 3 steps with these fixed sigmas. The result is visible noise/grain in the output (video 5).

This matches the official Lightricks implementation:
- **HuggingFace Diffusers** always loads distilled LoRA before stage 2, even when using the dev model for stage 1
- **Lightricks HF Space** exclusively uses dev + distilled LoRA for the two-stage pipeline

**Working two-stage configurations:**
- Distilled + Upscaler (#4) — distilled model handles the distilled sigmas natively
- Dev + LoRA + Upscaler (#6) — distilled LoRA adapts dev weights for distilled-style denoising

**For high-resolution dev output without LoRA**, use single-stage at full resolution (#7), though at significantly higher compute cost (1455s vs 78s).

---

## On-The-Fly Quantization

Case #8 demonstrates **on-the-fly 8-bit quantization** (`--transformer-quant qint8`), which replaces all transformer `Linear` layers with `QuantizedLinear` after weight loading.

**Benefits:**
- **43% less RAM during denoising** (15 GB vs 27 GB mean) — enables generation on 16 GB Macs
- **17% faster per-step inference** (8.3s vs 10.0s) — less memory bandwidth needed
- **LoRA compatible** — uses dequant → merge → requant pattern

**Trade-offs:**
- Higher peak RAM during loading (bf16 weights + quantization overhead)
- Potential minor quality degradation from weight quantization

Available options: `--transformer-quant bf16` (default), `qint8` (8-bit), `int4` (4-bit).

---

## Detailed Results

### 1. Distilled

8 steps, no CFG. Fastest single-stage option.

```bash
ltx-video generate "A beaver building a dam in a forest stream, detailed fur, water splashing, natural lighting" \
    -m distilled --enhance-prompt --seed 42 --profile --debug \
    -w 768 -h 512 -f 25 -o docs/examples/beaver-dam/distilled.mp4
```

<video src="https://github.com/VincentGourbin/ltx-video-swift-mlx/raw/main/docs/examples/beaver-dam/distilled.mp4" controls width="512"></video>

<details>
<summary>Profiling output</summary>

```
Denoising (8 steps):                 80.0s
  Step 0: 8.3s
  Step 1: 8.4s
  Step 2: 8.7s
  Step 3: 8.9s
  Step 4: 11.6s
  Step 5: 12.5s
  Step 6: 11.5s
  Step 7: 10.2s
  Average per step:                      10.0s
VAE Decoding:                            4.3s
Pipeline total (excl. loading/export):   84.3s

Peak GPU memory:                         52943 MB
Mean GPU memory (denoising):              26962 MB
```
</details>

---

### 2. Dev

40 steps with CFG 4.0. Best single-stage quality at 768x512.

```bash
ltx-video generate "A beaver building a dam in a forest stream, detailed fur, water splashing, natural lighting" \
    -m dev --enhance-prompt --seed 42 --profile --debug \
    -w 768 -h 512 -f 25 -o docs/examples/beaver-dam/dev.mp4
```

<video src="https://github.com/VincentGourbin/ltx-video-swift-mlx/raw/main/docs/examples/beaver-dam/dev.mp4" controls width="512"></video>

<details>
<summary>Profiling output</summary>

```
Text Encoding (Gemma + FE + Connector): 49.1s
Denoising (40 steps):                 745.5s
  Step 0: 25.3s   Step 10: 16.7s   Step 20: 19.5s   Step 30: 18.2s
  Step 1: 22.5s   Step 11: 16.8s   Step 21: 20.2s   Step 31: 18.2s
  Step 2: 21.3s   Step 12: 16.9s   Step 22: 20.6s   Step 32: 17.7s
  Step 3: 20.5s   Step 13: 16.9s   Step 23: 20.8s   Step 33: 17.6s
  Step 4: 19.1s   Step 14: 17.3s   Step 24: 20.4s   Step 34: 17.6s
  Step 5: 18.4s   Step 15: 17.3s   Step 25: 20.1s   Step 35: 17.6s
  Step 6: 17.6s   Step 16: 17.6s   Step 26: 19.3s   Step 36: 17.4s
  Step 7: 17.7s   Step 17: 18.0s   Step 27: 19.1s   Step 37: 17.3s
  Step 8: 17.4s   Step 18: 18.6s   Step 28: 19.0s   Step 38: 17.5s
  Step 9: 16.8s   Step 19: 18.9s   Step 29: 18.3s   Step 39: 17.4s
  Average per step:                      18.6s
VAE Decoding:                            4.2s
Pipeline total (excl. loading/export):   798.8s

Peak GPU memory:                         54523 MB
Mean GPU memory (denoising):              26969 MB
```
</details>

---

### 3. Dev + Distilled LoRA

Dev model weights with distilled LoRA fused in. 8 steps, no CFG — dev quality at distilled speed.

```bash
ltx-video generate "A beaver building a dam in a forest stream, detailed fur, water splashing, natural lighting" \
    --distilled-lora --enhance-prompt --seed 42 --profile --debug \
    -w 768 -h 512 -f 25 -o docs/examples/beaver-dam/dev-lora.mp4
```

<video src="https://github.com/VincentGourbin/ltx-video-swift-mlx/raw/main/docs/examples/beaver-dam/dev-lora.mp4" controls width="512"></video>

<details>
<summary>Profiling output</summary>

```
Text Encoding (Gemma + FE + Connector): 30.2s
Denoising (8 steps):                 67.5s
  Step 0: 9.7s
  Step 1: 8.4s
  Step 2: 8.4s
  Step 3: 8.3s
  Step 4: 8.4s
  Step 5: 8.2s
  Step 6: 8.1s
  Step 7: 8.1s
  Average per step:                      8.4s
VAE Decoding:                            4.1s
Pipeline total (excl. loading/export):   101.9s

Peak GPU memory:                         52943 MB
Mean GPU memory (denoising):              26962 MB
```
</details>

---

### 4. Distilled + Upscaler

Two-stage: distilled at 384x256, then upscale 2x and refine at 768x512.

```bash
ltx-video generate "A beaver building a dam in a forest stream, detailed fur, water splashing, natural lighting" \
    -m distilled --two-stage --enhance-prompt --seed 42 --profile --debug \
    -w 768 -h 512 -f 25 -o docs/examples/beaver-dam/distilled-upscaler.mp4
```

<video src="https://github.com/VincentGourbin/ltx-video-swift-mlx/raw/main/docs/examples/beaver-dam/distilled-upscaler.mp4" controls width="512"></video>

<details>
<summary>Profiling output</summary>

```
Text Encoding (Gemma + FE + Connector): 5.2s
Denoising (11 steps):                 44.4s
  Stage 1 (384x256, 8 steps):
    Step 0: 3.6s
    Step 1: 2.5s
    Step 2-7: ~2.4s each
  Stage 2 (768x512, 3 steps):
    Step 8: 8.4s
    Step 9: 7.7s
    Step 10: 7.7s
  Average per step:                      4.0s
VAE Decoding:                            4.1s
Pipeline total (excl. loading/export):   53.6s

Peak GPU memory:                         52943 MB
Mean GPU memory (denoising):              27467 MB
```
</details>

---

### 5. Dev + Upscaler (not recommended)

> **Warning**: This configuration produces visible noise artifacts. The dev model cannot properly refine in 3 steps with distilled sigmas. Use #6 (with LoRA) or #7 (single-stage 1024) instead. Kept here for reference to illustrate the limitation.

Two-stage: dev model (40 steps, CFG 4.0) at 384x256, then upscale 2x and refine at 768x512.

```bash
ltx-video generate "A beaver building a dam in a forest stream, detailed fur, water splashing, natural lighting" \
    -m dev --two-stage --steps 40 --guidance 4.0 --enhance-prompt --seed 42 --profile --debug \
    -w 768 -h 512 -f 25 -o docs/examples/beaver-dam/dev-upscaler.mp4
```

<video src="https://github.com/VincentGourbin/ltx-video-swift-mlx/raw/main/docs/examples/beaver-dam/dev-upscaler.mp4" controls width="512"></video>

<details>
<summary>Profiling output</summary>

```
Text Encoding (Gemma + FE + Connector): 9.1s
Denoising (43 steps):                 242.9s
  Stage 1 (384x256, 40 steps with CFG):
    Step 0: 4.3s  ... Step 39: 5.4s
    Average: ~4.8s/step
  Stage 2 (768x512, 3 steps):
    Step 40: 19.6s
    Step 41: 16.4s
    Step 42: 15.4s
  Average per step (all):                5.6s
VAE Decoding:                            4.1s
Pipeline total (excl. loading/export):   256.1s

Peak GPU memory:                         54523 MB
Mean GPU memory (denoising):              27100 MB
```
</details>

---

### 6. Dev + LoRA + Upscaler (Two-Stage)

The standard HuggingFace Space pipeline: dev model with distilled LoRA, 8 steps at 384x256, then upscale 2x and refine with 3 steps at 768x512. **Best quality/speed ratio.**

```bash
ltx-video generate "A beaver building a dam in a forest stream, detailed fur, water splashing, natural lighting" \
    --distilled-lora --two-stage --enhance-prompt --seed 42 --profile --debug \
    -w 768 -h 512 -f 25 -o docs/examples/beaver-dam/two-stage.mp4
```

<video src="https://github.com/VincentGourbin/ltx-video-swift-mlx/raw/main/docs/examples/beaver-dam/two-stage.mp4" controls width="512"></video>

<details>
<summary>Profiling output</summary>

```
Text Encoding (Gemma + FE + Connector): 4.6s
Denoising (11 steps):                 41.9s
  Stage 1 (384x256, 8 steps):
    Step 0: 2.9s
    Step 1-7: ~2.2s each
  Stage 2 (768x512, 3 steps):
    Step 8: 8.3s
    Step 9: 7.4s
    Step 10: 7.7s
  Average per step:                      3.8s
VAE Decoding:                            4.0s
Pipeline total (excl. loading/export):   50.5s

Peak GPU memory:                         52943 MB
Mean GPU memory (denoising):              27467 MB
```
</details>

---

### 7. Dev 1024x576 (Single-Stage)

Dev model at full 1024x576 resolution. No upscaler needed — highest quality output, but significantly slower. Confirms that single-stage dev produces clean results at any resolution.

```bash
ltx-video generate "A beaver building a dam in a forest stream, detailed fur, water splashing, natural lighting" \
    -m dev --enhance-prompt --seed 42 --profile --debug \
    -w 1024 -h 576 -f 25 -o docs/examples/beaver-dam/dev-1024.mp4
```

<video src="https://github.com/VincentGourbin/ltx-video-swift-mlx/raw/main/docs/examples/beaver-dam/dev-1024.mp4" controls width="512"></video>

<details>
<summary>Profiling output</summary>

```
Denoising (40 steps):                 ~1400s
  Average per step:                      ~35s
VAE Decoding:                            ~6s
Pipeline total (excl. loading/export):   ~1455s

Peak GPU memory:                         55000 MB
Mean GPU memory (denoising):              27000 MB
```
</details>

---

### 8. Distilled qint8 (8-bit Quantized)

Distilled model with on-the-fly 8-bit quantization. All transformer `Linear` layers are replaced with `QuantizedLinear` after weight loading. **43% less RAM during inference** with comparable quality.

```bash
ltx-video generate "A beaver building a dam in a forest stream, detailed fur, water splashing, natural lighting" \
    -m distilled --transformer-quant qint8 --enhance-prompt --seed 42 --profile --debug \
    -w 768 -h 512 -f 25 -o docs/examples/beaver-dam/distilled-qint8.mp4
```

<video src="https://github.com/VincentGourbin/ltx-video-swift-mlx/raw/main/docs/examples/beaver-dam/distilled-qint8.mp4" controls width="512"></video>

<details>
<summary>Profiling output</summary>

```
Transformer quantization:                7.1s (one-time, at load)
Text Encoding (Gemma + FE + Connector): 28.5s
Denoising (8 steps):                 66.2s
  Step 0: 8.1s
  Step 1: 7.7s
  Step 2: 8.1s
  Step 3: 8.2s
  Step 4: 8.4s
  Step 5: 8.5s
  Step 6: 8.6s
  Step 7: 8.6s
  Average per step:                      8.3s
VAE Decoding:                            4.1s
Pipeline total (excl. loading/export):   98.9s

Peak GPU memory:                         60531 MB
Mean GPU memory (denoising):              15306 MB
```
</details>

**Comparison with Distilled bf16 (#1):**

| Metric | bf16 (#1) | qint8 (#8) | Difference |
|--------|-----------|------------|------------|
| Mean RAM (denoising) | 27 GB | 15 GB | **-43%** |
| Average step time | 10.0s | 8.3s | **-17%** |
| Peak RAM | 52 GB | 61 GB | +17% (loading overhead) |
| Total pipeline time | 84.3s | 98.9s | +17% (includes quantization) |

---

## Reproduction

All commands can be run sequentially. Models are auto-downloaded on first run (~20-25 GB depending on variant).

To extract comparison frames:
```bash
for f in distilled dev dev-lora distilled-upscaler dev-upscaler two-stage dev-1024 distilled-qint8; do
    ffmpeg -i docs/examples/beaver-dam/$f.mp4 \
        -vf "select=eq(n\,12)" -vframes 1 \
        docs/examples/beaver-dam/frames/${f}_frame_12.png
done
```
