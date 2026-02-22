# LTX-Video-Swift-MLX

Swift implementation of [LTX-2](https://github.com/Lightricks/LTX-Video) (Lightricks Text-to-Video 2) optimized for Apple Silicon using [MLX](https://github.com/ml-explore/mlx-swift).

Generates videos from text prompts, running entirely on-device on Apple Silicon Macs.

## Features

- **Text-to-video generation** with LTX-2 19B parameter model
- **Two model variants**: Distilled (fast, 8 steps) and Dev (quality, 40 steps)
- **Two-stage pipeline**: Half-resolution generation + 2x spatial upscaling + refinement
- **LoRA support**: Distilled LoRA for fast dev-quality generation (8 steps instead of 40)
- **Prompt enhancement**: Gemma 3 12B generates detailed video descriptions from short prompts
- **Memory optimization**: 3-phase pipeline with configurable presets (light/moderate/aggressive)
- **VAE temporal tiling**: Generates long videos (200+ frames) within memory constraints
- **Profiling**: Detailed per-step timing and memory usage reporting

## Requirements

- macOS 15+ (Sequoia)
- Apple Silicon Mac (M1/M2/M3/M4)
- 32 GB+ unified memory recommended (16 GB minimum with distilled model)
- Xcode 16+

## Quick Start

### Build

```bash
git clone https://github.com/VincentGourbin/ltx-video-swift-mlx.git
cd ltx-video-swift-mlx
swift build
```

Or build with Xcode:
```bash
xcodebuild -scheme ltx-video -configuration Release -derivedDataPath .xcodebuild \
  -destination 'platform=macOS' build
```

### Generate a video

```bash
# Distilled model (fast, ~16 GB RAM)
ltx-video generate "A cat walking on the beach" \
    -w 768 -h 512 -f 25 -o output.mp4

# Dev model (best quality, ~25 GB RAM)
ltx-video generate "A cat walking on the beach" \
    -m dev -w 768 -h 512 -f 25 -o output.mp4

# Dev + distilled LoRA (dev quality at distilled speed)
ltx-video generate "A cat walking on the beach" \
    --distilled-lora -w 768 -h 512 -f 25 -o output.mp4

# Two-stage with upscaler (highest quality)
ltx-video generate "A cat walking on the beach" \
    --distilled-lora --two-stage -w 768 -h 512 -f 25 -o output.mp4

# With prompt enhancement
ltx-video generate "A cat on a beach" \
    --enhance-prompt --seed 42 -w 768 -h 512 -f 25 -o output.mp4
```

Models are downloaded automatically on first run from [Lightricks/LTX-2](https://huggingface.co/Lightricks/LTX-2) on HuggingFace.

## Model Variants

| Variant | Steps | CFG | RAM | Speed | Quality |
|---------|-------|-----|-----|-------|---------|
| Distilled | 8 | 1.0 (off) | ~16 GB | Fast | Good |
| Dev | 40 | 4.0 | ~25 GB | Slow | Best |
| Dev + Distilled LoRA | 8 | 1.0 (off) | ~25 GB | Fast | Near-dev |

All variants support the two-stage pipeline (`--two-stage`) which generates at half resolution, upscales 2x with a spatial upscaler, and refines with 3 additional steps.

## CLI Reference

### `ltx-video generate`

| Flag | Default | Description |
|------|---------|-------------|
| `<prompt>` | required | Text prompt describing the video |
| `-o, --output` | `output.mp4` | Output file path |
| `-w, --width` | `512` | Video width (divisible by 32, by 64 for two-stage) |
| `-h, --height` | `512` | Video height (divisible by 32, by 64 for two-stage) |
| `-f, --frames` | `25` | Frame count (must be 8n+1: 9, 17, 25, 33...) |
| `-s, --steps` | model default | Denoising steps (8 distilled, 40 dev) |
| `-g, --guidance` | model default | CFG scale (1.0=off, 4.0 for dev) |
| `-m, --model` | `distilled` | Model variant: `distilled` or `dev` |
| `--seed` | random | Random seed for reproducibility |
| `--distilled-lora` | off | Apply distilled LoRA (forces dev, 8 steps, no CFG) |
| `--two-stage` | off | Two-stage generation with 2x spatial upscaling |
| `--enhance-prompt` | off | Enhance prompt using Gemma 3 generation |
| `--lora` | none | Path to custom LoRA weights (.safetensors) |
| `--lora-scale` | `1.0` | LoRA scale factor |
| `--negative-prompt` | built-in | Negative prompt for CFG |
| `--guidance-rescale` | `0.0` | Guidance rescale phi (0.7 recommended with CFG) |
| `--cross-attn-scale` | `1.0` | Cross-attention scale (>1 = stronger prompt adherence) |
| `--stg-scale` | `0.0` | Spatio-Temporal Guidance scale (0.5 recommended) |
| `--stg-blocks` | `"29"` | STG block indices (comma-separated) |
| `--ge-gamma` | `0.0` | GE velocity correction gamma |
| `--gemma-path` | auto-download | Path to local Gemma model directory |
| `--ltx-weights` | auto-download | Path to unified LTX-2 weights file |
| `--hf-token` | none | HuggingFace token for gated models |
| `--profile` | off | Show detailed timing and memory breakdown |
| `--debug` | off | Enable debug output |
| `--dry-run` | off | Validate config without generating |

### `ltx-video download`

Pre-download model weights from HuggingFace.

| Flag | Default | Description |
|------|---------|-------------|
| `-m, --model` | `distilled` | Model variant to download |
| `--hf-token` | none | HuggingFace token |
| `--force` | off | Force re-download |

### `ltx-video info`

Show version and usage information.

## Examples

See [docs/examples/](docs/examples/) for detailed generation examples with visual comparisons.

## Performance

Benchmarked on Apple Silicon M3 Max 96GB, generating 25 frames at 768x512 with prompt enhancement:

| Configuration | Steps | Time | Notes |
|---------------|-------|------|-------|
| Distilled | 8 | 113s | Fastest single-stage |
| Dev + LoRA | 8 | 102s | Dev quality, distilled speed |
| Distilled + Upscaler | 8+3 | 81s | Fast with 2x upscaling |
| **Dev + LoRA + Upscaler** | **8+3** | **78s** | **Best quality/speed ratio** |
| Dev + Upscaler | 40+3 | 281s | Highest quality |
| Dev | 40 | 799s | Slowest (full-res CFG) |

## Architecture

```
Text Prompt
    │
    ▼
Gemma 3 12B (4-bit quantized) ──► 49 hidden states
    │
    ▼
Feature Extractor + Connector ──► text embeddings [1, 1024, 3840]
    │
    │ ◄── Gemma unloaded to free memory
    ▼
LTX-2 Transformer (48 blocks, flow-matching)
    │
    │  ┌─ Distilled: 8 steps, predefined sigmas
    ├──┤  Dev: 40 steps, token-shifted sigmas + CFG
    │  └─ Dev+LoRA: 8 distilled steps on dev weights
    │
    │ ◄── Transformer unloaded to free memory
    ▼
VAE Decoder (SimpleVideoDecoder) ──► video frames (temporal tiling for long videos)
    │
    ▼
MP4 Export (AVFoundation)
```

**Two-Stage Pipeline:**
```
Stage 1: Generate at half resolution (W/2 x H/2)
    │
    ▼
Spatial Upscaler: Denormalize → 2x upscale → Renormalize → AdaIN
    │
    ▼
Stage 2: Add noise (σ=0.909) → Refine at full resolution (3 steps)
```

## Swift Package Integration

```swift
import LTXVideo

let pipeline = LTXPipeline(model: .distilled)
try await pipeline.loadModels()

let config = LTXVideoGenerationConfig(
    width: 768, height: 512, numFrames: 25
)
let result = try await pipeline.generateVideo(
    prompt: "A sunset over the ocean",
    config: config
)

try await VideoExporter.exportVideo(
    frames: result.frames,
    width: result.width,
    height: result.height,
    to: URL(fileURLWithPath: "output.mp4")
)
```

## Constraints

- **Frame count**: Must be `8n + 1` (valid: 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, ...)
- **Resolution**: Width and height must be divisible by 32 (by 64 for two-stage)
- **Recommended resolutions**: 512x512, 768x512, 512x768, 832x480, 1024x576

## Credits

- [LTX-Video](https://github.com/Lightricks/LTX-Video) by Lightricks
- [HuggingFace Diffusers](https://github.com/huggingface/diffusers) (reference implementation)
- [MLX](https://github.com/ml-explore/mlx-swift) by Apple
- [Gemma 3](https://ai.google.dev/gemma) by Google

## License

MIT License. See [LICENSE](LICENSE).
