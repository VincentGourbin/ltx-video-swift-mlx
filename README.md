# LTX-Video-Swift-MLX

> **WORK IN PROGRESS** — This project is under active development. APIs may change, features may be incomplete, and there will be bugs. Use at your own risk.

Swift implementation of [LTX-2](https://github.com/Lightricks/LTX-Video) (Lightricks Text-to-Video 2) optimized for Apple Silicon using [MLX](https://github.com/ml-explore/mlx-swift).

Port of the Python [LTX-2-MLX](https://github.com/Acelogic/LTX-2-MLX) implementation.

## What it does

Generates short videos from text prompts, running entirely on-device on Apple Silicon Macs.

```
ltx-video generate "A golden retriever running through a sunny meadow" \
    --width 768 --height 512 --frames 33 --steps 8
```

## Requirements

- macOS 15+
- Apple Silicon Mac (M1/M2/M3/M4)
- 32GB+ RAM recommended (16GB minimum with distilled model)
- Xcode 16+

## Installation

```bash
git clone https://github.com/VincentGourbin/ltx-video-swift-mlx.git
cd ltx-video-swift-mlx
swift build
```

## Usage

### Download models (first run)

```bash
# Downloads Gemma 3 12B (4-bit) + LTX-2 distilled weights (~20GB total)
ltx-video download
```

### Generate a video

```bash
ltx-video generate "A cat walking on the beach" \
    --width 512 --height 512 --frames 25 --steps 8 \
    --output output.mp4
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--width` | 512 | Video width (must be divisible by 32) |
| `--height` | 512 | Video height (must be divisible by 32) |
| `--frames` | 25 | Number of frames (must be 8n+1: 9, 17, 25, 33...) |
| `--steps` | 8 | Denoising steps |
| `--seed` | random | Random seed for reproducibility |
| `--output` | output.mp4 | Output file path |
| `--profile` | off | Show detailed timing breakdown |

## Architecture

```
Text Prompt
    |
    v
Gemma 3 12B (4-bit quantized) --> 49 hidden states
    |
    v
Feature Extractor + Connector --> text embeddings [B, 256, 3840]
    |
    v
LTX-2 Transformer (48 blocks, flow-matching) --> denoised latent
    |
    v
VAE Decoder (SimpleVideoDecoder) --> video frames
    |
    v
MP4 Export (via ffmpeg)
```

## Performance

Benchmarks on Apple Silicon (768x512, 33 frames, 8 steps):

| Stage | Time |
|-------|------|
| Text Encoding | ~1.5s |
| Denoising (8 steps) | ~75s |
| VAE Decoding | ~8s |
| **Total** | **~85s** |

## Current status

- [x] Gemma 3 text encoding (4-bit quantized)
- [x] Feature extractor + connector
- [x] LTX-2 transformer (48 blocks)
- [x] Flow-matching scheduler (distilled sigmas)
- [x] VAE decoder (validated against Python)
- [x] End-to-end text-to-video generation
- [x] MP4 export
- [x] CLI interface
- [ ] Image-to-video conditioning
- [ ] LoRA support (architecture ready, untested)
- [ ] Performance optimization
- [ ] SwiftUI demo app

## Credits

- [LTX-Video](https://github.com/Lightricks/LTX-Video) by Lightricks
- [LTX-2-MLX](https://github.com/Acelogic/LTX-2-MLX) by Acelogic (Python reference)
- [MLX](https://github.com/ml-explore/mlx-swift) by Apple
- [Gemma 3](https://ai.google.dev/gemma) by Google

## License

MIT License. See [LICENSE](LICENSE).
