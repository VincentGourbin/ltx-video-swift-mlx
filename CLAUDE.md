# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Swift Package implementing LTX-2 (Lightricks Text-to-Video 2) video generation optimized for Apple Silicon using MLX. This is a port of the Python implementation at https://github.com/Acelogic/LTX-2-MLX.

## Build & Test Commands

```bash
# Build the package
swift build

# Run tests
swift test

# Run a specific test
swift test --filter testVersion
```

## Architecture

The package follows a modular design:

- **Pipeline/** - `LTXPipeline` orchestrates the generation flow: text encoding → latent generation → VAE decoding → video export
- **Models/** - Neural network components:
  - `DiT3D` - 3D Diffusion Transformer (spatial + temporal attention)
  - `VAE3D` - Video VAE for encoding/decoding frames to/from latent space
  - `TextEncoder` - T5 encoder for text prompts
- **Scheduler/** - `LTXScheduler` implements flow-matching diffusion sampling
- **Utils/** - Video encoding (MP4) and HuggingFace model downloading

## Key Dependencies

- `mlx-swift` (v0.30.0+) - MLX framework for Apple Silicon ML
- `swift-transformers` (v1.1.0+) - HuggingFace Transformers for Swift

MLX products used: `MLX`, `MLXNN`, `MLXRandom`, `Transformers`

## Model Constraints

**Frame count**: Must be `8n + 1` (valid: 9, 17, 25, ..., 481). Max 481 = 20 s at 24 fps, the RoPE positional range (`maxPos[0]` = 20 s).

**Resolution**: Must be divisible by 32. Recommended: 512x512, 768x512, 512x768, 832x480, 1024x576

**Model variants** (from HuggingFace Acelogic):
- `distilledFP8` (~12GB RAM) - Fastest, FP8 quantized
- `distilled` (~16GB RAM) - Balanced
- `dev` (~25GB RAM) - Full quality

## Engineering Knowledge Base

`docs/knowledge/` is an [OKF](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md)
bundle of measured, durable engineering knowledge: benchmarks, decisions with
rationale, verified pitfalls, investigation records, diagnostic playbooks.
**Read [docs/knowledge/index.md](docs/knowledge/index.md) before debugging
performance, LipDub quality, weight loading, or build/test tooling** — most
traps in those areas have already been root-caused and documented there.
When you root-cause something new or measure something durable, add a concept
to the bundle (and a line to its `log.md`) instead of leaving it in a PR
description.

## Implementation Notes

- Use MLX lazy evaluation to minimize memory usage
- The Python reference uses JAX-style operations; MLX has similar APIs
- Target platform: macOS 26.3+ (Tahoe, Apple Silicon)
- Reference: INSTRUCTIONS.md contains detailed API signatures and implementation guidance
