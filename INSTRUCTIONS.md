# LTX-Video-Swift-MLX

## Overview

This Swift Package provides MLX-based video generation using the LTX-2 (Lightricks Text-to-Video 2) model, optimized for Apple Silicon.

## Reference Implementation

The Python implementation to port is available at:
- **Repository**: https://github.com/Acelogic/LTX-2-MLX
- **Models**: https://huggingface.co/Acelogic (ltx-video-2b-* variants)

## Architecture

```
ltx-video-swift-mlx/
├── Package.swift
├── Sources/
│   └── LTXVideo/
│       ├── Pipeline/
│       │   └── LTXPipeline.swift       # Main pipeline orchestrating generation
│       ├── Models/
│       │   ├── DiT3D.swift             # 3D Diffusion Transformer
│       │   ├── VAE3D.swift             # 3D Video VAE (encoder/decoder)
│       │   └── TextEncoder.swift       # T5 text encoder
│       ├── Scheduler/
│       │   └── LTXScheduler.swift      # Flow-matching scheduler
│       └── Utils/
│           ├── VideoUtils.swift        # MP4 encoding, frame handling
│           └── ModelDownloader.swift   # HuggingFace download utilities
├── Tests/
│   └── LTXVideoTests/
└── INSTRUCTIONS.md
```

## Model Variants

| Variant | HuggingFace Repo | RAM Required | Description |
|---------|------------------|--------------|-------------|
| distilledFP8 | Acelogic/ltx-video-2b-distilled-v0.9.7-fp8-mlx | ~12GB | Fastest, FP8 quantized |
| distilled | Acelogic/ltx-video-2b-distilled-v0.9.7-mlx | ~16GB | Balanced speed/quality |
| dev | Acelogic/ltx-video-2b-v0.9.7-mlx | ~25GB | Full quality |

## Key Implementation Details

### Frame Count Constraint

LTX-2 requires frame counts following the formula: `8n + 1`

Valid values: 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97

### Resolution Constraints

- Must be divisible by 32
- Recommended: 512x512, 768x512, 512x768, 832x480, 1024x576

### Pipeline Flow

1. **Text Encoding**: T5 encoder processes the prompt
2. **Latent Generation**: DiT3D generates video latents through diffusion
3. **VAE Decoding**: VAE3D decodes latents to pixel space
4. **Video Export**: Frames assembled into MP4

### Key Classes to Port

#### LTXPipeline

```swift
public class LTXPipeline {
    /// Load models from HuggingFace
    public func loadModels(progressCallback: ((Double, String) -> Void)?) async throws

    /// Generate video from text prompt
    public func generateVideo(
        prompt: String,
        negativePrompt: String?,
        width: Int,
        height: Int,
        numFrames: Int,
        numSteps: Int,
        guidance: Float,
        seed: UInt64,
        onProgress: ((Int, Int) -> Void)?,
        onCheckpoint: ((Int, CGImage) -> Void)?
    ) async throws -> VideoGenerationResult

    /// Clear models from memory
    public func clearAll() async
}
```

#### DiT3D (Diffusion Transformer 3D)

The core model architecture based on transformer blocks with:
- 3D positional encodings (spatial + temporal)
- Cross-attention to text embeddings
- Self-attention within video frames

#### VAE3D (Video Variational Autoencoder)

- Encodes video frames to latent space
- Decodes latents back to pixel space
- Handles temporal consistency across frames

### Dependencies

```swift
dependencies: [
    .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.30.0"),
    .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.0"),
]
```

## Integration with Fluxforge Studio

Once complete, add to Fluxforge Studio via SPM:

```
File > Add Package Dependencies
URL: file:///Users/vincent/Developpements/ltx-video-swift-mlx
Branch: main
```

Update `LTX2Service.swift` to use the real pipeline instead of placeholder code.

## Testing

```bash
swift test
```

## Notes

- The Python reference uses JAX-style operations; MLX has similar APIs
- Pay attention to memory management - video models are large
- Use MLX's lazy evaluation to minimize memory usage
- Consider streaming checkpoints for progress visualization

## Resources

- [MLX Documentation](https://ml-explore.github.io/mlx-swift/docs/)
- [LTX-Video Paper](https://arxiv.org/abs/2408.09890)
- [LTX-2-MLX Python Implementation](https://github.com/Acelogic/LTX-2-MLX)
