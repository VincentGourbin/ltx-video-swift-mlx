# LTX-Video-Swift-MLX Examples

Visual examples comparing different generation configurations.

## Beaver Dam

A beaver building a dam in a forest stream — comparing all 6 pipeline configurations.

**[View full comparison with benchmarks](beaver-dam/)**

| Configuration | Preview | Steps | Time |
|---------------|---------|-------|------|
| [Distilled](beaver-dam/#1-distilled) | ![](beaver-dam/frames/distilled_frame_12.png) | 8 | 113s |
| [Dev](beaver-dam/#2-dev) | ![](beaver-dam/frames/dev_frame_12.png) | 40 | 799s |
| [Dev + LoRA](beaver-dam/#3-dev--distilled-lora) | ![](beaver-dam/frames/dev-lora_frame_12.png) | 8 | 102s |
| [Distilled + Upscaler](beaver-dam/#4-distilled--upscaler) | ![](beaver-dam/frames/distilled-upscaler_frame_12.png) | 8+3 | 81s |
| [Dev + Upscaler](beaver-dam/#5-dev--upscaler) | ![](beaver-dam/frames/dev-upscaler_frame_12.png) | 40+3 | 281s |
| [Dev + LoRA + Upscaler](beaver-dam/#6-dev--lora--upscaler-two-stage) | ![](beaver-dam/frames/two-stage_frame_12.png) | 8+3 | 78s |

*All benchmarks on Apple Silicon M3 Max 96GB, 768x512, 25 frames, with prompt enhancement.*
