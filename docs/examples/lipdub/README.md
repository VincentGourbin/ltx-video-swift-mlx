# LipDub — Lip-sync a reference video to a new prompt

Generate a lip-synced video using Lightricks' LipDub IC-LoRA. The pipeline takes a
**reference video with audio**, keeps the audio track, and synthesizes a new
matching video conditioned on:

- The reference video frames (via the IC-LoRA video reference pattern — appended
  at downscaled resolution, RoPE positions scaled to the target coordinate space)
- The reference audio (via appended audio tokens with **negative** RoPE positions
  so the reference sits "before" the target in time)

This is the Swift/MLX port of Lightricks
[`packages/ltx-pipelines/src/ltx_pipelines/lipdub.py`](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/src/ltx_pipelines/lipdub.py).

## How it works

Two-stage distilled pipeline, both stages dual-stream (video + audio):

**Stage 1** (half-res, 8 steps):
- Encode reference video at `halfRes / reference_downscale_factor` via VAE
- Encode reference audio via AudioVAE
- Build a `VideoReferenceContext` (multi-frame, spatial scale by `downscale_factor`)
- Build an `AudioReferenceContext` (negative RoPE shift: last reference token's
  END time sits at exactly `-0.04`s before the target audio at `t=0`)
- Denoise dual-stream, both refs appended via `runDenoiseStep`

**Stage 2** (full-res, 3 refinement steps):
- Re-encode reference video at `fullRes / reference_downscale_factor`
- Re-build audio reference from the **Stage 1 denoised audio** (Python
  `lipdub.py:264` pattern — the audio is essentially propagated forward)
- Video is re-noised from the upscaled Stage 1 latent; audio is **frozen**
  (the `vel.audio` returned by `runDenoiseStep` is discarded)

Final audio output = decoded Stage 1 audio latent (matches Python lipdub.py:264).

The `reference_downscale_factor` (typically `2`) is read from the IC-LoRA's
safetensors metadata at load time — no need to pass it explicitly.

## CLI

```bash
ltx-video lipdub "a person speaking the dialogue" \
    --reference-video path/to/reference.mp4 \
    -w 768 -h 512 -f 121 \
    --seed 42 \
    -o lipdub_output.mp4
```

### Options

| Flag | Default | Description |
|---|---|---|
| `<prompt>` | — | Text description of the output |
| `--reference-video` | — | Path to source `.mp4` (frames + audio both extracted) |
| `-w` / `-h` | 768 / 512 | Output resolution (must be divisible by 64) |
| `-f` | 121 | Frame count (must be `8n+1`; should match reference video) |
| `--seed` | random | Seed for reproducibility |
| `--reference-strength` | `1.0` | Video reference conditioning strength |
| `--lora` | auto-download | Override LipDub IC-LoRA path |
| `--hf-token` | auto (see below) | HuggingFace token for gated downloads |

### HuggingFace authentication

The LipDub IC-LoRA is hosted on a gated HF repo
([`Lightricks/LTX-2.3-22b-IC-LoRA-LipDub`](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-LipDub)).
You must:

1. Accept the license on HuggingFace
2. Authenticate via one of:
   - `huggingface-cli login` (writes `~/.cache/huggingface/token` — picked up automatically)
   - `export HF_TOKEN=hf_xxx`
   - `--hf-token hf_xxx` on the CLI

The same token is also used for the LTX-2.3 base weights (also gated).

## Constraints

- **Audio is mandatory** — the reference video must have an audio track.
- **Width / height divisible by 64** (and by `reference_downscale_factor`,
  typically 2, after halving for Stage 1).
- **Frame count `8n+1`** — the reference video's frame count snapped down.
- **`--image`, `--keyframe`, `--video` are incompatible** with `lipdub` — this
  is a dedicated pipeline, not a flag on `generate`.

## Performance

Measured on M3 Max 96 GB:

| Frames | Resolution | Time | Notes |
|---|---|---|---|
| 9 | 768×512 | 164 s | Smoke test (the reference VAE encode dominates at small frame counts) |
| 121 | 768×512 | (untested) | Expect ~15–20 min |

The video reference is re-encoded **twice** (once per stage at the matching
downscaled resolution), so VAE encoder time is roughly 2× compared to other
pipelines. The audio reference is encoded once.
