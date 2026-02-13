#!/usr/bin/env python3
"""
Generate reference values for Swift validation.

This script uses the Python MLX LTX-2 implementation to generate reference
outputs that can be compared with the Swift implementation.

Usage:
    python scripts/generate_reference.py [--output /path/to/output.json]
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add LTX-2-MLX to path
LTX_MLX_PATH = Path("/tmp/LTX-2-MLX")
if LTX_MLX_PATH.exists():
    sys.path.insert(0, str(LTX_MLX_PATH))

import mlx.core as mx
import mlx.nn as nn
import numpy as np

MODEL_PATH = Path(os.path.expanduser("~/Library/Caches/ltx-video/models/ltx-distilledFP8"))


def test_connector_embedding():
    """Test text encoder connector with a simple input."""
    from safetensors import safe_open
    import torch

    print("=== Testing Connector Embedding ===")

    connectors_path = MODEL_PATH / "connectors.safetensors"
    if not connectors_path.exists():
        return {"error": f"File not found: {connectors_path}"}

    # Load the aggregate_embed weight
    with safe_open(str(connectors_path), framework="pt") as f:
        text_proj_weight = f.get_tensor("text_proj_in.weight")

    # Convert to numpy for comparison
    weight_np = text_proj_weight.float().numpy()

    print(f"text_proj_in.weight shape: {weight_np.shape}")
    print(f"text_proj_in.weight dtype: {weight_np.dtype}")
    print(f"text_proj_in.weight stats: min={weight_np.min():.6f}, max={weight_np.max():.6f}, mean={weight_np.mean():.6f}")

    # Create a simple test input (batch=1, seq_len=16, dim=188160)
    # 188160 = 3840 * 49 (Gemma hidden dim * num layers)
    np.random.seed(42)
    test_input = np.random.randn(1, 16, 188160).astype(np.float32) * 0.1

    # Compute expected output: input @ weight.T (since Linear stores [out, in])
    expected_output = test_input @ weight_np.T

    print(f"Test input shape: {test_input.shape}")
    print(f"Expected output shape: {expected_output.shape}")
    print(f"Expected output stats: min={expected_output.min():.6f}, max={expected_output.max():.6f}, mean={expected_output.mean():.6f}")

    return {
        "test_name": "connector_aggregate_embed",
        "input_shape": list(test_input.shape),
        "output_shape": list(expected_output.shape),
        "input_checksum": float(np.sum(test_input)),
        "output_checksum": float(np.sum(expected_output)),
        "output_mean": float(expected_output.mean()),
        "output_std": float(expected_output.std()),
        "weight_shape": list(weight_np.shape),
        "weight_mean": float(weight_np.mean()),
        "seed": 42
    }


def test_scheduler_sigmas():
    """Test scheduler sigma schedule."""
    print("\n=== Testing Scheduler Sigmas ===")

    # Distilled sigmas from Python reference
    distilled_sigmas = [
        1.0, 0.99375, 0.9875, 0.98125, 0.975,
        0.909375, 0.725, 0.421875, 0.0
    ]

    print(f"Distilled sigmas (8 steps): {distilled_sigmas}")

    return {
        "test_name": "scheduler_sigmas",
        "num_steps": 8,
        "sigmas": distilled_sigmas,
        "is_distilled": True
    }


def test_rope_frequencies():
    """Test 3D RoPE frequency computation."""
    print("\n=== Testing RoPE Frequencies ===")

    # Parameters matching LTX-2
    dim = 4096
    theta = 10000.0
    num_heads = 32
    head_dim = dim // num_heads  # 128

    # Compute frequencies for a small grid
    frames, height, width = 4, 8, 8

    # Create position grid (simplified)
    total_positions = frames * height * width

    # Compute inverse frequencies
    half_dim = head_dim // 2
    inv_freq = 1.0 / (theta ** (np.arange(0, half_dim, dtype=np.float32) / half_dim))

    # Position indices
    positions = np.arange(total_positions, dtype=np.float32)

    # Compute angles
    angles = np.outer(positions, inv_freq)

    # Compute cos and sin
    cos_vals = np.cos(angles)
    sin_vals = np.sin(angles)

    print(f"RoPE parameters: dim={dim}, theta={theta}, heads={num_heads}")
    print(f"Grid: {frames}x{height}x{width} = {total_positions} positions")
    print(f"cos shape: {cos_vals.shape}")
    print(f"cos stats: min={cos_vals.min():.6f}, max={cos_vals.max():.6f}")

    return {
        "test_name": "rope_frequencies",
        "dim": dim,
        "theta": theta,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "grid_shape": [frames, height, width],
        "cos_shape": list(cos_vals.shape),
        "cos_checksum": float(np.sum(cos_vals)),
        "sin_checksum": float(np.sum(sin_vals))
    }


def test_vae_shapes():
    """Test VAE decoder expected shapes."""
    print("\n=== Testing VAE Shapes ===")

    # LTX-2 VAE parameters
    latent_channels = 128
    out_channels = 3

    # Spatial/temporal scaling factors
    time_scale = 8
    height_scale = 32
    width_scale = 32

    # Example latent shape (what transformer outputs)
    batch = 1
    latent_frames = 4  # (33 - 1) / 8 = 4 for 33 output frames
    latent_height = 16  # 512 / 32 = 16
    latent_width = 16   # 512 / 32 = 16

    # Expected output shape
    output_frames = latent_frames * time_scale + 1  # 33
    output_height = latent_height * height_scale    # 512
    output_width = latent_width * width_scale       # 512

    print(f"Latent shape: {batch}x{latent_channels}x{latent_frames}x{latent_height}x{latent_width}")
    print(f"Output shape: {batch}x{out_channels}x{output_frames}x{output_height}x{output_width}")

    return {
        "test_name": "vae_shapes",
        "latent_shape": [batch, latent_channels, latent_frames, latent_height, latent_width],
        "output_shape": [batch, out_channels, output_frames, output_height, output_width],
        "scale_factors": {
            "time": time_scale,
            "height": height_scale,
            "width": width_scale
        }
    }


def test_transformer_block_output():
    """Test a single transformer block output shape."""
    print("\n=== Testing Transformer Block Shapes ===")

    # LTX-2 transformer parameters
    inner_dim = 4096
    num_heads = 32
    head_dim = 128
    context_dim = 3840  # Gemma output dim

    # Example input shapes
    batch = 1
    seq_len = 256  # 4 * 8 * 8 (frames * height * width in latent space)
    context_len = 128  # Text tokens

    print(f"Transformer block config:")
    print(f"  inner_dim: {inner_dim}")
    print(f"  num_heads: {num_heads}")
    print(f"  head_dim: {head_dim}")
    print(f"  context_dim: {context_dim}")
    print(f"Input shapes:")
    print(f"  x: [{batch}, {seq_len}, {inner_dim}]")
    print(f"  context: [{batch}, {context_len}, {context_dim}]")
    print(f"Output shape: [{batch}, {seq_len}, {inner_dim}]")

    return {
        "test_name": "transformer_block_shapes",
        "config": {
            "inner_dim": inner_dim,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "context_dim": context_dim
        },
        "input_shapes": {
            "x": [batch, seq_len, inner_dim],
            "context": [batch, context_len, context_dim]
        },
        "output_shape": [batch, seq_len, inner_dim]
    }


def main():
    parser = argparse.ArgumentParser(description="Generate reference values for Swift validation")
    parser.add_argument("--output", "-o", default=None, help="Output JSON file path")
    args = parser.parse_args()

    print("=" * 60)
    print("LTX-Video-Swift-MLX Reference Generator")
    print("=" * 60)
    print(f"Model path: {MODEL_PATH}")
    print("")

    # Run all tests
    results = {
        "version": "1.0.0",
        "model_path": str(MODEL_PATH),
        "tests": []
    }

    try:
        results["tests"].append(test_connector_embedding())
    except Exception as e:
        print(f"Error in connector test: {e}")
        results["tests"].append({"test_name": "connector_aggregate_embed", "error": str(e)})

    results["tests"].append(test_scheduler_sigmas())
    results["tests"].append(test_rope_frequencies())
    results["tests"].append(test_vae_shapes())
    results["tests"].append(test_transformer_block_output())

    # Output results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    else:
        print("\n" + "=" * 60)
        print("RESULTS (JSON)")
        print("=" * 60)
        print(json.dumps(results, indent=2))

    return results


if __name__ == "__main__":
    main()
