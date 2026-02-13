#!/usr/bin/env python3
"""Validate Swift Gemma3 text encoding against Python MLX reference.

Uses the ALREADY DOWNLOADED model from ~/Library/Caches/models/gemma-3-12b-mlx/
Does NOT re-download anything.

Compares:
1. Token IDs (must be identical)
2. Hidden states from Gemma (49 layers)
3. Feature extractor output
"""

import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

# ─── Config ──────────────────────────────────────────────────────
MODEL_DIR = Path.home() / "Library/Caches/models/gemma-3-12b-mlx"
PROMPT = "A cat walking on the beach"
MAX_LENGTH = 256


def extract_hidden_states(model, input_ids):
    """
    Run Gemma forward pass extracting all hidden states.

    Uses the mlx_lm model structure, faithfully replicating
    the MLXLLM Gemma3Model.__call__ with mask creation.

    Returns: (last_hidden_state, all_hidden_states)
    """
    # Navigate to the inner Gemma model
    if hasattr(model, 'language_model'):
        inner = model.language_model.model
    elif hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        inner = model.model
    else:
        raise RuntimeError(f"Cannot find inner model. Type: {type(model)}, keys: {list(model.keys())}")

    config_file = MODEL_DIR / "config.json"
    with open(config_file) as f:
        raw = json.load(f)
    text_cfg = raw.get("text_config", raw)
    hidden_size = text_cfg.get("hidden_size", 3840)
    sliding_window_pattern = text_cfg.get("sliding_window_pattern", 6)
    window_size = text_cfg.get("sliding_window", 1024)

    # Embed and scale by sqrt(hidden_size)
    h = inner.embed_tokens(input_ids)
    h *= mx.array(hidden_size ** 0.5, dtype=mx.bfloat16).astype(h.dtype)

    all_hidden_states = [h]  # Start with embedding

    # Create attention masks exactly like MLXLLM's Gemma3Model.__call__
    # With no cache, create_attention_mask returns .causal for short sequences
    from mlx_lm.models.gemma3_text import create_attention_mask

    cache = [None] * len(inner.layers)

    # Global mask (no cache, no window)
    global_mask = create_attention_mask(h, cache[sliding_window_pattern - 1])

    # Sliding window mask
    if sliding_window_pattern > 1:
        sliding_window_mask = create_attention_mask(
            h, cache[0], window_size=window_size
        )
    else:
        sliding_window_mask = None

    # Process layers with proper masks
    for i, layer in enumerate(inner.layers):
        is_global = (i % sliding_window_pattern == sliding_window_pattern - 1)
        mask = global_mask if is_global else sliding_window_mask
        h = layer(h, mask, cache[i])

        # For the last layer, apply norm before storing
        if i == len(inner.layers) - 1:
            h = inner.norm(h)

        all_hidden_states.append(h)

        # Periodic eval (every 8 layers, matching Swift)
        if (i + 1) % 8 == 0:
            mx.eval(h)

    mx.eval(h)
    return h, all_hidden_states


def main():
    print("=" * 60)
    print("Python MLX - Gemma3 Text Encoding Validation")
    print("=" * 60)
    print(f"Model: {MODEL_DIR}")
    print(f"Prompt: \"{PROMPT}\"")
    print(f"Max tokens: {MAX_LENGTH}")
    print()

    # ─── Step 1: Load model via mlx_lm (handles quantization) ───
    print("Step 1: Loading model via mlx_lm (local, no download)...")
    from mlx_lm import load as mlx_lm_load
    t0 = time.time()
    model, tokenizer = mlx_lm_load(str(MODEL_DIR))
    t1 = time.time()
    print(f"  Loaded in {t1-t0:.1f}s")
    print(f"  Model type: {type(model).__name__}")
    print()

    # ─── Step 2: Tokenize ────────────────────────────────────────
    print("Step 2: Tokenizing...")
    tokens = tokenizer.encode(PROMPT)
    print(f"  Raw tokens: {len(tokens)}")
    print(f"  Token IDs: {tokens}")

    # Left-pad to MAX_LENGTH using eos_token_id (matching Swift)
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = eos_token_id  # Swift uses eos as pad
    padding_needed = MAX_LENGTH - len(tokens)
    if padding_needed > 0:
        padded_tokens = [pad_token_id] * padding_needed + tokens
    else:
        padded_tokens = tokens[-MAX_LENGTH:]
        padding_needed = 0

    attention_mask = [0.0] * padding_needed + [1.0] * (MAX_LENGTH - padding_needed)

    input_ids = mx.array([padded_tokens])
    mask_array = mx.array([attention_mask])

    print(f"  Padded shape: {input_ids.shape}")
    print(f"  Padding: {padding_needed} pad + {MAX_LENGTH - padding_needed} real")
    print(f"  Pad token ID (eos): {pad_token_id}")
    print(f"  First 5 padded: {padded_tokens[:5]}")
    print(f"  Last 10 padded: {padded_tokens[-10:]}")
    print()

    # ─── Step 3: Forward pass with hidden states ─────────────────
    print("Step 3: Running Gemma forward pass (via mlx_lm model)...")
    t0 = time.time()
    last_hidden, all_hidden_states = extract_hidden_states(model, input_ids)
    t1 = time.time()

    print(f"  Last hidden: {last_hidden.shape} {last_hidden.dtype}")
    print(f"  Hidden states: {len(all_hidden_states)} layers")
    print(f"  Time: {t1-t0:.2f}s")
    print()

    # ─── Step 4: Statistics ──────────────────────────────────────
    mean_val = last_hidden.mean()
    diff = last_hidden - mean_val
    var_val = (diff * diff).mean()
    std_val = mx.sqrt(var_val)
    mx.eval(mean_val, std_val)

    print(f"  Last hidden stats: mean={mean_val.item():.6f}, std={std_val.item():.6f}")
    print()

    # ─── Step 5: Feature extraction (via LTX-2-MLX) ─────────────
    print("Step 4: Feature extraction (using LTX-2-MLX package)...")
    from LTX_2_MLX.model.text_encoder.feature_extractor import GemmaFeaturesExtractorProjLinear

    feature_extractor = GemmaFeaturesExtractorProjLinear(
        hidden_dim=3840,
        num_layers=49,
    )
    # Without trained weights, this is random projection — but we can compare structure
    feature_output = feature_extractor.extract_from_hidden_states(
        hidden_states=all_hidden_states,
        attention_mask=mask_array,
        padding_side="left",
    )
    mx.eval(feature_output)
    print(f"  Feature output: {feature_output.shape}")

    fmean = feature_output.mean()
    fdiff = feature_output - fmean
    fvar = (fdiff * fdiff).mean()
    fstd = mx.sqrt(fvar)
    mx.eval(fmean, fstd)
    print(f"  Feature stats: mean={fmean.item():.6f}, std={fstd.item():.6f}")
    print()

    # ─── Step 6: Dump reference values ───────────────────────────
    print("=" * 60)
    print("REFERENCE VALUES FOR SWIFT COMPARISON")
    print("=" * 60)

    # Token IDs
    print(f"\n[TOKENS]")
    print(f"  raw_count = {len(tokens)}")
    print(f"  raw_tokens = {tokens}")
    print(f"  pad_token_id = {pad_token_id}")
    print(f"  padded_first_5 = {padded_tokens[:5]}")
    print(f"  padded_last_10 = {padded_tokens[-10:]}")

    # Hidden state stats per layer
    print(f"\n[HIDDEN_STATES] count={len(all_hidden_states)}")
    for i in [0, 1, 24, 47, 48]:
        if i < len(all_hidden_states):
            s = all_hidden_states[i]
            m = s.mean()
            rms = mx.sqrt((s * s).mean())
            mx.eval(m, rms)
            print(f"  layer_{i}: mean={m.item():.6f}, rms={rms.item():.6f}, shape={s.shape}, dtype={s.dtype}")

    # Detailed values at specific positions (for numerical comparison)
    print(f"\n[LAST_HIDDEN_DETAILED]")
    last = all_hidden_states[-1]
    mx.eval(last)
    for pos in [0, 249, 255]:
        vals = last[0, pos, :5]
        mx.eval(vals)
        print(f"  pos_{pos}_first5 = {[round(float(v), 6) for v in vals.tolist()]}")

    print(f"  mean = {mean_val.item():.6f}")
    print(f"  std = {std_val.item():.6f}")

    # Save to JSON
    output = {
        "prompt": PROMPT,
        "max_length": MAX_LENGTH,
        "raw_tokens": tokens,
        "pad_token_id": int(pad_token_id),
        "padding_needed": padding_needed,
        "hidden_states_count": len(all_hidden_states),
        "last_hidden_shape": list(last_hidden.shape),
        "last_hidden_dtype": str(last_hidden.dtype),
        "last_hidden_mean": float(mean_val.item()),
        "last_hidden_std": float(std_val.item()),
        "feature_output_shape": list(feature_output.shape),
        "feature_output_mean": float(fmean.item()),
        "feature_output_std": float(fstd.item()),
    }

    # Sample values
    output["sample_values"] = {}
    for pos in [0, 249, 255]:
        vals = last[0, pos, :10]
        mx.eval(vals)
        output["sample_values"][f"pos_{pos}_first10"] = [round(float(v), 6) for v in vals.tolist()]

    # Per-layer stats
    output["layer_stats"] = {}
    for i in range(len(all_hidden_states)):
        s = all_hidden_states[i]
        m = s.mean()
        rms = mx.sqrt((s * s).mean())
        mx.eval(m, rms)
        output["layer_stats"][f"layer_{i}"] = {
            "mean": round(float(m.item()), 6),
            "rms": round(float(rms.item()), 6),
        }

    output_path = Path(__file__).parent / "reference_values.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to: {output_path}")

    print()
    print("✅ Python validation complete!")


if __name__ == "__main__":
    main()
