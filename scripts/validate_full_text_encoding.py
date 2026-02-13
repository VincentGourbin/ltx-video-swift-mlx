#!/usr/bin/env python3
"""Validate full text encoding pipeline: Gemma → Feature Extractor → Connector.

Uses:
- ALREADY DOWNLOADED Gemma 12B 4-bit from ~/Library/Caches/models/gemma-3-12b-mlx/
- ALREADY DOWNLOADED connectors from ~/Library/Caches/models/ltx-connectors/
- LTX_2_MLX Python package for reference encoder implementation

Compares the full pipeline including real connector weights.
"""

import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

# ─── Config ──────────────────────────────────────────────────────
MODEL_DIR = Path.home() / "Library/Caches/models/gemma-3-12b-mlx"
CONNECTORS_PATH = (
    Path.home()
    / "Library/Caches/models/ltx-connectors/connectors/diffusion_pytorch_model.safetensors"
)
PROMPT = "A cat walking on the beach"
MAX_LENGTH = 256


def extract_hidden_states(model, input_ids):
    """Run Gemma forward pass extracting all hidden states."""
    if hasattr(model, "language_model"):
        inner = model.language_model.model
    elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        inner = model.model
    else:
        raise RuntimeError(f"Cannot find inner model. Type: {type(model)}")

    config_file = MODEL_DIR / "config.json"
    with open(config_file) as f:
        raw = json.load(f)
    text_cfg = raw.get("text_config", raw)
    hidden_size = text_cfg.get("hidden_size", 3840)
    sliding_window_pattern = text_cfg.get("sliding_window_pattern", 6)
    window_size = text_cfg.get("sliding_window", 1024)

    h = inner.embed_tokens(input_ids)
    h *= mx.array(hidden_size**0.5, dtype=mx.bfloat16).astype(h.dtype)

    all_hidden_states = [h]

    from mlx_lm.models.gemma3_text import create_attention_mask

    cache = [None] * len(inner.layers)
    global_mask = create_attention_mask(h, cache[sliding_window_pattern - 1])
    if sliding_window_pattern > 1:
        sliding_window_mask = create_attention_mask(
            h, cache[0], window_size=window_size
        )
    else:
        sliding_window_mask = None

    for i, layer in enumerate(inner.layers):
        is_global = i % sliding_window_pattern == sliding_window_pattern - 1
        mask = global_mask if is_global else sliding_window_mask
        h = layer(h, mask, cache[i])
        if i == len(inner.layers) - 1:
            h = inner.norm(h)
        all_hidden_states.append(h)
        if (i + 1) % 8 == 0:
            mx.eval(h)

    mx.eval(h)
    return h, all_hidden_states


def load_connector_weights(encoder, weights_path):
    """Load connector weights from HuggingFace-format safetensors.

    Keys in the file:
    - text_proj_in.weight -> feature_extractor.aggregate_embed.weight
    - video_connector.* -> embeddings_connector.*
    """
    weights = mx.load(str(weights_path))
    loaded = 0

    # Feature extractor aggregate_embed
    key = "text_proj_in.weight"
    if key in weights:
        encoder.feature_extractor.aggregate_embed.weight = weights[key]
        loaded += 1

    # Connector: learnable_registers
    key = "video_connector.learnable_registers"
    if key in weights:
        encoder.embeddings_connector.learnable_registers = weights[key]
        loaded += 1

    # Connector: transformer blocks
    for block_idx in range(2):
        block = encoder.embeddings_connector.transformer_1d_blocks[block_idx]
        src_prefix = f"video_connector.transformer_blocks.{block_idx}."

        attn_map = {
            "attn1.to_q.weight": ("attn1", "to_q", "weight"),
            "attn1.to_q.bias": ("attn1", "to_q", "bias"),
            "attn1.to_k.weight": ("attn1", "to_k", "weight"),
            "attn1.to_k.bias": ("attn1", "to_k", "bias"),
            "attn1.to_v.weight": ("attn1", "to_v", "weight"),
            "attn1.to_v.bias": ("attn1", "to_v", "bias"),
            "attn1.to_out.0.weight": ("attn1", "to_out", "weight"),
            "attn1.to_out.0.bias": ("attn1", "to_out", "bias"),
            "attn1.norm_q.weight": ("attn1", "q_norm", "weight"),
            "attn1.norm_k.weight": ("attn1", "k_norm", "weight"),
        }

        for pt_suffix, (attn_name, layer_name, param_name) in attn_map.items():
            pt_key = f"{src_prefix}{pt_suffix}"
            if pt_key in weights:
                attn = getattr(block, attn_name)
                layer = getattr(attn, layer_name)
                setattr(layer, param_name, weights[pt_key])
                loaded += 1

        ff_map = {
            "ff.net.0.proj.weight": ("project_in", "proj", "weight"),
            "ff.net.0.proj.bias": ("project_in", "proj", "bias"),
            "ff.net.2.weight": ("project_out", None, "weight"),
            "ff.net.2.bias": ("project_out", None, "bias"),
        }

        for pt_suffix, (l1_name, l2_name, param_name) in ff_map.items():
            pt_key = f"{src_prefix}{pt_suffix}"
            if pt_key in weights:
                l1 = getattr(block.ff, l1_name)
                layer = getattr(l1, l2_name) if l2_name else l1
                setattr(layer, param_name, weights[pt_key])
                loaded += 1

    print(f"  Loaded {loaded} weight tensors from {weights_path.name}")
    return loaded


def main():
    print("=" * 60)
    print("Python - Full Text Encoding Validation (with real weights)")
    print("=" * 60)
    print(f"Model:      {MODEL_DIR}")
    print(f"Connectors: {CONNECTORS_PATH}")
    print(f"Prompt:     \"{PROMPT}\"")
    print(f"Max tokens: {MAX_LENGTH}")
    print()

    # Check files exist
    if not MODEL_DIR.exists():
        print(f"ERROR: Gemma model not found at {MODEL_DIR}")
        sys.exit(1)
    if not CONNECTORS_PATH.exists():
        print(f"ERROR: Connectors not found at {CONNECTORS_PATH}")
        sys.exit(1)

    # ─── Step 1: Load Gemma model ──────────────────────────────────
    print("Step 1: Loading Gemma model...")
    from mlx_lm import load as mlx_lm_load

    t0 = time.time()
    model, tokenizer = mlx_lm_load(str(MODEL_DIR))
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # ─── Step 2: Tokenize with left-padding ────────────────────────
    print("\nStep 2: Tokenizing...")
    tokens = tokenizer.encode(PROMPT)
    eos_token_id = tokenizer.eos_token_id
    padding_needed = MAX_LENGTH - len(tokens)
    padded_tokens = [eos_token_id] * padding_needed + tokens
    attention_mask = [0.0] * padding_needed + [1.0] * (MAX_LENGTH - padding_needed)

    input_ids = mx.array([padded_tokens])
    mask_array = mx.array([attention_mask])
    print(f"  Tokens: {len(tokens)} real + {padding_needed} pad = {MAX_LENGTH}")
    print(f"  Last 10 tokens: {padded_tokens[-10:]}")

    # ─── Step 3: Gemma forward pass ────────────────────────────────
    print("\nStep 3: Gemma forward pass...")
    t0 = time.time()
    last_hidden, all_hidden_states = extract_hidden_states(model, input_ids)
    gemma_time = time.time() - t0
    print(f"  {len(all_hidden_states)} hidden states, time: {gemma_time:.2f}s")

    # ─── Step 4: Create text encoder and load weights ──────────────
    print("\nStep 4: Creating text encoder + loading connector weights...")
    sys.path.insert(0, "/tmp/watermark-venv/lib/python3.12/site-packages")
    from LTX_2_MLX.model.text_encoder.encoder import create_text_encoder

    encoder = create_text_encoder()
    load_connector_weights(encoder, CONNECTORS_PATH)
    mx.eval(encoder.parameters())
    print("  Encoder ready with real weights")

    # ─── Step 5: Feature extractor ─────────────────────────────────
    print("\nStep 5: Feature extraction...")
    t0 = time.time()
    feature_output = encoder.feature_extractor.extract_from_hidden_states(
        hidden_states=all_hidden_states,
        attention_mask=mask_array,
        padding_side="left",
    )
    mx.eval(feature_output)
    print(f"  Shape: {feature_output.shape}, time: {time.time()-t0:.2f}s")

    fe_mean = feature_output.mean().item()
    fe_std = mx.sqrt(((feature_output - feature_output.mean()) ** 2).mean()).item()
    print(f"  Stats: mean={fe_mean:.6f}, std={fe_std:.6f}")

    # Sample values at key positions
    fe_pos0_first10 = feature_output[0, 0, :10].tolist()
    fe_pos249_first10 = feature_output[0, 249, :10].tolist()
    fe_pos255_first10 = feature_output[0, 255, :10].tolist()
    print(f"  pos_0_first10   = {[round(v, 6) for v in fe_pos0_first10]}")
    print(f"  pos_249_first10 = {[round(v, 6) for v in fe_pos249_first10]}")

    # ─── Step 6: Full pipeline (feature extractor + connector) ─────
    print("\nStep 6: Full pipeline (feature extractor + connector)...")
    t0 = time.time()
    output = encoder.encode_from_hidden_states(
        hidden_states=all_hidden_states,
        attention_mask=mask_array,
        padding_side="left",
    )
    mx.eval(output.video_encoding, output.attention_mask)
    pipe_time = time.time() - t0
    print(f"  Video encoding shape: {output.video_encoding.shape}, time: {pipe_time:.2f}s")
    print(f"  Attention mask shape: {output.attention_mask.shape}")

    enc_mean = output.video_encoding.mean().item()
    enc_std = mx.sqrt(
        ((output.video_encoding - output.video_encoding.mean()) ** 2).mean()
    ).item()
    print(f"  Stats: mean={enc_mean:.6f}, std={enc_std:.6f}")

    enc_pos0_first10 = output.video_encoding[0, 0, :10].tolist()
    enc_pos249_first10 = output.video_encoding[0, 249, :10].tolist()
    enc_pos255_first10 = output.video_encoding[0, 255, :10].tolist()
    print(f"  pos_0_first10   = {[round(v, 6) for v in enc_pos0_first10]}")
    print(f"  pos_249_first10 = {[round(v, 6) for v in enc_pos249_first10]}")
    print(f"  pos_255_first10 = {[round(v, 6) for v in enc_pos255_first10]}")

    mask_sum = output.attention_mask.sum().item()
    print(f"  Mask sum: {mask_sum} (expected: {MAX_LENGTH} if all positions valid)")

    # ─── Step 7: Save reference values ─────────────────────────────
    print("\n" + "=" * 60)
    print("SAVING REFERENCE VALUES")
    print("=" * 60)

    reference = {
        "prompt": PROMPT,
        "max_length": MAX_LENGTH,
        "raw_tokens": tokens,
        "pad_token_id": int(eos_token_id),
        "padding_needed": padding_needed,
        "padded_last_10": padded_tokens[-10:],
        # Hidden states
        "hidden_states_count": len(all_hidden_states),
        "last_hidden_mean": float(last_hidden.mean().item()),
        "last_hidden_std": float(
            mx.sqrt(((last_hidden - last_hidden.mean()) ** 2).mean()).item()
        ),
        # Feature extractor output
        "feature_extractor": {
            "shape": list(feature_output.shape),
            "mean": fe_mean,
            "std": fe_std,
            "pos_0_first10": [round(float(v), 6) for v in fe_pos0_first10],
            "pos_249_first10": [round(float(v), 6) for v in fe_pos249_first10],
            "pos_255_first10": [round(float(v), 6) for v in fe_pos255_first10],
        },
        # Connector output (full pipeline)
        "connector_output": {
            "shape": list(output.video_encoding.shape),
            "mean": enc_mean,
            "std": enc_std,
            "pos_0_first10": [round(float(v), 6) for v in enc_pos0_first10],
            "pos_249_first10": [round(float(v), 6) for v in enc_pos249_first10],
            "pos_255_first10": [round(float(v), 6) for v in enc_pos255_first10],
            "mask_sum": int(mask_sum),
        },
    }

    # Per-layer hidden state stats
    reference["layer_stats"] = {}
    for i in [0, 1, 24, 47, 48]:
        if i < len(all_hidden_states):
            s = all_hidden_states[i]
            m = s.mean()
            rms = mx.sqrt((s * s).mean())
            mx.eval(m, rms)
            reference["layer_stats"][f"layer_{i}"] = {
                "mean": round(float(m.item()), 6),
                "rms": round(float(rms.item()), 6),
            }

    output_path = Path(__file__).parent / "reference_full_text_encoding.json"
    with open(output_path, "w") as f:
        json.dump(reference, f, indent=2)
    print(f"  Saved to: {output_path}")

    print("\n✅ Full text encoding validation complete!")


if __name__ == "__main__":
    main()
