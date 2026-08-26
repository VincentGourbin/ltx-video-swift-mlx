"""Reference forward for the LTX-2.5 duration head.

Runs Lightricks' own ``ltx_core.duration_head.DurationHead`` on the real
checkpoint weights and dumps the numbers our Swift port must reproduce, plus the
seconds-to-frames conversion from ``ltx_pipelines.utils.helpers``.

Why this exists: ``DurationHeadE2ETests`` used to pin the port against a NumPy
re-implementation written during the same effort. That catches a porting slip
but not a shared misreading — if both sides assumed the wrong head count they
would agree and both be wrong. This runs upstream's actual module instead.

The head is a regression model: its output is a number nobody can eyeball. A
mis-ported attention pooler still returns a plausible-looking duration.

Input is deterministic and synthetic (``sin(i/10 + j/100)``), video-only, so the
comparison needs no text encoder and no 26 GB Gemma bundle.

Usage:
    pip install torch safetensors
    git clone https://github.com/Lightricks/LTX-2
    LTX2_ROOT=$PWD/LTX-2 \
      PYTHONPATH=$LTX2_ROOT/packages/ltx-core/src \
      python3 scripts/duration_head_reference.py \
        /path/to/ltx-2.5-duration-head-bf16.safetensors
"""

import ast
import json
import os
import sys

import torch
from safetensors.torch import load_file

from ltx_core.duration_head import DurationHead
from ltx_core.types import SpatioTemporalScaleFactors

LTX2_ROOT = os.environ.get("LTX2_ROOT", "LTX-2")
HELPERS = f"{LTX2_ROOT}/packages/ltx-pipelines/src/ltx_pipelines/utils/helpers.py"

TOKENS, VIDEO_DIM = 8, 4096
PREFIX = "duration_head."
FPS = 24.0

# The same values LTXDurationHead.predictFrameCount defaults to, which are the
# same values upstream's DurationPredictor.__call__ defaults to.
MIN_SECONDS, MAX_SECONDS = 1.0, 20.0


def load_upstream_grid_functions() -> dict:
    """Pull the two grid functions out of upstream's helpers.py, unexecuted.

    Importing ``ltx_pipelines.utils.helpers`` normally drags in the whole media
    stack (``av``, ``OpenImageIO``) for what is two functions of integer
    arithmetic. Rather than reimplement them — which would defeat the point of a
    parity script — parse the file and execute only those definitions. The code
    is upstream's, byte for byte; only its surroundings are skipped.
    """
    if not os.path.exists(HELPERS):
        raise SystemExit(f"Upstream helpers.py not found at {HELPERS}. Set LTX2_ROOT.")
    tree = ast.parse(open(HELPERS).read(), filename=HELPERS)
    wanted = {"snap_frames_to_grid", "seconds_to_clamped_num_frames"}
    picked = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    if {n.name for n in picked} != wanted:
        raise SystemExit(f"helpers.py no longer defines {wanted} — upstream moved them.")

    namespace = {
        "SpatioTemporalScaleFactors": SpatioTemporalScaleFactors,
        "VIDEO_SCALE_FACTORS": SpatioTemporalScaleFactors.default(),
    }
    exec(compile(ast.Module(body=picked, type_ignores=[]), HELPERS, "exec"), namespace)
    return namespace


UPSTREAM = load_upstream_grid_functions()
seconds_to_clamped_num_frames = UPSTREAM["seconds_to_clamped_num_frames"]


def synthetic_tokens(dtype: torch.dtype) -> torch.Tensor:
    """`sin(i/10 + j/100)` — the input DurationHeadE2ETests builds in Swift."""
    rows = torch.arange(TOKENS, dtype=dtype).reshape(TOKENS, 1) * 0.1
    cols = torch.arange(VIDEO_DIM, dtype=dtype).reshape(1, VIDEO_DIM) * 0.01
    return torch.sin(rows + cols).unsqueeze(0)


def run(weights_path: str, dtype: torch.dtype) -> dict:
    raw = load_file(weights_path)
    state = {k[len(PREFIX):]: v.to(dtype) for k, v in raw.items() if k.startswith(PREFIX)}
    if not state:
        raise SystemExit(f"No '{PREFIX}*' tensors in {weights_path}")

    head = DurationHead().to(dtype)
    missing, unexpected = head.load_state_dict(state, strict=False)
    if missing or unexpected:
        # Loud rather than silent: a renamed tensor would otherwise leave a
        # randomly-initialised layer in place and still produce a number.
        raise SystemExit(f"state_dict mismatch. missing={missing} unexpected={unexpected}")

    head.eval()
    with torch.no_grad():
        seconds = head(video_tokens=synthetic_tokens(dtype), audio_tokens=None)

    value = float(seconds.item())
    return {
        "seconds": value,
        "log_duration": float(torch.log(seconds).item()),
        "frames": seconds_to_clamped_num_frames(
            value,
            frame_rate=FPS,
            min_frames=round(MIN_SECONDS * FPS),
            max_frames=round(MAX_SECONDS * FPS),
        ),
    }


def grid_table() -> dict:
    """Pin the seconds-to-frames conversion on its own.

    Includes the four durations measured on real prompts, so the Swift
    `snapToGrid` is checked against upstream on values that actually occur —
    including one that clamps.
    """
    out = {}
    for seconds in [5.15625, 5.28125, 16.875, 19.5, 23.5, 0.5, 20.0]:
        out[f"{seconds}"] = seconds_to_clamped_num_frames(
            seconds,
            frame_rate=FPS,
            min_frames=round(MIN_SECONDS * FPS),
            max_frames=round(MAX_SECONDS * FPS),
        )
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    print(json.dumps({
        "float32": run(sys.argv[1], torch.float32),
        "float64": run(sys.argv[1], torch.float64),
        "seconds_to_frames": grid_table(),
    }, indent=2))
