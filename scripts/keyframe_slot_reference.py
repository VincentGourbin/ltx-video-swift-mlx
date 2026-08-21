"""Reference positions for generated keyframe slots.

Runs Lightricks' own ``VideoGeneratedKeyframeSlots._slot_positions`` and dumps
the RoPE coordinates our Swift port must reproduce, plus the base sequence's
own grid for contrast. CPU float32: ground truth, not a benchmark.

Upstream keeps positions as ``[start, end)`` patch bounds; this port carries the
middle of that span, so the comparison is against ``(start + end) / 2``.
"""
import json, sys
import torch
from ltx_core.conditioning.types.keyframe_slots import VideoGeneratedKeyframeSlots
from ltx_core.tools import VideoLatentTools
from ltx_core.components.patchifiers import VideoLatentPatchifier, VideoLatentShape

FPS = 24.0
FRAMES, HEIGHT, WIDTH = 3, 2, 3        # latent grid
SLOTS = [0, 24, 96]

tools = VideoLatentTools(
    patchifier=VideoLatentPatchifier(patch_size=1),
    target_shape=VideoLatentShape(batch=1, channels=128, frames=FRAMES, height=HEIGHT, width=WIDTH),
    fps=FPS,
)

out = {}
for index in SLOTS:
    positions = VideoGeneratedKeyframeSlots._slot_positions(index, tools, torch.device("cpu"))
    # (B, 3, T, 2) -> middles
    middles = positions.mean(dim=-1) if positions.ndim == 4 else positions
    out[str(index)] = middles[0].tolist()

print(json.dumps(out))
