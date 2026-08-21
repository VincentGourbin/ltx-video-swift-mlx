"""Reference forward pass for the video transformer.

Runs Lightricks' own ``LTXModel`` on a small config with fixed random weights
and fixed inputs, and dumps everything our Swift port needs to reproduce the
result element-wise: the weights, the inputs, and the output velocity.

Small on purpose. The real checkpoint is 22B parameters, which no CPU float32
reference can hold; the arithmetic under test — RoPE, AdaLN-Single, qk-normed
attention, cross-attention, the GELU-approximate FFN, the scale-shift output —
is the same at every width, and a mismatch in any of it shows up here.

    PYTHONPATH=<ltx-core>/src python3 scripts/transformer_reference.py out.safetensors
"""
import sys
import torch
from safetensors.torch import save_file

from ltx_core.model.transformer.model import LTXModel, LTXModelType
from ltx_core.model.transformer.rope import LTXRopeType
from ltx_core.model.transformer.modality import Modality
from ltx_core.components.patchifiers import VideoLatentPatchifier, VideoLatentShape
from ltx_core.tools import VideoLatentTools

OUT = sys.argv[1] if len(sys.argv) > 1 else "transformer_reference.safetensors"
# "ltx2" = the legacy 6-value block (no cross-attention AdaLN);
# "ltx23" = the 9-value block every shipped 2.3/2.5 checkpoint uses.
VARIANT = sys.argv[2] if len(sys.argv) > 2 else "ltx2"
if VARIANT not in ("ltx2", "ltx23"):
    raise SystemExit(f"unknown variant {VARIANT}")

HEADS, HEAD_DIM = 2, 8
INNER = HEADS * HEAD_DIM
CHANNELS, LAYERS, CAPTION_DIM = 4, 2, 16
FRAMES, HEIGHT, WIDTH = 2, 2, 3          # latent grid
TEXT_TOKENS = 5
SIGMA = 0.7
FPS = 24.0

torch.manual_seed(11)

model = LTXModel(
    model_type=LTXModelType.VideoOnly,
    num_attention_heads=HEADS, attention_head_dim=HEAD_DIM,
    in_channels=CHANNELS, out_channels=CHANNELS, num_layers=LAYERS,
    cross_attention_dim=CAPTION_DIM, norm_eps=1e-6,
    positional_embedding_theta=10000.0,
    positional_embedding_max_pos=[20, 2048, 2048],
    rope_type=LTXRopeType("split"),
    apply_gated_attention=(VARIANT == "ltx23"),
    cross_attention_adaln=(VARIANT == "ltx23"),
    use_prompt_adaln_single=(VARIANT == "ltx23"),
).eval().to(torch.float32)

# Default init leaves scale_shift_table empty (uninitialised memory) and most
# norms at 1; fill everything from one seed so the reference is reproducible.
with torch.no_grad():
    for name, param in model.named_parameters():
        param.copy_(torch.randn(param.shape, dtype=torch.float32) * 0.1)

tools = VideoLatentTools(
    patchifier=VideoLatentPatchifier(patch_size=1),
    target_shape=VideoLatentShape(
        batch=1, channels=CHANNELS, frames=FRAMES, height=HEIGHT, width=WIDTH),
    fps=FPS,
)
state = tools.create_initial_state(device=torch.device("cpu"), dtype=torch.float32)

tokens = FRAMES * HEIGHT * WIDTH
latent = torch.randn(1, tokens, CHANNELS, dtype=torch.float32)
context = torch.randn(1, TEXT_TOKENS, CAPTION_DIM, dtype=torch.float32)

video = Modality(
    latent=latent,
    sigma=torch.tensor([SIGMA], dtype=torch.float32),
    timesteps=torch.full((1, tokens), SIGMA, dtype=torch.float32),
    positions=state.positions,
    context=context,
    context_mask=torch.ones(1, TEXT_TOKENS, dtype=torch.float32),
)

# Intermediates, so a mismatch can be localised to a stage instead of a number.
intermediates = {}

def capture(name):
    def hook(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        if isinstance(tensor, torch.Tensor):
            intermediates[name] = tensor.detach().contiguous()
    return hook

model.patchify_proj.register_forward_hook(capture("patchify_proj"))
for i, block in enumerate(model.transformer_blocks):
    block.register_forward_hook(capture(f"block{i}"))
    block.attn1.register_forward_hook(capture(f"block{i}.attn1"))
    block.attn2.register_forward_hook(capture(f"block{i}.attn2"))
    block.ff.register_forward_hook(capture(f"block{i}.ff"))

with torch.no_grad():
    velocity, _ = model(video=video, audio=None, perturbations=None)

tensors = {f"weight.{k}": v.contiguous() for k, v in model.state_dict().items()}
tensors["input.latent"] = latent
tensors["input.context"] = context
tensors["input.positions"] = state.positions.contiguous()
tensors["output.velocity"] = velocity.contiguous()
for name, tensor in intermediates.items():
    tensors[f"stage.{name}"] = tensor
save_file(tensors, OUT)
print(f"wrote {OUT} [{VARIANT}]: {len(tensors)} tensors, velocity {tuple(velocity.shape)}, "
      f"mean|v| {velocity.abs().mean().item():.6f}")
