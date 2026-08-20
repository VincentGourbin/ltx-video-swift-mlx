# Prompt-enhancer bench

Four raw prompts, each enhanced two ways: by the reference service
(`diffusers/LTX-2.4-Prompt-Enhancer`, whose inference API is callable) and by
this package's local Gemma 4 E2B-it enhancer. `pN-*.txt` is the input,
`pN-*.space.txt` and `pN-*.local.txt` the two outputs.

The prompts separate causes: a dense timed multimodal prompt with a
conditioning image (p1), a plainly timed one (p2), an untimed scene (p3), and
one carrying quoted dialogue (p4). Only p1 has a matching conditioning image —
passing it to the others makes the model describe the car instead of the
requested scene, which invalidated a first round of this bench.

## What it measured (2026-08-20)

| | p1 (timed, i2v) | p2 (timed) | p3 (untimed) | p4 (dialogue) |
|---|---|---|---|---|
| Space, characters | 1504 | 1003 | 874 | 1009 |
| local, characters | 1473 | 1098 | 963 | 1082 |
| n-gram on vs off | **differs** | identical | identical | identical |

Two conclusions:

- **Local captions are not thinner than the reference.** An earlier round
  suggested they were a fraction of the length; that was the bench passing the
  wrong conditioning image, plus a real truncation bug (reasoning consumed the
  token budget and the caption stopped mid-word). Both fixed.
- **N-gram blocking is inert except on timestamp-dense prompts, where it
  hurts.** gemma-4-swift-mlx 1.5.0 makes it *possible* to run it with
  reasoning, and that part works; it simply does not pay here. Default is off,
  with the flag kept for the loop protection it exists to provide.
