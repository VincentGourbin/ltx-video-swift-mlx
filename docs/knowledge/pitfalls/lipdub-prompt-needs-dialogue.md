---
type: Pitfall
title: The LipDub prompt must contain the literal dialogue text
description: A scene-only prompt gives the IC-LoRA nothing to lip-sync to — structurally wrong mouth poses regardless of how correct the pipeline is.
tags: [lipdub, prompt, ic-lora]
timestamp: 2026-07-16T00:00:00Z
---

The LipDub IC-LoRA was trained on prompts of the form:

```
[Scene/character description], speaking in [LANGUAGE] saying: "[ACTUAL DIALOGUE TEXT]"
```

The English wrapper is constant even when the dialogue is in another
language; the dialogue uses the target language's own script (Cyrillic,
Hanzi, …). The text prompt drives WHAT is said; the audio reference provides
the voice. A prompt like *"a man speaking French in a podcast studio"*
(scene only) produces structurally wrong mouth poses — verified user-facing
failure, not a theory. Single speaker only; match dialogue length to the
clip; negative prompt is irrelevant (distilled, no CFG).

# The defense

- Always format `generateLipDub` / `lipdub` CLI prompts with
  `speaking in <LANG> saying: "<TEXT>"`.
- The VLM prompt-enhancement path can rephrase or drop the wrapper —
  `LTXPipeline` repairs it (`speaking|speaks|saying|says` + `in` detection,
  wrapper re-glued when lost). Keep that repair when touching enhancement.

# Citations

[1] Lightricks ComfyUI workflow `LTX-2.3_ICLoRA_Lipdub_Two_Stage_Distilled.json`
(master branch), linked from the
[HF model card](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-LipDub).
