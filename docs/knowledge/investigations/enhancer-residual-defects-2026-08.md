---
type: Investigation
title: The enhancer's two residual defects are the reference implementation's, not the port's
description: Measured across four prompts against the LTX-2.4 Prompt-Enhancer Space. Viewpoint stacking appears in both (2-3 mentions there, 2-4 here); detached adverbs appear three times more often in the Space; and only the Space breaks word spacing inside quoted dialogue.
tags: [prompt-enhancer, gemma4, measurement, ltx-2.5]
timestamp: 2026-08-21T00:00:00Z
---

## The question

Two defects survived the Gemma-4 enhancement campaign and were explicitly left
out of the gemma-4-swift-mlx team's scope:

* **viewpoint contradiction** — a caption opens with one shot description and
  then declares a different viewpoint mid-sentence;
* **detached adverb** — an adverb separated from its verb by a comma.

Both were seen in this port's output. The open question was whether they were
ours to fix.

## The measurement

The [LTX-2.4 Prompt-Enhancer Space](https://huggingface.co/spaces/diffusers/LTX-2.4-Prompt-Enhancer)
exposes its inference through `gradio_client`, so the reference implementation's
own captions are obtainable for the same four prompts as the local bench
(`docs/examples/ltx-2.5/enhancer-bench/`): a timed multimodal prompt with an
image, a plainly timed one, an untimed one, and one carrying dialogue.

| prompt | viewpoint mentions (Space / local) | detached adverbs (Space / local) | glued words (Space / local) |
|---|---|---|---|
| p1 2CV | 2 / 3 | 3 / 1 | 0 / 0 |
| p2 pier | 2 / 2 | 0 / 0 | 0 / 0 |
| p3 rainy street | 3 / 4 | 1 / 0 | 0 / 0 |
| p4 dialogue | 2 / 2 | 2 / 1 | **2 / 0** |

## What it says

**The viewpoint defect is the model's.** On p2 both implementations produce the
*same phrase from the same input*: "A static wide shot frames an empty wooden
pier … at the two-second mark, a seagull lands precisely on the far wooden
post, **captured from a high-angle viewpoint**". A locked-off wide shot does not
acquire a high angle at the two-second mark. The reference writes it too, word
for word, so nothing about this port produces it.

**The adverb defect is the model's, and worse there**: six occurrences in the
Space's four captions against two here.

**One defect is the Space's alone**: it drops the space inside quoted dialogue —
`"That'snot the alternator."`, `"Then what isit?"` — where this port writes both
lines correctly. Quoted dialogue is exactly the region the n-gram experiments
touched ([[ngram-blocking-mangles-prompt-quoting]]), and it is the one place a
caption defect reaches the video directly, since LipDub is trained to read the
target dialogue out of the prompt ([[lipdub-prompt-needs-dialogue]]).

## What follows

There is no port bug here. Improving either defect means diverging from
upstream's system prompt — a product decision, not a correctness fix — and any
such change has to be judged on the **video** it produces, not on caption prose:
this campaign has already recorded once that a caption reading better is not the
same as a clip being better.

The experiment left to run, when a GPU is free: add one constraint line to the
system prompt ("state the shot type and viewpoint once, at the start") and
measure the viewpoint-mention count and caption length against this table.
