---
type: Benchmark
title: LTX-2.5 against MiniMax-H3 on a 10 s 1344x768 case, same machine and seed
description: 50 min 50 against 3 h 55 for the same work, with a 2.8x spread between two identical LTX runs. LTX hits the requested timecode and H3 misses it by 2 s; H3 locks its audio to its own image (0.39 s) and LTX does not (1.12 s). Prompt enhancement made LTX worse here.
tags: [benchmark, ltx-2.5, comparison, audio, timecode, thermal, ltx-2.5-campaign]
timestamp: 2026-08-21T00:00:00Z
---

MiniMax's official reproducible case (`starship bridge / hyperspace jump`), same
M3 Max 96 GB, same seed 0, qint8 both sides, audio generated jointly by both.
1344×768; 243 frames for H3, 241 for LTX (the 8n+1 constraint).

# Wall clock

| | H3 (turbo, 4 evaluations) | LTX-2.5 distilled (11 evaluations) |
|---|---|---|
| total | 14 112 s — **3 h 55** | 3 050 s — **50 min 50** |
| denoising | 217 min 24 (92.4 %) | 42 min 43 (84.1 %) — stage 1 11:32, stage 2 31:12 |
| video VAE decode | 16 min 30 (7.0 %) | 3 min 28 (6.8 %) |
| text encoding | 3.29 s | 16.0 s |
| audio decode | 5.4 s | 8.2 s |
| peak MLX / process | 34.5 / 59.4 GB | 47.0 / 53.4 GB |

**One full-resolution LTX step costs 11 min 12; one H3 step costs 54 min 21** —
4.9× less. LTX's huge step spread (3:53 ± 4:15) is structural, not noise: its
first eight steps run at 672×384.

**Load phases are not comparable.** This port's "Load X" lines read in
milliseconds because MLX memory-maps weights and evaluates lazily, so the real
I/O lands inside the first phase that touches them. H3's 68 s of loading has no
counterpart here.

# The thermal caveat, which is larger than most differences

**Two LTX runs of identical work measured 7 748 s and 2 802 s — a factor 2.8.**
The slow one followed several hours of continuous GPU work; the fast one started
on an idle machine. Only idle-machine runs are comparable, and any cost claim
built on a run that followed other work is worth less than it looks. Consistent
with the back-to-back variance already recorded in
[[generation-baselines-m3max]].

# Timecode and audio-visual lock

The prompt asks for a hyperspace flash at 00:04.500. Measured by mean per-frame
luminance and by audio envelope:

| | visual peak | peak value | audio peak | audio↔image |
|---|---|---|---|---|
| H3 | 6.46 s | 0.61 | 6.85 s | **0.39 s** |
| **LTX, plain prompt** | **3.92 s** | **0.996** | 2.80 s | 1.12 s |
| LTX, enhanced prompt | 7.00 s | 0.795 | 6.00 s | 1.00 s |

Each wins half the criterion. LTX places the event within 0.3–0.6 s of the
request and produces the only true white-out; **it does not lock its audio to
its own visual event**. H3 locks tightly but lands two seconds late.

# Prompt enhancement made it worse here

The hypothesis was that the enhanced caption — which folds the sound description
into the visual timeline — would help LTX lock its audio. Measured: the lock
improved by 0.12 s (nothing), and the flash moved from 3.92 s to **7.00 s**,
away from the target, with a weaker white-out. The whole arc shifted ~1.6 s
later; the closing beat lands at 9.6 s, against the clip's end.

Together with the 2CV case, where enhancement helped, this makes the enhancer's
value **case-dependent and unproven**: two data points that disagree. Do not
treat `--enhance-prompt` as a default improvement. See
[[enhancer-residual-defects-2026-08]].

# Prompt fidelity

Both render the full arc. This port is closer to the text on three specifics —
an empty bridge (the prompt describes only the captain; H3 adds two crew), a
single massive *curved* window (H3's is segmented), dark eyes (H3's are blue).
H3 is clearly sharper: skin, fabric and console micro-detail.

Artefacts, prompts and Perfetto trace live outside the repo, next to the H3
output: `starship-comparison/RESULTATS-ltx-vs-h3.md`.
