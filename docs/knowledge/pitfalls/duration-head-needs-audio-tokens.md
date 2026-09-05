---
type: Pitfall
title: The duration head must see audio connector tokens, not just video
description: '`predictFrameCount` fed the head video connector tokens only; upstream always builds the audio connector and gives the head both streams. Fixed in fix/duration-head-audio-tokens.'
tags: [ltx25, duration-head, frames, connector, audio]
timestamp: 2026-09-05T00:00:00Z
---

`--frames auto` (`LTXPipeline.predictFrameCount`) fed the LTX-2.5 duration head
**video connector tokens only**. Upstream always builds the audio connector —
even for a video-only generation — and gives the duration head both the video
and audio connector outputs. The gap changed the head's predictions
substantially: on the reference scene (`A quiet late-night laundromat with
flickering fluorescent lights.`, seed 42, 2.5-distilled) it predicted **27.0 s
→ 473 frames (clamped)** with video tokens only, versus **4.09375 s → 97
frames** with both — a >6x difference on the same prompt.

# Why

Two independent facts compound:

1. **Upstream always builds the audio connector.** The reference
   `encoder_configurator.py` constructs `audio_embeddings_connector`
   unconditionally, and `distilled.py` passes both `text_embeddings` and
   `audio_text_embeddings` to the duration head — regardless of whether the
   generation itself uses audio.
2. **We build it conditionally.** `loadModels()` calls
   `createTextEncoder(gatedAttention:)`, whose default
   `includeAudioConnector` is `false`. The audio connector — and the
   `feature_extractor.audio_aggregate_embed.*` weights that feed it — only
   exist on `self.textEncoder` after `loadAudioModels()` (the `--audio` code
   path) has run. `predictFrameCount` reused whatever encoder was already
   loaded, so a plain (non-`--audio`) `--frames auto` run always handed the
   head `audioTokens: nil`.

# Fix

`predictFrameCount` no longer trusts `self.textEncoder`'s audio connector
state. If it's `nil` (the common case — no `--audio` flag), it builds a
**separate, throwaway** `VideoGemmaTextEncoderModel` with
`includeAudioConnector: true`, loads the connector weights for it from the
already-resolved checkpoint (`includeAudio: true` on
`LTXCheckpointSource.loadComponents`, which is what pulls in the extra
`audio_embeddings_connector.*` and `feature_extractor.audio_aggregate_embed.*`
keys — 262 weights applied instead of 131), and uses that encoder only for the
one `encodeFromHiddenStates` call that feeds the head. `self.textEncoder`
itself is never replaced, so the actual generation path (`videoEncoding`) is
byte-identical before and after this fix — verified by comparing `VAE raw
output` on a plain generation on both sides of the change.

One related trap: `resolveCheckpointSync()` does not resolve the text-encoder
bundle path for split checkpoints (LTX-2.5) — it's built for the transformer
reload path (LoRA, quantization) that never needs `paths.textEncoder`. Calling
it here throws *"is a split checkpoint but no text-encoder path was
resolved"* deep inside `loadComponents`. The fix goes through
`resolveCheckpoint()` (async) instead, which reads the files that are already
present on disk without re-downloading anything.

# Consequences

- **Every `--frames auto` prediction changed**, on every checkpoint that ships
  a duration head (LTX-2.5 only, so far). See
  [duration-head-does-not-read-written-durations](duration-head-does-not-read-written-durations.md)
  for the before/after tables. This is a prediction-quality fix, not a
  regression: the new numbers match upstream's actual inputs to the head.
- `DurationPromptE2ETests`' assertions were tied to the old (wrong) inputs and
  were updated to the newly measured values in the same PR; its
  ceiling-demonstrating test was retired because the scene it used no longer
  clamps.
- A `--frames auto` run now briefly builds and evaluates a second connector
  (video + audio, ~262 small tensors) purely to query the duration head. This
  is cheap relative to the Gemma/transformer load it happens alongside.

# Guarded by

`DurationPromptE2ETests` (gated on `LTX25_CACHE_ROOT`) — see
[duration-head-does-not-read-written-durations](duration-head-does-not-read-written-durations.md).
