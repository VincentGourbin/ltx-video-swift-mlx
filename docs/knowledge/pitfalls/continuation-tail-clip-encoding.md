---
type: Pitfall
title: The continuation-tail clip must be re-encoded with frame 0 at t=0 (obsolete — the tail is now read natively)
description: Historical. The API used to require a caller-prepared 9-frame clip, and every ffmpeg seek-based recipe for it failed. continuationTailPath now takes the previous segment directly and the framework reads its last 9 frames itself.
tags: [lipdub, continuation, avfoundation, video-io, historical]
timestamp: 2026-07-26T00:00:00Z
---

> **Status: resolved by API change.** `continuationTailPath` now accepts the
> **previous segment's video** and `loadVideo(tail: true)` reads its last 9
> frames natively — no clip preparation, no external tool, no seek. A
> pre-trimmed 9-frame clip still works (its last 9 frames are all of them), so
> callers written against the old contract keep working. What follows is why the
> old contract was a trap; it also stands as a warning for any future API that
> asks a caller to hand-cut a clip.

`encodeContinuationTail` reads the tail clip through `loadVideoFrames`
(`LatentUtils.swift`), which uses an `AVAssetImageGenerator` with
**zero tolerance** on both sides:

```swift
generator.requestedTimeToleranceBefore = .zero
generator.requestedTimeToleranceAfter  = .zero
```

and asks for frame 0 at exactly `t = 0.0`. Any clip whose first frame does
not sit precisely there fails:

```
Failed to extract frame 0 at 0.0s: AVFoundationErrorDomain Code=-11832 "Cannot Open"
```

The recipe originally shipped in the PR #35/#40 help text —
`ffmpeg -sseof -0.4 -i seg.mp4 tail.mp4` — produces exactly such a clip: an
**input seek** leaves the first frame at a non-zero presentation time (and,
with `-c copy`, also cuts at a GOP boundary → `-11821 "Cannot Decode"`).

# Measured

Tested against the real extractor, on a 121-frame LipDub segment:

| Tail extraction | frame 0 readable |
|---|---|
| `-sseof -0.4 … -c copy` | ✗ `-11821 Cannot Decode` |
| `-sseof -0.4 … -c:v libx264` | ✗ `-11832 Cannot Open` |
| `-sseof -0.4 … -c:v h264_videotoolbox` | ✗ `-11832` |
| `-sseof -0.4 … -c:v mpeg4` | ✗ `-11832` |
| **`-vf select+setpts`, all-intra** | **✓** |

The codec is not the variable — the input seek is.

# The defense

Re-encode with a frame **filter** (no input seek) and rebase the timestamps:

```
ffmpeg -i seg.mp4 -vf "select='gte(n,NFRAMES-9)',setpts=PTS-STARTPTS" -r 24 \
       -c:v libx264 -g 1 -crf 12 -pix_fmt yuv420p -an tail.mp4
```

Substitute `NFRAMES` with the segment's frame count. `PTS-STARTPTS` rebases
the first frame to t=0; `-g 1` makes every frame a keyframe. **Only needed on
versions predating the native tail read** — current callers pass the previous
segment and skip all of this.

**Do not write the placeholder as `N`.** `N` is also ffmpeg's own per-frame
variable inside `setpts`, so a recipe using it twice (`select='gte(n,N-9)'`
with `setpts=N/24/TB`) reads as one placeholder and invites a global
substitution — which breaks, with a symptom that varies by ffmpeg build:

| Substitution | Result |
|---|---|
| `setpts=N/24/TB` left as the variable | 9 frames, first PTS 0.000 — correct |
| both `N` replaced by the frame count | **124 frames** (all stamped alike, then re-timed by `-r 24`), first PTS 0.000 — reads fine but is the wrong clip |
| (reported on ffmpeg 6.1.1) | 3 frames at PTS 5.0 → `-11832` |

The 124-frame case is the dangerous one: the uniform read samples its 9 frames
**across the clip's whole duration**, so a tail clip longer than 9 frames
silently anchors on frames spread over the entire segment instead of its last
0.375 s. No error, wrong anchor.

That failure mode is what motivated the API change: an argument whose
correctness depends on the caller having cut *exactly* 9 frames, with
timestamps starting at zero, using a recipe whose placeholder collides with an
ffmpeg variable, is a contract that will be broken. Reading the tail inside the
framework removes all three failure surfaces at once — and the length
requirement disappears entirely, since "the last 9 frames" is well-defined for
any input length.

Framework-side hardening left undone (deliberate): the extractor could accept
a non-zero tolerance for frame 0, or surface a message naming the clip
encoding instead of the raw AVFoundation error. Worth doing if a second
caller hits it.
