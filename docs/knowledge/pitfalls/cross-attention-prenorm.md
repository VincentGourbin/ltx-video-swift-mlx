---
type: Pitfall
title: Cross-attention's own q_norm does not stand in for the block's pre-norm
description: The legacy 6-value block fed cross-attention the raw residual. Upstream feeds it the RMS-normalized one; the attention's q_norm acts on the projected query, which is a different operation. 1.1e-2 relative error, found the day a parity harness existed for the transformer.
tags: [transformer, cross-attention, normalization, parity, root-cause]
timestamp: 2026-08-21T00:00:00Z
---

Upstream's block returns two things from the self-attention residual:

```python
x_fma = x + y * gate
return x_fma, rms_norm(x_fma, norm_weights, eps=eps)
```

and cross-attention consumes the **second** one. Its own docstring is explicit
that it "does not normalize again" — the normalization has already happened.

This port's 9-value block (every shipped LTX-2.3 and 2.5 checkpoint) was right,
because its `adaln(x, scale:shift:)` normalizes internally before applying the
modulation. The legacy 6-value block was not: it passed the raw residual, on the
reasoning that the attention's `q_norm` handled it.

It does not. `q_norm` normalizes `to_q(x)` — the query *after* projection, per
head-group. Normalizing `x` changes what `to_q`, `to_k` and the residual scale
all see. The two are different operations, and one does not substitute for the
other.

Measured on the parity harness, two blocks, float32:

| cross-attention input | relative error vs upstream |
|---|---|
| raw residual | 1.1e-2 |
| RMS-normalized residual | 6.2e-6 |

**No shipped generation changes**: `LTXTransformerConfig.default` is the only
config with `crossAttentionAdaLN: false`, and no model variant selects it —
`.ltx23` and `.ltx25` both take the 9-value path. The bug was latent, and worth
recording for what found it rather than for what it broke.

**What found it**: the transformer parity harness
(`scripts/transformer_reference.py`), on the day it first ran. Nothing else
could have — the error is far too small to see in a video and far too large to
be float noise, which is exactly the band a port loses track of. See
[[na-tile-mask-cache-key]] for the same lesson on the decoder side.
