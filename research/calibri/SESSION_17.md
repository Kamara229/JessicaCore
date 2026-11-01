# CALIBRI session 17 — adapt tensor geometry or reject the idea

## Why this session?

v16 physical 2-core Tensor Train **worked mathematically** but performed
poorly on an actual pretrained Pythia-14m layer:
- pure TT rank8: 8,214 bytes, dev perplexity 3,789.66 (original 173.38).
- TT rank8 + packed INT4 residual: 41,238 bytes, dev PPL 177.53.
- TT rank16 + packed INT4 residual: 49,430 bytes, dev PPL 186.77.
- Teacher-logit 12-step healing reduced *train* KL, but dev PPL worsened
  to 179.92. It was overfit or otherwise failed to generalize.
- The simplest INT6: 49,426 bytes and dev PPL 173.27, passed the gate.
- Native TT8 Python CPU matmul 0.247ms p50 vs dense FP32 0.065ms.
=> **No positive compression or speed claim.**

The outstanding hypothesis is not that any 4D reshape is beneficial,
but that a **different topology plus quantization of its residual** might
be more favorable at the same total physical byte budget.

## Test, no hindsight on test

One pre-existing real Pythia-14m output projection 128×512. For each
TT decomposition of shape (m2,n2) in:
(4,32), (8,16), (8,32), (16,16), (16,32), (2,64),
try rank 2,4,6,8 and residual precision 4 or 5 bits. This makes
6×4×2 = 48 geometry candidates; physically reject those above 1.5×
INT4 full-byte budget *before* scoring.

A physical bundle holds TT core (including FP16 factors + header/CRC)
and a complete signed INT4 or INT5 residual codec (scales + bitstream
+ header/CRC), plus a global 16-byte framing header. Thus no hidden
free base, scales, metadata or decoder state. The quantized residual
is computed relative to the **serialized FP16** reconstructed TT core.

Use direct TT contractions in model forward, followed by a dense
*dequantized* residual linear operator. This tests geometry and
physical format, **not** a fully fused packed inference kernel.

Choose the physically smallest candidate whose measured development
PPL <= 1.01× original and physical size <=1.5× INT4. Compare against
ordinary INT4/5/6/8. The test split has been viewed in earlier
CALIBRI experiments, so it is exploratory rather than a publishable
holdout. No candidate is shipped to Jessica.

**Breakthrough gate:** a TT geometry must pass quality while being
smaller than the simple INT6 under the same verification conditions.
Even if a variant passes, broader datasets, real MoE and device
performance still must be verified before asserting advantage.

## Prior work

TT neural compression predates CALIBRI; TD-MoE (ICLR 2026)
already uses joint tensor structures for real experts. Novelty must
be demonstrated against strong baselines, not asserted from combining
known methods.
- https://arxiv.org/abs/1509.06569
- https://proceedings.iclr.cc/paper_files/paper/2026/hash/fc8ee7c7ab5b5f6b1615045dfb617ed6-Abstract-Conference.html

No main merge, Render, Supabase, APK or production release.
