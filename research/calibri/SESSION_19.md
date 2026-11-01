# CALIBRI session 19 — execute REAL compressed weights without dense restoration

## Why

v18 showed a physically 43,832-byte TT rank2+INT5 residual that gave
TinyStories PPL ~37.045 vs original ~37.041 and simple INT6 ~37.204.
Yet the same candidate scored much worse on WikiText-2 test:
255.015 vs INT6 248.685. **Therefore no quality-preservation claim.**

More importantly, v16–v18 used a FULL dense FP32 restored matrix for
the residual weights at runtime. The physical stored size was compact,
but the executable operator did not demonstrate compressed-weight
inference. The next experiment must address this before further
architectural speculation.

## v19 actual implementation

- \`packed5_v19.c\`: compiled portable C CPU reference supporting
  signed packed INT5 row-scaled streaming matrix/vector multiplication
  directly over the physical bitstream without expanding the full
  dense residual to FP16/FP32.
- \`packed_native_v19.py\`: combined native Tensor Train contraction
  (from earlier sessions) with direct packed residual C GEMM.
  No \`torch.nn.Linear\` with restored dense residual in the candidate
  execution path. This works on flat batches or real hidden sequences.
- Assert physical CRC and shape/length, 5 irregular or regular matrix
  shapes and corrupt-symbol rejection. For each of three v17 geometries,
  check numerical equivalence against dense reconstruction on randomly
  generated input batches of 1, 16, and 128 activation vectors.
- Time both direct TT+packed5 C versus *reference* TT+dense FP32
  residual on GitHub x86-64 CPU, with warmups, p50/p95. This is exploratory:
  Python/ctypes overhead, cold cache, SIMD and thread scheduling differ
  from any Android/NEON production implementation.
- Substitute the physical operator into a **real trained Pythia-14m**
  layer and verify whole-language-model PPL is numerically identical
  to the dense restored **same** TT+residual bytes on 1,024
  WikiText-2 validation tokens. It is NOT a new quality benchmark.

## Accuracy versus speed tradeoff

This is more honest than saying "smaller file means faster model":
without native kernel it is only on-disk size. If the C packed operator
turns out slower than the dense reference, disclose it. A CPU C matvec
might scan weight bytes repeatedly per token: prefetch/caching or
SIMD vectorization could help but could also eliminate the storage
advantage. Development can proceed to ARM only after a reliable
repeatable CPU measurement.

A packed INT5 residual plus TT can still run slower than INT6 on
optimized SIMD and fail cross-domain quality. No production use.

## Scientific boundaries

Tensor trains, 5-bit bitpacking, fused low-bit matvec and
weight-plus-residual decomposition are existing ideas. The source is
one small dense layer, not an MoE. This is a *feasibility/negative-control
test* for one necessary piece of a proposed larger architecture.

No main merge, Render, Supabase, mobile deployment, use of user data,
or full-model speed claims. Never upload checkpoints via Actions.

## Measured outcome — GitHub Actions run 41337036402 (SUCCESS)

First run failed a too-strict absolute-PPL numerical equality check. The
second run **printed the actual full-model scores and passed an explicit
0.1% RELATIVE numerical equivalence gate**, without altering input data.

One real pretrained Pythia output projection 128×512. The physical
TT rank2+INT5 residual uses **43,832 bytes**, compared to **131,072**
bytes for the same matrix in FP16. Persistent INT5 symbol bytes +
FP32 row scales within this CPU reader use **41,472 bytes**; this is
not whole-model RSS.

Numerics: max abs activation-output difference **1.0133e-6** versus
dense restored exact same-quantized geometric weights (randomized
batch shapes 1/16/128). Whole-model WikiText-2 validation PPL:
packed **173.49494**, restored **173.49599**, gap **0.001055**.
These are *not fresh* quality results.

Performance, SINGLE runner, exploratory CPU p50 milliseconds:
| Batch activations | Direct packed INT5+TT | TT+FP32 restored residual |
|---:|---:|---:|
| 1 | 0.2841 | 0.1590 |
| 16 | 1.5657 | 0.2242 |
| 128 | **12.0099** | **0.4735** |

The direct packed kernel was **25.4× slower** at batch 128. This
definitively disproves "compressed storage implies faster execution"
for this implementation. It is not ARM/GPU throughput evidence.
