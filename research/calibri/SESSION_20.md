# CALIBRI session 20: can we fix the actual direct-packed slowdown?

## v19 result — important negative performance gate

Github Actions run https://github.com/Kamara229/JessicaCore/actions/runs/41337036402
**SUCCESS** after explicitly measuring and tolerancing normal FP32
accumulation differences. The native C direct INT5 bitpacked operator
was numerically valid, including a real Pythia-14m MLP projection:
- Physical geo-small 43,832 B, persistent INT5 data plus FP32 scales
  41,472 bytes (not 128×512 FP32 residual).
- Maximum difference vs restored dense residual: ~1e-6 in reference
  randomized activation tests.
- Restored same-weights WikiText-2 dev PPL 173.49599 and packed
  dev PPL 173.49494, gap 0.001055, both far closer to each other
  than to failed original v17 test. Only a consistency check.
- But for batch=128 on x86 GitHub CPU, geo-small packed operator
  **12.01 ms p50** vs TT+dense FP32 residual **0.47 ms p50**
  (~25× slower). Batch=1: packed 0.284 ms vs dense 0.159 ms.
- This proves neither GPU/ARM speed nor model-wide memory reduction.

v19 C code loops batch outer then rows and bits inner. For a batch of
128 activations it decodes the very same bitstream 128 times.

## v20 falsifiable hypothesis

Decode each packed INT5 row only **ONCE per whole input batch** into
a temporary float32 row vector, then multiply that row by every input
vector. This uses **O(input-width)** temporary scratch (512*4=2048
bytes for real Pythia layer), never a full 128×512 residual matrix.
Storage bytes remain exactly unchanged. Verify across both odd and
aligned widths; compare same physical blobs, same input and CPU
environment against v19 repeat-decode and the restored dense
FP32 reference.

Measure p50/p95 latency for batches of 1,16,128 for three fixed v17
geometries. Verify all numerical outputs against both references, and
full language-model PPL equivalence within 0.1% relative to restored
same-quantized geometry (already-seen WikiText dev; no quality claim).

## Caveat

Row-level temporary dequantization is still dequantization. The novelty
hypothesis is NOT proven by avoiding full-matrix restoration. SIMD and
hardware-specific fused matmul are separate work; a C pointer loop is
only a research prototype. Even if amortized speed improves, a native
BLAS call could still be much faster. No automatic deployments.

## Measured outcome — GitHub Actions run 41337244905 (SUCCESS)

The row-scratch C kernel uses only **2,048 bytes** of temporary float32
storage for this 512-input-channel matrix; no whole residual matrix is
restored in the candidate. Numerical tests passed across four widths,
including non-byte-aligned rows. End-to-end same-quantization WikiText
validation PPL gap between one-row packed and restored was **0.000155**.

All timings below are *within this same v20 CI run* for geo-small,
p50 milliseconds on GitHub x86 CPU (20 samples):

| Input vectors | Re-decode per vector v19 kernel | Once-per-row v20 kernel | Dense residual reference |
|---:|---:|---:|---:|
| 1 | 0.1180 | 0.1470 | 0.0629 |
| 16 | 0.8010 | 0.5757 | 0.0895 |
| 128 | **5.1584** | **3.6198** | **0.1808** |

The optimized row approach improves batch-128 speed by **~1.43×**
compared with the earlier C repeat-decode implementation in the
**same run**, but remains **~20× slower** than TT plus fully restored
FP32 residual. For batch=1 it is slower than repeat-decode, so runtime
must choose the operator based on shape and device.

Absolute times differ from v19's separate CI runner; never draw
cross-run performance conclusions from those values. This is only an
x86 CPU Python/ctypes microbenchmark, not ARM/Android or full-model
tokens-per-second. No quality/novelty claims.
