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
