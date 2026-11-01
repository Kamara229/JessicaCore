# CALIBRI session 21 — can SIMD remove the direct-packed execution penalty?

## Proven before the test

v19 and v20 proved byte-correct direct packed INT5 computation with
the real pretrained Pythia-14m 128×512 matrix. No full dense residual
is allocated inside these compressed execution paths.

v20 within-run x86 p50 batch=128: repeat-bit-decode **5.1584ms**,
one-row bounded scratch **3.6198ms**, TT+dense restored **0.1808ms**.
So row caching alone helps by 1.43x but stays ~20x slower than dense.

A real innovation would be a reproducible Pareto improvement in
quality, TOTAL physical bytes, and actual hardware runtime. The
current compressed kernel **does not** provide that.

## v21 new test

On exactly the *same physical 5-bit bitstream* and fixed v17
geometries, decode once per row using O(cols)=2KiB float scratch,
then compute output using fused eight-wide AVX2+FMA dot products.
Compiled GNU C features runtime CPU ISA detection and skips AVX2
on unsupported CPUs. It is x86-only, not a phone kernel.

Compare three implementations under the same GitHub workflow:
- Portable (scalar) one-row C streamed INT5, v20.
- AVX2 FMA one-row C streamed INT5, v21.
- PyTorch TT + restored dense FP32 exact same-quantized residual.

Input activation batches: 1,16,128. Measure p50/p95, warm starts,
20 iterations. Numerically compare each against dense reference on
five odd/even tensor shapes, and verify one full Pythia module
substitution has <0.1% relative dev PPL gap. Dev PPL only checks
numerical equivalence, not task/model quality.

**Gates**: native ISA available, no weight reconstruction, no
corrupt/invalid memory reads, measurable latency improvement at
batch>=16 without a harmful correctness difference. Report negative
results if AVX2 is still slower than the dense reference.

## Prior-art and product boundaries

AVX2, bitpacking, tensor trains and fused matvec are well known.
No novelty claim. Full-run memory, token/s, Android NEON/GPU or
true MoE expert routing are NOT addressed here; experiment is an
incremental CPU reference and remains research-only.

## Completed results — GitHub Actions 41337588189 (SUCCESS)

CPU ISA: x86_64 AVX2+FMA supported; five odd/even shape checks passed.
Physical geo-small representation remained **43,832 bytes**, with
a **2,048 byte one-row float scratch**. Full-model WikiText-2 dev PPL
for equivalent quantized model: FP32 restored 173.495995,
streamed AVX2 173.490865; difference 0.005129 (<0.1% relative
gate). This is ONLY a numerical equivalence check on reused data.

Same-run CPU p50 milliseconds, geo-small:
| Input batch | Portable v20 | AVX2 v21 | Restored FP32 dense residual |
|---:|---:|---:|---:|
| 1 | 0.2715 | **0.2130** | 0.1239 |
| 16 | 1.1605 | **0.3077** | 0.1794 |
| 128 | 7.5635 | **0.9212** | 0.3883 |

AVX2 was **~8.2x faster than portable row-cache** at batch128, within
this run, but still **~2.37x slower than dense restored**. At batch1
it remained ~1.72x slower. x86 results do NOT generalize to Android
ARM/NEON. Further development must consider matrix-level cache reuse,
shape-conditional scheduling and real mobile kernels. Bitpacking/AVX2
are established techniques and not innovations by themselves.
