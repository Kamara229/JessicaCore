# CALIBRI v44 — actual binary-file mmap decoding vs restored dense GEMM (pre-registered)

## Measured evidence before v44

v43 draft PR #173, verified GitHub Actions 41364750930, all 3 jobs PASS: original trained Pythia14m model, **12 real pretrained MLP matrices, 786,432 coefficients**, actual MLP-only physical sizes **343,840 B / 3.49772 bpw** signed CT33 mixed INT3/4 TRAIN-scale-calibrated; **401,224 B / 4.08146 bpw** CF15 fixed signed INT4 TRAIN-scale-calibrated; **499,528 B / 5.08146 bpw** fixed INT5; **597,832 B / 6.08146 bpw** fixed INT6. PPL on first-use OpenBookQA was respectively **161.31, 96.26, 66.32, 63.69**, versus original uncompressed **60.94**; on SNLI **39.31, 24.56, 17.41, 16.28**, vs original **16.07**. All model/physical source SHA and v43 PPL reproducible between CPUs. Critically this was still *dense restored full-model PPL*, NOT packed execution; all rates cover only 12 MLP weights.

## Frozen v44 hypothesis

Does actual **file-backed byte-wise decode + blockwise multiply** of source MLP weights avoid materializing complete original full FP32 W matrices, and how expensive is the CPU throughput in an intentionally simple Python+NumPy reference, compared to a completely restored dense NumPy GEMM? **NO claim of speedup is made in advance**.

- Real source model: unchanged `EleutherAI/pythia-14m`, unchanged original source model revision and source MLP tensor SHA.
- TRAIN selection text: WikiText2 TRAIN first 2048 tokens, unchanged tokenizer, unchanged output activation capture.
- Codec 1: **v43 frozen rowwise CT33 3/4-bit physically packed, per-row TRAIN-output-SSE calibrated FP16 scales**; all 12 source MLPs in one CRC+SHA protected CB36 file, exact **343,840 actual bytes**. This is LOCAL (NOT gated joint) to avoid adding v39 train loss search to a performance-only experiment.
- Codec 2: **v43 frozen CF15 signed INT4 physically packed, TRAIN-A diagonal second-moment calibrated FP16 row scales**; all 12 source MLPs into one physical CB36 file, exact **401,224 actual bytes**. This is above 3.5bpw, so no same-quality or same-byte comparison implied across codecs.
- Full original model read once to extract exactly **8 genuine input activations** for EACH of its **12 original MLP projections**. The same real inputs serve all methods; never run random/synthetic matrix/activations as the primary workload.
- Save TWO complete source weight CB36 physical files to disk and read-back verify original source 12 MLP structure and per-module CRC/outer SHA. Save genuine 8-token module inputs and GOLDEN dense reference matmul outputs for each physical codec, computed by separately fully decoding its physically serialized weights and `np.matmul`.
- Launch **fresh isolated subprocesses** per (codec, mode) so memory does not include original PyTorch transformer or prior dense weights.

**Mode A: mmap_stream** opens actual on-disk CB36 via `mmap.mmap`, verifies module CRC and CB36 outer SHA, records only byte offsets/rowwise length/FP16 scales. For every source module, decode **at most 16 rows × 512 input features × 4 FP32 bytes = 32,768 byte** temporary W scratch. Multiply the 8 genuine token inputs by this short FP32 row block and write its output columns; drop the block before the next one. This mode must NEVER allocate any original full MLP W or a 12-matrix dense clone, verified by separate buffer allocation accounting and code path.

**Mode B: dense_restored** opens the SAME CB36 physical file, validates CRC/SHA, decodes and retains all **12 full FP32 W matrices, exactly 786432×4 = 3,145,728 bytes of dense trained MLP coefficients**, and performs one NumPy/OpenBLAS GEMM per matrix using the identical eight real inputs.

Both modes compare ALL 12 computed outputs against the independently recorded GOLDEN FULL-decoded results from the exact physical file; require maximum absolute error <= **2e−4**. Measure **2 warmups and 5 independent wall-time repetitions**, each a sum of 12 matrix multiplications at batch 8; record raw time list, median and min/max, number of source MLPs, real file SHA, actual scratch bytes, total held full FP32 W bytes. Force `OPENBLAS_NUM_THREADS=1` for worker, CPU ISA ATen AVX2 with oneDNN AVX2 requested in parent for source capture. Report **resource.getrusage(...).ru_maxrss** peak process KiB and **tracemalloc** bytes separately; treat RSS as total-process including NumPy runtime, Python, mmap file cache, input/output arrays and mapped file pages — NOT as an isolated inference memory benchmark. This is warm file-cache, not cold random disk IO, ARM, GPU, native fused quantized GEMM or generation.

**Cross-host:** two separate Ubuntu 22.04/24.04 GitHub hosted CPUs, pinned PyTorch2.5.1 CPU, numpy1.26.4 etc, original trained source model/token SHA, ACTUAL physical CB36 SHA across hosts and 3,145,728 dense weight bytes vs <= 32,768 packed scratch. Third CI must FAIL on inconsistent source/CRC/fidelity/memory *structural* contracts but **must NOT fail** because the mmap row loop is slower or because OS peak RSS differs across CPU types. Report measured negative speed honestly.

## Scientific decision

A verified physical mmap rowstream that computes accurate GEMMs without any full W is an engineering feasibility proof of *blockwise weight access and bounded decoded W scratch* at the existing stored source MLP physical rate. It does not imply native production viability. If Python reference incurs 50×/100×+ latency relative to dense NumPy, record that as a negative result and move to a **compiled/native fused per-row or multi-row packed GEMM** benchmark with an honest complexity/ISA control before saying compression saves inference time. Always separate **MLP-only physical byte counts, peak decoded W scratch, process RSS, true end-to-end full-model memory and throughput**.

Research branch/draft PR only, no merge into main, no changes to Render, Supabase, APK, production Jessica or ordinary chat tests. Does not replicate Switch/QMoE 0.8bpw or demonstrate new mathematical multidimensional learned parameter architecture or a complete deployable compressed checkpoint.


## Implementation correction BEFORE final memory claim — initial GitHub run 41365548158

The first full benchmark ran successfully on both independent CPUs,
all source/physical SHA and 12/12 forward outputs passed (maximum
absolute output discrepancy ~2.9e−6 CT33, ~3.8e−6 CF15).
The Python mmap rowstream was **~295–309× slower** than dense
NumPy 12-projection execution across the two real CPU hosts.
Its correct scratch budget is **32,768 B** versus actual full
decoded twelve FP32 MLP weights **3,145,728 B**.

However the first run's `resource.ru_maxrss` was identically
**~2.1 GiB** for BOTH packed and dense independent subprocesses,
strongly indicating that the memory high-water mark was polluted
by the preceding parent Python process that had loaded an original
PyTorch transformer and datasets, rather than reflecting
the small NumPy-only MLP kernel processes. Therefore
**DO NOT claim a measured overall process RSS advantage or lack
of advantage from these first values.**

To fix the methodological mistake without changing weight
quantization, training, scratch size, worker decode or benchmark
parameters, separate the heavy original-model/corpus preparation
and the NumPy timing into **different GitHub Actions steps**.
The initial step now writes/reads the same physical MLP files,
eight genuine captured original module activations and
separately generated dense-reference results and exits.
The next step starts a clean lightweight Python stdlib
coordinator and fresh isolated NumPy benchmark subprocesses,
which avoids inheriting the heavy PyTorch parent's
historical peak RSS via process creation. The strict physical
file/activation SHA is checked against the preparation
manifest. The original speed ratio is retained as
unrevised evidence, but the new independently allocated
process RSS results are the only candidate values for the
final memory claim. This is a **measurement-isolation fix**,
not tuning to improve numerical speed or source fidelity.
