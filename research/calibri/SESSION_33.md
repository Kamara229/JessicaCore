# CALIBRI session 33: controlled TT-free adaptive 2D bits as an ablation

## Why this is now the next step (not PR #160)

PR #160 / CALIBRI v31 demonstrated an Intel/AMD numerical reproducibility
control: PyTorch selected ATen CPU DispatchStub AVX2 + oneDNN ISA request
AVX2 gave bitwise identical pretrained Switch BF16 logits and router masks
in the two samples (Actions 41355127113). It **did not compress weights or
measure language-model perplexity**.

PR #161 / CALIBRI v32 **already measured** true physical compressed
weights and WikiText PPL on real Pythia-14M one MLP matrix, with equal-byte
INT4 control (Actions 41356201302). These are established results, not a
task awaiting measurement:

- Original uncompressed PPL **173.38149**.
- Rowwise physical INT4 **33,042 bytes**, PPL **179.67859**.
- Multidimensional TT rank-4 + adaptive 3/4-bit residual stream
  **33,006 bytes**, PPL **219.21966**.
- Real byte-backed Python stream **4.414–4.898 ms** vs the same restored
  dense decoded matrix **0.03475–0.03553 ms**, ~**127–138× slower**.
- Two CPU runs had identical source/bitstream hashes and PPL;
  **negative** rate–distortion and throughput. The 36-byte storage win
  is not a useful compression improvement at this much worse PPL.

## New preregistered v33 experiment (one controlled hypothesis)

The v32 TT factorization on the (128 × 512) Pythia MLP may be harming
the rate–distortion balance. Remove tensor-train **entirely**, keep the
overall size cap **at or below the very same physically serialized
33,042-byte rowwise INT4 control**. On the REAL trained Pythia-14M
matrix, encode direct matrix tiles with a signed 3-bit or 4-bit
physical bitstream, one FP16 scale per tile, three byte tile metadata
and a self-describing CRC32 protected stream. Predeclare two *spatially*
distinct 2D layouts, 16×32 and 8×64. All source coefficients are real;
synthetic weights are used only for codec smoke tests.

For every tile, estimate the difference between 3- and 4-bit
reconstruction weighted by real TRAIN split layer input second
moments. Start with all INT3; upgrade the highest positive train-only
reconstruction-gain tiles to INT4 while respecting strict physical
byte budget. Before evaluating WikiText validation PPL, freeze a
single winner 2D layout based on lowest total TRAIN weighted MSE proxy;
do not change it after the held-out PPL has been observed. Report both
layouts' exploratory PPL for transparency, and the preselected winner.

On pinned real public Pythia-14M and WikiText2 split:
- train activation capture on 256 train tokens;
- true full-model PPL with ONLY one MLP projection changed on
  1,024 validation tokens in 128-token non-overlapping windows;
- compare original full model, rowwise INT4 packed baseline, and
  both TT-free tiled candidates; retain v32 as prior historical
  diagnostic (but v32 results may come from different CPU allocation);
- write actual binary physical files, read back, CRC/hash verify, and
  publish file sizes including metadata and scales;
- **independent streaming research reference** reads physical tile
  bytes on demand, keeps no full reconstructed weight matrix, computes
  one microbatch of real train activations and checks same candidate
  dense reconstruction (maximum error ≤ 1e-4);
- measure Python tile-by-tile stream and exactly equivalent dense
  matrix multiply in each runtime. Record slow/negative outcomes.
  This is not a fast optimized packed kernel, ARM app, or model-wide
  packed forward.

Run independently on two GitHub Linux hosted CPUs with ATen AVX2
selected, oneDNN cap AVX2, PyTorch 2.5.1 CPU, deterministic algorithms,
2 threads, pinned model/data revisions, exact SHA of training and
validation token streams and physically written candidate binaries.
The source check fails on corruption or mismatches, not simply because
PPL is worse. No candidate quality improvements are assumed.

All prior CALIBRI experiments have used WikiText2, including its
validation/test; this is exploratory iterative research, not a blind
confirmatory benchmark. Both rowwise quantization and block
quantization are established methods; do not claim these are
fundamentally new neural-network weights or a patented architecture.
This stage measures physical bytes, actual PPL, and a research-only
CPU reference streaming path, **not** compressed Switch MoE quality,
new 0.8 bits/weight, true fused native kernel, or whole-model bytes.

PR-only, separate branch; never modify `main`, Render, Supabase, APK,
working Jessica, production credentials or data. No ordinary Jessica
chat-side test requested.


## Measured outcome — CI 41356881637 (three GitHub Actions jobs PASS)

The first attempt failed BEFORE scoring PPL because the experiment
referred to a misspelled training-proxy dictionary key. The key was
corrected without modifying any selection criteria or using validation
labels. Second run 41356881637 passed both independent real-Pythia
weights runs and source/metric comparison. Both hosted runs loaded
matching model and corpus revisions, identical train and validation
token SHA256 and identical source weight SHA256. Both independently
serialized candidate binary streams to disk; checksums and full-model
PPL were identical across both hosts.

**Measured physically serialized projection storage** (includes metadata,
CRC, FP16 scales, real packed symbols):
- FP16 original 128×512 source projection: **131,072 bytes**.
- Rowwise 4-bit fixed physical control: **33,042 bytes**.
- TT-free adaptive 16×32 mixed INT3/INT4: **32,980 bytes**.
- TT-free adaptive 8×64 mixed INT3/INT4: **32,980 bytes**.
Both tiled physical formats save **62 bytes (0.188%)** over the INT4
rowwise comparator; this is nearly equal-byte, not a strong
compression gain.

**WikiText2 validation PPL**, measuring the real 14M-parameter
model with only one MLP projection changed:
- Original pretrained weights: **173.3814885**.
- Rowwise physical INT4 control: **179.6785857**.
- Adaptive 16×32 tiles: **181.7027218**.
- Adaptive 8×64 tiles: **180.9361160**.

The TRAIN-only weighted reconstruction proxy selected **8×64**
BEFORE validation PPL was evaluated. That choice happened to be
slightly better than 16×32, but it is still **0.700% worse PPL than
simple packed INT4** at only 0.188% less bytes. Therefore the new
TT-free rate–distortion hypothesis also **did not beat** the
matched-byte control. Compared with the earlier TT+residual v32
PPL 219.2197, removing TT reduced the huge quality penalty but
did not produce an actual matched-budget advantage.

**Separate CPU operation microbenchmark**, same candidate decoded
dense W vs real byte-backed tiled stream on 32 train activations:
- Ubuntu 22.04 Python packed stream **7.9444 ms** vs decoded dense
  **0.06371 ms**, **124.70× slower**.
- Ubuntu 24.04 Python packed stream **2.5050 ms** vs decoded dense
  **0.02628 ms**, **95.33× slower**.
The streamed operation matched the candidate's decoded dense output
within **1.04×10⁻⁷ max absolute difference**, and it did not
instantiate the full dense matrix. Nevertheless a Python per-tile
loop is not a native fused packed kernel; it loses badly in latency.
These numbers are single-projection process-local measurements,
NOT end-to-end model latency or ARM performance. PPL still used
full decoded dense weights in the model.

**Verdict:** physical serialization, reproducibility, byte limit and
mathematical streaming equivalence PASSED; relative accuracy and
latency vs simple INT4 FAILED. No claim of new multidimensional
neural weight effectiveness or 0.8 bit/weight improvement.

**Next isolated engineering hypothesis (not yet measured):** under a
*materially lower* physical byte budget (e.g. ≤3.5 bits/weight),
compare activation-aware mixed tiles with strict global fixed 3/4bit
control and fixed calibrated rowwise INT4 across an untouched new
corpus; only after a quality Pareto advantage consider writing a
native C++ bitstream matmul. Do not spend cycles benchmarking Python
as if it were a production kernel.
