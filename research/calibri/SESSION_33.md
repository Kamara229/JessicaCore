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
