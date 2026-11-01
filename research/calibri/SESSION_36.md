# CALIBRI session 36 — physical compression of all 12 Pythia MLP matrices and first AG News test

## Scientific context: what has already been measured

PR #160 (v31) is a CPU numerical reproducibility study, **not**
the compressed weight/PPL study. PR #161 (v32), #162 (v33), #163
(v34), #164 (v35) already ran on real pretrained Pythia-14m weights.

v35 Actions 41357964814 (three independent CI jobs PASS) compressed
**one** Pythia 128×512 projection to physically stored rowwise
INT3/INT4 28,628 bytes (~3.495 bits per parameter), then used
train-only exact per-row output SSE to calibrate the pre-existing FP16
row scales and symbols WITHOUT changing any row bitwidth or the total
physical bytes. Both independent CPU hosts produced bit-identical
physical SHA and exact PPL:
- WikiText validation PPL uncalibrated 210.07865 → calibrated
  196.14494 at exactly 28,628 bytes; teacher 173.38149.
- TinyStories validation PPL uncalibrated 44.97012 → calibrated
  41.45463 at exactly 28,628 bytes; teacher 38.18278.
- A higher-rate 33,042B INT4 rowwise anchor still performed better
  (WikiText 179.67859, TinyStories 40.47687).

This was a meaningful **positive same-budget calibration** result,
but only for one of the twelve 65,536-parameter MLP matrices. Both
WikiText and TinyStories have been explored repeatedly before, and
no compiled byte-backed forward speed is established.

## Frozen v36 hypothesis and evaluation plan BEFORE new corpus read

Transfer v35's EXACT calibration factors, train-only optimization and
unchanged per-row 3-/4-bit allocation to ALL **12 real Pythia-14M MLP
projections** (six layers, two dense matrices per layer):
6 matrices of shape 128×512 and 6 of shape 512×128.
Total 786,432 pretrained MLP coefficients. DO NOT retrain source
weights, modify attention/embedding/normalization or pick the best
individual layers from test PPL.

For each pretrained matrix, preserve v34's physical ≤28,672 byte
=3.5 bpw budget **including rowwise header, CRC and FP16 scales**.
Both the source rowwise train-weighted INT3/4 allocation and the
calibrated copy have exactly equal per-row bitwidths and bytes.
TRAIN calibration uses only 256 WikiText TRAIN tokens and real
captured layer inputs (256×128 or 256×512); for each row choose the
FP16 scale + integer symbols minimizing actual X_train·(W-Q)
squared output error among the FROZEN v35 scale grid. No validation
or new corpus may influence factor selection or bitwidth allocation.

Write three ACTUAL physical binary bundles:
- baseline 12-matrix uncali­brated rowwise CT33 bitstream;
- calibrated **same-size** 12-matrix CT33 stream;
- higher-rate physically serialized INT4 rowwise reference,
  explicitly **above 3.5bpw**.

`CB36` outer bundle includes version/format, layer id, projection
kind, length for all twelve CT33/INT4 payloads and a SHA256 integrity
footer. Include the complete outer container in physical byte counts
and verify direct disk reread, per-projection CRC decode and global
SHA256. The **global physical 3.5bpw limit including bundling overhead**
is 344,064 bytes (786,432 × 3.5/8). Check the actual file length,
not formula estimates, and fail if exceeded.

Evaluate the full pretrained 14M-parameter model with ALL 12 MLP
matrices replaced together (attention/embeddings/etc unchanged) after
restoring from these genuine physical files. Record full-model PPL for
four configurations: uncompressed original teacher, uncalibrated
3.5bpw all-12, calibrated identical-byte all-12, higher-rate all-12
INT4. For each configuration, run 1024 nonoverlapping 128-token
contexts in two corpora:

1. WikiText2 **validation**, a familiar *development/control* corpus.
2. **NEW `fancyzhx/ag_news` test split** (1024 tokens from the
   first fixed-order nonempty AG News article `text` records;
   separate checksum, dataset revision and number of stories).
   This is the first predeclared AG News evaluation in CALIBRI and
   a cross-domain one-shot stress test; do not adjust compression
   parameters or rerun selection on this test's PPL.

Pin `EleutherAI/pythia-14m` model revision from official HuggingFace
source and exact WikiText/AG News dataset revisions. Two independent
GitHub Actions CPU hosts: Python 3.11, PyTorch 2.5.1+CPU, Transformers
4.46.3, two CPU threads, deterministic ON, selected
ATen CPU dispatcher AVX2 (verify actual selected value), requested
oneDNN ISA cap AVX2. Report source matrix hashes, token hashes,
actual physical bundle SHA256, per-matrix shape/byte/bitwidths,
train output SSE ratio and both corpora PPLs from each host.

## Success / negative gates

A physical byte + hash + CRC/source reproducibility PASS is
separate from a scientific PPL/quality win. If jointly compressing
all 12 matrices degrades quality severely, document it and do NOT
pretend the one-projection improvement generalizes. If calibrated
all-12 has lower PPL than uncalibrated all-12 at exactly identical
bytes on AG News, this supports local output calibration within
this narrow experiment, but NOT novelty or dominance over all
strong calibrated 4bit and 3.5bit methods. If AG News is unavailable
or malformed, report missing data, do not substitute WikiText/TinyStories
and mislabel it as a new corpus.

**Limitations:** the physical bundles hold only 12 compressed MLP
weight matrices and are not deployable full-model checkpoints. An
uncompressed full model’s theoretical FP16 MLP projection reference
is 1,572,864 bytes; the true model load for PPL remains FP32/decoded
dense. The first 1024 AG News test tokens are a tiny, not random sample
of the dataset; evaluation is exploratory, not broad statistical
generalization. There is no model-wide physical stored checkpoint
comparison, compressed inference latency or ARM/mobile speed yet.
These methods are known quantization/rounding techniques, not
a fundamentally proven new multidimensional tensor neural weight
representation. The project still does not reproduce QMoE's
0.8bits/weight on a trained SwitchTransformer.

Only GitHub draft PR and isolated research branch. Do not change
main, working Jessica, Render, Supabase, APK or secrets. No regular
Jessica chat test is requested.
