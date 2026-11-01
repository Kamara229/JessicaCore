# CALIBRI session 35 — predeclared same-physical-byte TRAIN activation covariance calibration

## Clarifying PR reference and completed evidence

**PR #160** documents `v31` PyTorch CPU dispatcher reproducibility,
NOT physical compression nor WikiText PPL. These measurements have
already been completed and archived in separate research drafts:

- PR #161 / v32: real Pythia 128×512 MLP TT+adaptive residual,
  **33,006 bytes**, WikiText PPL **219.21966**, compared with physically
  stored rowwise INT4 **33,042 bytes**, PPL **179.67859**.
  The Python streamed contraction was 127–138× slower than a decoded
  dense operator. **Negative quality and speed findings**.
- PR #162 / v33: TT removed, direct 2D physical INT3/INT4 tiles
  **32,980 bytes**; PPL **180.93612** for train-selected geometry
  versus simple 33,042-byte INT4 PPL **179.67859**.
  **Negative matched-byte quality finding**.
- PR #163 / v34: reduced the hard physical per-matrix budget to
  **28,672 bytes = 3.5bpw including headers and scales**.
  Real file bytes for three rate-matched candidates **28,628**:
  rowwise 1×512 mixed INT3/INT4 PPL **210.07865**;
  16×32 2D PPL **217.52392**; 8×64 2D PPL **225.29608**.
  Uncompressed PPL **173.38149**, above-budget 33,042-byte
  rowwise INT4 PPL **179.67859**. This supports **simpler rowwise**
  representation, not genuinely superior multidimensional encoding.

## v35 experimental hypothesis — TRAIN-ONLY covariance-aware rounding

Do NOT change bit allocation, byte budget or matrix topology.
Copy v34's *one* 128×512 real pretrained Pythia projection and its
exact train-only 1×512 rowwise physical 3-/4-bit allocation.

On real, frozen 256 WikiText TRAIN token layer-input activations
(256×512), for every row independently, search ~27 candidate FP16
scales including the BASELINE existing FP16 scale. Requantize that row
using **the SAME fixed signed 3- or 4-bit codebook capacity** and
select the scale/rounding giving the smallest actual
`||X_train @ (W_row - Q_row)||²`, capturing cross-input covariance.
This differs from v34's diagonal train-activation error allocation.

Do not look at validation or TinyStories during calibration, bit
selection or any tuning. The predeclared limited scale grid and
baseline-preserving candidate mean TRAIN SSE cannot be increased
other than tiny finite-precision variation. If recalibration has
no useful gain, report this as a **negative** result.

Physically serialize:
1. The **original v34 rowwise physical CT33 stream**.
2. A second CT33 stream with only its row-level FP16 scales and
   signed bit-packed coefficient codes changed, and the same
   **exact per-row INT3/INT4 bitwidths**.

Both files must have precisely the **same 28,628 bytes** (or
fail on changed geometry/size); both ≤28,672 bytes (3.5bpw),
including CRC, headers, scales and packed symbols. Save, read,
check CRC, decoded geometry, SHA256 and train SSE improvement.

## Evaluation

On the same pinned pretrained `EleutherAI/pythia-14m` checkpoint,
one original MLP projection replaced and all other weights unchanged,
compare **actual full-model perplexity** across both corpora:
- WikiText-2 validation (1,024 tokens, context 128), the explicitly
  requested metric — repeated exploratory development corpus.
- TinyStories validation (1,024 tokens, context 128), a different
  *secondary-corpus stress test*, **not a blind unseen benchmark**
  because prior CALIBRI sessions have used TinyStories.
Four configurations: original FP32 source teacher, unchanged v34
3.5-bpw rowwise binary, calibrated v35 **same-byte** rowwise binary,
and an explicitly **over-budget** physical rowwise INT4 anchor
33,042 bytes.

Two independent GitHub-hosted CPUs, pin source model/data/token SHA256,
PyTorch 2.5.1 CPU, Transformers 4.46.3, ATen capability AVX2 actually
selected, oneDNN max AVX2, deterministic ON, 2 threads. Preserve
physical bytes, binary SHA256 and both corpus PPLs. Report cross-run
disagreement, do not average/hide negative quality. No full-model
physical-size or inference-speed claim. PPL is on restored dense
compressed weights, not a native fused packed kernel.

If validation PPL improves at truly identical physical bytes, do
not claim a CALIBRI breakthrough: require an independently held-out
corpus, many projections, and further fair rate-matched baselines
before moving to compiled packed kernels. This is a calibration
ablation using established numerical techniques rather than a
fundamentally new multidimensional form of weights.

Only separate research branch, draft PR and GitHub Actions.
Do not modify main, Render, Supabase, APK, working Jessica, secrets
or user data. No ordinary Jessica-chat test requested.


## ACTUAL real-pretrained v35 results — GitHub Actions 41357964814 (all 3 jobs PASS)

Two independent fresh GitHub-hosted CPUs ran the pinned real Pythia-14M
checkpoint. The comparator confirmed identical original matrix, model
revision, WikiText training/validation and TinyStories validation
token SHA256, original and calibrated physical file SHA256, and
**exact same PPL** across both hosts. No model quality results from
the ordinary Jessica chat were used or requested.

**Actual physically stored bytes:** both the original v34
rowwise 3-/4-bit CT33 reference AND v35 recalibrated variant
**28,628 bytes = 3.49463 bpw** for the ONE 128×512 source MLP
matrix, including all headers/CRC/FP16 per-row scales and bitstreams.
Each had the **same exact per-row bit allocation** (57 INT4 and 71
INT3 rows). The physical source/reference binary hash was
`209774b4f60e2d1a930aa362c771a072de08240863c1dbf28fe27e28629de917`;
the calibrated physical hash was
`3a494fd3fb4d7be87dadd28c13fdb9a92d50dcf1866bbf48d98775a3c974cc8b`.
The calibrated 128 rows changed their FP16 scales or symbols and
the true train-layer output-error **SSE ratio fell to 0.48414263**
versus the original at identical row bitwidths.

**Full-model perplexity** (unchanged 14M source weights except for
the single selected 128×512 matrix):

| Configuration | Actual projection bytes | WikiText2 validation PPL | TinyStories validation PPL |
|---|---:|---:|---:|
| Original uncompressed Pythia | 131,072 FP16 reference bytes | 173.38149 | 38.18278 |
| v34 rowwise mixed INT3/4, *uncalibrated* | **28,628** | **210.07865** | **44.97012** |
| v35 rowwise mixed INT3/4, TRAIN output calibrated | **28,628** | **196.14494** | **41.45463** |
| Rowwise INT4, ABOVE 3.5bpw cap | **33,042** | **179.67859** | **40.47687** |

**Observed improvement over a strictly rate-matched control:**
WikiText PPL decreased by **13.93370 (6.63%)** and TinyStories PPL
decreased by **3.51549 (7.82%)**, with **0 extra physical bytes**,
unchanged matrix shape and bit assignments. This is positive evidence
for train-covariance-aware recalibration of existing quantized
neural weight rows, **not** evidence of newly invented multidimensional
parameter tensors.

**Important negative/limitation:** although improved, the calibrated
3.495bpw representation still has worse quality than the old
*above-budget* 4.033bpw INT4 control (196.14 vs 179.68 WikiText,
41.45 vs 40.48 TinyStories); and both are worse than unchanged
teacher. PPL was evaluated with physically decoded dense parameters,
not an in-place packed forward; there is NO compressed inference
speed or full-model checkpoint size result. Only ONE of twelve MLP
projections is altered. Both validation corpora have previously been
used during CALIBRI research, so improvement is exploratory and
does NOT provide an independent blind confirmatory estimate or
generalize across architectures.

**Decision for next separate research task:** pre-register a
*holdout not previously used for Pythia validation*, then calibrate
and compress **all 12 real Pythia MLP matrices** at exact same
per-matrix physical budgets against both uncalibrated and calibrated
rowwise matched-rate controls, measure actual total packed file bytes,
full-model quality, and resource footprint. Select no new
hyperparameters using validation. Only on a reproducible Pareto
win consider a compiled native byte-backed inference kernel; preserve
negative results and do not claim QMoE 0.8bpw. Keep this research
isolated from Jessica main, Render, Supabase and APK.
