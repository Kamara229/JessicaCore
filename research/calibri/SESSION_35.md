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
