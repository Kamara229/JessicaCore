# CALIBRI v45 — pre-quantization + post-quantization teacher compensation bridge

## Scope and scientifically honest claim (pre-registered before measurements)

Proposed by the user: build a smart intermediate adapter between compressed
weights on disk and Jessica's future inference path. The crucial limitation is
information theory: an irreversible compressed file **cannot magically recover
its discarded information**. An offline trained, physically stored correction
sidecar can encode additional information, and a live runtime adapter may then
apply the sidecar without access to teacher weights or labels.

This is a Pythia-14M RESEARCH-ONLY proof, NOT Jessica integration.

### Fixed pretrained source and controlled separation

- Original trained `EleutherAI/pythia-14m` checkpoint; 12 real MLP matrices,
  786432 coefficients, no base model retraining or changes to other weights.
- **PRE**: WikiText-2 TRAIN tokens [0:2048] only; frozen v44 local physical
  rowwise CT33 signed INT3/4, frozen v35 FP16 scale/codes TRAIN-output SSE
  recalibration; 343840 actual physical MLP-only bytes including per-row scales,
  CRC and outer manifest/SHA. This serves as the unchanged pre-only baseline.
- **POST**: *disjoint* WikiText-2 TRAIN tokens [2048:4096] original pretrained
  module activations; teacher W available offline. Fit a rank-2 low-rank
  residual compensator of error W−Q with input-second-moment weighting:
  `M=(W−Q)*sqrt(E[X_B^2])`, `M≈U_2 diag(s_2) V_2.T`, with balanced FP16
  factors A/B so `y=Qx + A(Bx)`. No recovery from compressed bytes alone.
- Separate **affine** output calibration: teacher-target ridge-to-identity
  gain and bias, per output channel, fitted entirely on TRAIN B;
  `y=gain*(Qx)+bias`. Ridge is fixed at 5% of B-output variance.
- **hybrid**: additive rank-2 residual followed by independently fitted
  output gain/bias on B. Predeclared rank=2, ridge=0.05, float16, no parameter
  choice selected by test PPL.
- All three POST variants physically persist a `C45C` sidecar with strictly
  checked per-layer geometry, metadata, CRC32 and SHA256. Every source original
  binary is first stored, read, CRC/SHA-verified and counted on disk.
- Separately execute **real file-backed v44 mmap row-stream + C45C post
  correction** on 8 original teacher input vectors per MLP. Strict decoded W
  scratch ≤32768 bytes and |stream−dense-decompressed-corrected| ≤2e−4. This
  validates functional use from storage without restoring a full W, NOT native
  performance. Full-language-model PPL uses PyTorch hooks on *dense-restored
  compressed Q weights*; do NOT mislabel that as whole-model packed runtime.
- Compare 6 full-model PPL configurations: original pretrained teacher;
  same 343840B PRE-only CT33; PRE+rank2 POST; PRE+affine POST; PRE+hybrid POST;
  independent frozen TRAIN-A calibrated 401224B INT4 reference. Count both
  actual physical CT33 plus sidecar bytes, and show TOTAL bpw including both.
  Correction candidates are **NOT SAME BYTE** as the PRE-only baseline.
  Combined sizes must remain strictly < 401224 bytes, but quality may still
  underperform the calibrated INT4 control; record that negative result.
- Evaluate known WikiText2 validation (1024 tokens, reference only);
  **new ARC-Challenge validation[:128]** (config differs from prior ARC-Easy);
  **new disjoint Yelp Review Full test[64:128]** (prior CALIBRI used test[:64]),
  2048 Pythia tokenizer tokens each. Text NEVER loaded until all physical
  PRE/POST files are serialized and frozen. Report all negative outcomes.
  These are short exploratory language-model PPL excerpts, not classification
  or answer accuracy; corpus name reuse is explicitly flagged.
- Two independent GitHub-hosted Ubuntu 22.04/24.04 CPU runners, pinned
  PyTorch/transformers/NumPy; selected ATen AVX2 and requested oneDNN AVX2;
  compare original source/token SHA and physical pretrained CT33/INT4 SHA.
  Sidecar SVD can exhibit host-specific floating/FP16 rounding; report raw
  sidecar SHA equality/disagreement, don't fake cross-host bitwise equality.
  Third independent job compares exact file byte counts and true CORRECTED
  output accuracy; PPL improvement is **not an integrity requirement**.
- Source/test data integrity, scratch bound and deterministic training
  partitions are HARD gates. Quality win/loss and latency are measured
  outcomes, never required positive outcomes.

## Decision criteria

The immediate engineering proof is: calibrated sidecar from disjoint
TRAIN-B fits and runs on physically compressed CB36 weights with the
previous bounded mmap output buffer and measured PPL versus equal base +
higher-rate INT4. If POST reduces first-use PPL on both new slices it is
promising; if it degrades either, document the domain regression. Even a
positive PPL result is not proof of new mathematical compression or quality
at 0.8bpw. An additional truly budget-matched comparison and *native
packed inference* are prerequisites for performance claims.

**Never change main, production Jessica, Render, Supabase, APK, credentials,
or user data. Only isolated research branch and draft PR.**
