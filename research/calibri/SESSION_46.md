# CALIBRI v46 — INT6-level complete checkpoint, speed-first practical tier

## User's changed optimization objective

Do NOT seek to dominate all quantizers on all dimensions. An acceptable
FIRST PRACTICAL TIER is **INT6-level physical at-rest compression**, modest
quality loss, and especially **no more than 20% slowdown**.

Strict scope interpretation: this tier favors inference speed and sacrifices
RAM savings. All trained parameters are actually stored in one full-checkpoint
`CB46` binary archive using signed **6-bit rowwise** for 2D weight matrices,
with CRC/FP16 scales, raw original 1D vectors, manifest metadata/outer SHA.
This is NOT claiming 6-bit RAM during inference: all compressed 2D matrices
are reconstructed ONCE at cold startup into dense FP32 resident PyTorch
weights. Subsequent full-model autoregressive forward requires no repeated
Python bit unpack and can use PyTorch/ATen optimized kernels.

This creates a legitimate tradeoff: full model physical disk storage
reduction and near-original warm inference latency, **without runtime weight
RAM savings**. If this is not acceptable, later native packed-in-RAM kernels
are a separate product tier and may sacrifice speed. Never pass a
decoded-hot-cache result off as direct packed inference.

## Frozen test design BEFORE any measured result

- The true pretrained EleutherAI Pythia-14m source weights are loaded from
  a pinned HuggingFace revision; full state_dict tracked via SHA and total
  coefficient count. All eligible 2D pretrained floating matrices are
  physically INT6 via existing real, CRC-verified `CF15` signed INT6
  rowwise format, including **embeddings and attention**, not just 12 MLPs.
  Small vectors/non-matrix weights preserved in original dtype inside
  the same physical `CB46` with actual outer SHA and CRC. Saved on disk,
  reread/decoded with strict source-key and tensor-shape checks.
- **SLA physical INT6-level** maximum: `<=6.5 bpw` computed as **actual
  complete physical CB46 checkpoint bytes / ALL state_dict coefficients**.
  This includes per-row FP16 scales, raw 1D parameters, tensor names, CRC,
  outer SHA and physically persisted metadata. `6.5` includes overhead;
  the primary 2D codes themselves are strictly 6 bits each. No hypothetic
  original-model size assumptions. Separate physical sidecar counted in the
  POST variant. Config/tokenizer is out-of-band, stated.
- `raw`: all 2D parameters undergo basic rowwise fixed-6 packing.
- `PRE`: only 12 known sensitive MLP matrices use frozen v43 FP16
  TRAIN-A (WikiText TRAIN [0:2048]) second-moment scale calibration,
  exactly the same physical 6-bit codes/bytes. All other 2D parameters
  held byte-identical to RAW. This is a direct controlled ablation.
- `PRE+POST`: same PRE complete INT6 archive plus physically serialized
  **v45 C45C FP16 affine teacher correction** on 12 MLP outputs.
  Corrections trained only using disjoint WikiText TRAIN [2048:4096]
  teacher vs quantized MLP outputs. Fixed per-channel gain/bias with ridge
  to identity 5% of TRAIN-B prediction variance. No validation/test tuning.
  Correction alters forward outputs via torch hooks at runtime. This
  may harm or help actual full-model quality and incurs measured overhead.
- **Genuine compressed-only runtime:** load a fresh
  `AutoModelForCausalLM.from_config` instance, restore its complete model
  weights solely from physical CB46 (after source teacher trained checkpoint
  no longer needed), and optionally load the C45C sidecar. This is NOT
  production Jessica because tokenizer/config are still loaded separately.
- Measure full-model **PPL** on old WikiText-2 DEV 1024 tokens (known, do
  not claim new) and two NEVER-used fixed excerpts:
  `allenai/ai2_arc` ARC-Challenge **test[:128]** (distinct from v45
  validation) and `Yelp/yelp_review_full` **test[192:256]** (disjoint
  from older test[0:128]). Fixed first 2048 Pythia tokenizer tokens per
  external domain. Freeze compression and sidecar BEFORE reading any
  external evaluation text.
- **Quality engineering target** for this exploratory session: PPL of the
  compressed full-model <= **1.05 × original pretrained teacher PPL**
  on EACH of the two new domains. The user has not set a numeric
  quality tolerance; **5% is explicitly a provisional conservative
  working acceptance criterion, not an imposed user decision**.
- **Warm full-model CPU speed benchmark:** 128-token true Pythia TRAIN-A
  prompt prefill and **32 teacher-forced autoregressive KV-cache decode
  tokens**, batch=1, same token sequence, original FP32 source and
  compressed-decoded variants on same host, pinned AVX2/2 threads.
  Exactly 2 warmups + 7 timed repetitions, separately report prefill,
  decode and their sum. **20% slowdown threshold = (compressed median
  total prefill+decode latency / original source median) <= 1.20**;
  also report prefill and decode ratios separately. These are true
  entire Pythia forwards, NOT just MLP microbenchmarks. Separate cold
  `from_config`+physical decode loading seconds from warm forward.
  Compare against source FP32 warm forward on each host, NOT a misleading
  hub download/checkpoint cold start.
- Two independent GitHub CPUs (Ubuntu22.04/24.04), third independent
  comparator that strictly verifies model and data revision/hash, full
  physical byte counts+source SHA, 7 actual timings and all PPL. Negative
  quality or speed is recorded, **not CI failure**. CI fails only for
  source/integrity/geometry/invalid measurement violations.
- No Jessica main, Supabase, Render, APK, secrets, user chat, main branch.

## Decision gate

1. If full at-rest <=6.5bpw, PPL relative teacher <=1.05 on both heldout
   corpora, and warm total forward slowdown <=20% on both CPUs, this is a
   **first measurable speed-first prototype**, not production readiness.
2. If quality fails but speed passes, classify sensitive original
   embeddings/attention and allocate extra precision within total
   6.5bpw without using test labels for tuning.
3. If speed fails, remove costly POST hooks or fuse affine corrections
   into weights during one-time load (for speed; account additional
   tensors) before any packed native kernel experiment.
4. To save active runtime RAM rather than only checkpoint storage,
   build a separate native INT6/INT8 GEMM/attention inference engine and
   judge real end-to-end speed against the same dense hot-cache baseline.

PR remains draft research-only, no merge or deployment.
