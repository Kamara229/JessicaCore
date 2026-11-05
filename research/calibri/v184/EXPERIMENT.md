# CALIBRI v184 — precommitted cross-layer shared rank-2 trainable source

This is a **new fully physically paid representation**, not another
unaccounted dense FP32 repair. It follows the negative v176 joint
K32 centroids/row scales, v181 5bpp quality, v182 V1-native equivalence,
and v183 failed independent first-layer oracles.

## New mathematics

Keep the original immutable complete CB59: 211 native K32/V8
five-bit indexed original 2D matrices, tied input/output head and
raw FP32 1D source. Add **one shared rank-2 input basis for all
matrices of the same input width**:

W_eff x = CB59(W)x + A_layer (B_width x).

Each of 211 per-matrix A rows uses *2 signed INT8* values with one
physically charged FP32 scale per matrix. Each width B (576 or 1536)
is a *single shared rank-2 FP16* matrix reused across all 30
transformer blocks, not separately stored for each layer.
Tied input embedding and output head share the SAME A and B, so
the source definition is consistent. Total A bytes = 409,344,
all A gains = 844 bytes, both shared B matrices = 8,448 bytes.
Before UTF-8 manifest, header, outer CRC and SHA these add
418,636 bytes to original CB59 (11,229,303 bytes), leaving
122,124 bytes for metadata under the strict **11,770,063-byte**
physical cap (<=0.700 bits per 134,515,008 unique original
parameters). The actual encoder REJECTS every model exceeding
the complete file cap; codebook/tensor metadata and digests count.

The new **source consists of actual quantized A and B**, NOT a
normal dense model hidden behind a smaller file. A direct C-native
kernel adds these rank-2 corrections to CB59 packed GEMV and the
on-demand tied embedding. No dense original 2D matrices load in
the native inference runtime.

## Precommitted training and evaluation

- Distinct public TRAIN only: frozen v176 public WikiText and
  AG-News train slices, plus v176 synthetic instruction training
  examples (NOT the 96 historical v169 development questions).
  Exactly 12 predetermined samples, 16 fixed gradient steps,
  optimizer lr A=0.015 and lr shared B=0.0025; random seed 184.
  All 211 A matrices and both B bases receive end-to-end
  gradients across 30 transformer blocks.
- CB59 codebooks, indices, row scales and all pretrained 1D norms
  are **frozen**; only A/B are learnable. FP16 and signed INT8
  straight-through rounding are active **during TRAIN**. Do not
  train a FP32 factor that is then silently rounded only for QA.
- Training uses bounded next-token supervision from original
  FP32 on *public TRAIN texts* plus distinct known-answer
  supervised public TRAIN examples. TRAIN cross-entropy is an
  optimization tool **not a success gate**. No QA target leakage.
- Physically pack the final fixed training iterate FIRST; freeze
  complete bytes and SHA before any dev QA or CPU speed benchmarks.
  Compare the final QAT surrogate against the *same paid native*
  source on the same TRAIN input. Fail hard if mismatch >0.002.
- Actual user-quality diagnostics: frozen 20 development questions
  for first-token diversity/FP32 agreement; then 24 diverse
  actual public DEVELOPMENT questions for full 32-token greedy
  generation, EOS, machine-confirmed correct answers and full raw
  text. Compare **original FP32**, **original native CB59** and
  **new physical native CB184** on the *same exact prompts*.
  Machine-grade conservatively; do not reclassify ungraded as wrong.
- Eight paired 32-new-token native CPU comparisons (two repeats x
  arithmetic/binding/factual/logic). Target >=90% CB59 speed,
  only when answers remain coherent.
- No NLL/KL/PPL accept/reject threshold. Real correct/coherent
  answers and measured CPU speed are required, not reconstructed
  weight RMSE, gradients, improving TRAIN loss or CI success.

**Scientific failure is a valid result**. If actual QA remains
collapsed, report it and do not deploy/promote. Shared rank-2
has only ~413,568 trainable scalar coordinates and 16 steps, so
this *exploratory* proof-of-architecture cannot be assumed to
restore the badly damaged 135M-parameter source.

All files under isolated research branch/draft PR and associated
GitHub Actions. Never modify main, LIVE Jessica, Render, Supabase,
APK, Green Tunnel or private v89 TEST. Jessica is a literal
chat relay, **not a test runner**.
