# CALIBRI v47 — optimize for user's INT6 quality/speed operating point, not domination

## Direct predecessor: measured real v46 success and remaining gap

v46 draft PR #176, Actions 41393421200, all 3 jobs PASS on two independent CPUs:
complete pretrained Pythia-14M physical all-state INT6 `CB46` is 10,803,661 B
(6.1438056 physical bits per stored parameter); PRE+POST 10,819,141 B
(6.1526088 bpw). Full-model warm Pythia (batch1 128-token prompt
prefill+32 KV-cache autoregressive steps): PRE+POST median latency ratio vs
original 1.01811 / 1.02540 (Ubuntu22/24). Both satisfy user's <=1.20x
speed loss. Heldout fixed ARC-Challenge test[:128] PPL original 69.31981
vs PRE+POST 74.00146 (+6.75% PPL), Yelp test[192:256] original 135.36616
vs PRE+POST 139.67847 (+3.19%): provisional >=5% quality criterion
fails on ARC; 5% is NOT yet user's chosen quality tolerance.

User explicitly prioritizes INT6-grade storage, <=20% warm runtime
speed penalty, ACCEPTABLE quality, and practical compromise. Do not
claim that compression must outperform standard quantization in all metrics.

## v47 frozen hypothesis

At a full physical model storage ceiling of **6.50 bpw** including all
metadata, spend unused budget (~0.35bpw) on highly valuable *specific
trained parameter ROWS* in the complete Pythia checkpoint, instead of
applying higher precision to entire embedding tensors, which would
overspend. This is a **speed-first hot-FP32-cache** implementation:
all INT6 source weights still expand once at cold load; residual rows
are applied ONCE at cold load; no per-token unpacking or patch kernel.
No compressed-RAM inference claim. Model configuration and tokenizer
are separate, not hidden in the physical coefficient-size metric.

### PRE and gradient-based POST residual selection

- Original trained `EleutherAI/pythia-14m` checkpoint pinned at actual
  HuggingFace revision, full source tensor shapes/SHA, NOT synthetic weights.
- Save and read-back the EXACT v46 `CB46` 10,803,661-byte complete model
  with 6-bit physical signed CF15 per 2D matrix, original 1D tensors,
  genuine MLP WikiText TRAIN-A [0:2048] source-activation scale calibration.
- Load from that physical checkpoint into a fresh model; use ONLY those
  2048 WikiText TRAIN-A tokens, divided into 16 fixed 128-token contexts,
  to calculate the quantized **full-model NLL gradient** with respect to
  all current trained 2D weights.
- Given original W and physical INT6 decoded Q (offline only), each row
  candidate has true residual `delta_i=W_i−Q_i`.
  TRAIN-A first-order predicted reduction in full-model language NLL:
  `gradient_gain_i=max(0,-sum_j(∂NLL/∂Q_ij)*delta_ij)`.
  Rank per actual physical row byte `6 bytes row index + 2*row_width
  FP16 residual bytes` plus per-module header overhead.
- Store selected delta_i in a new actual physical `CR47` sidecar:
  **FP16 residual rows**, manifest records module name and geometry,
  each row has tensor-index and row-index, and integrity CRC32/SHA256
  plus **parent CB46 SHA256** binding. At cold load, safely validate
  outer CRC, SHA, source geometry and parent; then add each stored
  delta to the restored FP32 row exactly once. Do NOT use the
  original teacher at inference.
- Real ceiling: `floor(6.5 × all_model_parameter_count / 8)` bytes.
  Count the entire CB46 base + actual CR47 manifest, rows/metadata/CRC/SHA.
  Reserve **15,480 B** for separately stored TRAIN-B affine POST C45C
  correction so the PRE+gradient+POST variant also fits the ceiling.

### Fair physical-byte controls

- `gradient_global`: full TRAIN-A quantized model loss-guided row
  allocation across embedding, attention and MLP matrices (any trained
  state_dict 2D matrix); selection by gradient-gain / true stored bytes.
- `sse_global`: strongest per-row plain original-vs-INT6 squared weight
  error / true bytes under the SAME total physical budget.
- `sse_module_size_matched`: for EVERY trained tensor, use the SAME
  number of residual FP16 rows as `gradient_global` (and thus the
  **EXACT same physical file bytes**, not just approximately equal
  budgets), but choose rows by simple squared weight reconstruction
  error instead of signed NLL gradient. An honest matched-rate ablation.
- `PRE+POST_affine` original v46 control: TRAIN-B [2048:4096]
  15,480-byte C45C with FP16 per-output-channel fixed ridge gain/bias;
  `gradient+POST_affine` fits TRAIN-B teacher original vs gradient
  patched outputs. Never fit row priority on TRAIN-B or any heldout data.
- Plain `INT6_PRE_only`, `teacher_original` also measured.
  No model base gradient on validation/test, no external text until all
  physical source/correction binaries are persisted and verified.

### Real full-model heldout and inference

- Known WikiText-2 validation 1,024 Pythia-token window is reference
  only, not an untouched test.
- External *predeclared* heldout 2048-token excerpts EACH, FIRST used in
  v47: `allenai/openbookqa` main **test[:128]**, `glue` QNLI
  **validation[:128]**, `Yelp/yelp_review_full` **test[256:320]**.
  These are disjoint from previously used OpenBookQA validation and
  Yelp test[0:256] slices. A previously studied dataset family does
  not mean a new independent benchmark population. No labels are used.
- True full-model language PPL of original teacher and all six
  physically loaded INT6/corrected variants using unchanged
  PyTorch inference. No selective hiding of negative corpora.
- End-to-end Pythia forward benchmark: same real WikiText TRAIN-A token
  trace, batch1, 128-token KV-cache prefill+32 autoregressive teacher
  input steps, AVX2 capped CPU and 2 threads, **2 warmups plus 7 raw
  wall-time repetitions**, median and min/max, per-method prefill and
  decode timing. Ratio vs original teacher must be <=1.20 on both
  independent Ubuntu22.04 and Ubuntu24.04 CPU hosts.
- Temporary working quality gate: <=5% PPL increase vs original
  teacher on EACH new slice. This is NOT user-declared quality
  tolerance; report per-domain rather than asserting task quality.
- CI physical integrity and functional validity must PASS; positive
  speed or quality *scientific* outcome is optional and honestly
  recorded. Independent third verifier checks full original source,
  actual physical CB46, CRC/SHA, identical-module matched physical
  byte counts, all 7 raw timing values, all corpus quality data and
  both host speed/quality verdicts. CPU gradient/PPL minor numerical
  drifts must not be hidden.

### Engineering boundaries

No native packed compute and **no inference RAM savings** yet: at cold
load complete INT6 model expands to dense FP32, and row residuals modify
that hot cache. This deliberately prioritizes user's speed <=20%
objective over full-time compressed-in-RAM storage. Disk compression
is genuine; complete model checkpoint differs from earlier MLP-only
results. No proof of 0.8bpw QMoE, multimodal new weights, or deployment.
No changes to Jessica main, Render, Supabase, APK, production chats,
user information or external services beyond GitHub research branch/CI.

## Interpretation gate

A PRE+gradient-post improvement on genuinely unused quality excerpts
with verified physical <=6.5bpw and <=1.2 warm speed ratio yields
narrow feasibility of **TRAIN-selected sparse original-weight repair**.
But even if PPL improves, its physical cost must be compared with
the exact-size SSE control and mixed-precision INT7/INT8 rows in later
experiments. If signed full-model first-order gradient is noisy,
record failures rather than re-rank on revealed heldouts. End-to-end
Jessica-specific deployment is a separate, explicitly opt-in project.
