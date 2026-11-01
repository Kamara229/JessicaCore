# CALIBRI v39 — joint model TRAIN-NLL, exactly equal physical bytes, new SciQ corpus

## Verified prerequisites (not pending)

v36 (PR #165) already loaded **all twelve** Pythia-14m real pretrained
MLP projections (786,432 weights) and measured actual physically
serialized 343,840B / 3.49772bpw CB36 bundles and source-matched AG News
full-model PPL. Calibrated PPL was still 650.91 versus original 235.47.

v37 (PR #166) used TRAIN-only individual-layer NLL and marginal
per-row activation-output SSE to allocate 343,840 physical bytes globally.
Its calibrated result was worse than the matched-byte per-matrix local
control on WikiText development and a new AG News slice.

v38 (PR #167, GitHub Actions 41360326071 **all three jobs PASS**)
required each module to retain 75%–125% of the INT4 rows in its
original local v36 allocation. ALL bundles were exactly 343,840B,
and the new GLUE SST-2 corpus confirmed worse GLOBAL TRAIN-calibrated
PPL than LOCAL: SST-2 680.31779 vs 570.07926 and WikiText DEV 501.85659
vs 443.63432. New bounded allocation did NOT repair the quality collapse.
Both CPU runs produced same binary SHA and output PPL.

These are established negative findings. Merely repeating the
first-use SST-2 test or tuning against it would bias evaluation.

## Frozen next hypothesis: use joint FULL-MODEL TRAIN NLL as selector

Retain real, pinned `EleutherAI/pythia-14m`, twelve original source
MLP matrices, train activation capture, unchanged v35 FP16 per-row
scale calibration grid, and **exact physical 343,840-byte CB36 bundle**,
including signed 3/4-bit symbols, FP16 row scales, module CRC and outer
container SHA. Other original model layers remain unchanged.

Compare:
1. Original uncompressed trained Pythia teacher.
2. v36 local per-matrix uncalibrated and TRAIN-calibrated at 343,840B.
3. v38 floor-protected TRAIN-proxy global uncalibrated and calibrated
   at 343,840B, as a historical heuristic control.
4. **v39 JOINT TRAIN-NLL** selected bit allocation, both uncalibrated
   and TRAIN-calibrated physical 343,840B, with exactly equal per-row
   3/4 bitwidths between calibrated/un-calibrated.
5. Physically stored INT4 high-rate quality anchor (401,224B) **above**
   the 343,840B cap.

Start v39 from the original LOCAL per-matrix physical row bit
allocation (NOT from test-optimized v37/v38).
Preserve precommitted **per-module bounds** of 75% minimum and
125% maximum of LOCAL INT4 row count.
Use frozen **WikiText-2 TRAIN** 256 tokens / context 128 for
ALL bit allocation, calibration and model-NLL optimization.
No evaluation corpus may enter this procedure.

Generate deterministic candidate bitwidth swaps, with every proposed
transaction having EXACT ZERO physical byte delta:
- Within a module, remove the lowest TRAIN activation-SSE gain 4-bit
  row and add the highest positive TRAIN activation-SSE gain 3-bit row.
- Between equal-width (128 or 512 input features) modules, move one
  4-bit row to another module, obeying module floors/ceilings.
- Across feature widths, trade four 16-byte row upgrades for one
  64-byte upgrade or vice versa; floor/ceiling protection must hold.

Rank the frozen proposal pool by TRAIN-layer output SSE, but select
accepted swaps ONLY by actual end-to-end **full-model TRAIN NLL**
measured with ALL 12 physically compressed + TRAIN-calibrated
MLP matrices present simultaneously (not uncompressed teacher and
not a sum of independent layer errors). Two fixed selection rounds,
maximum twenty full-model TRAIN-NLL probes per round; accept a
candidate iff its actual TRAIN NLL strictly falls by at least 1e-7.
If no accepted candidate, preserve LOCAL unchanged as a genuine
negative optimizer result. No tuning after inspecting validation PPL.

Each candidate is physically re-encoded; the actual CB36 length must
be exactly **343,840 bytes**, not estimated. Immutable model/row
integrity and FP16 calibration bit assignment are checked at every
swap. Preserve a full training-proposal NLL log to distinguish
observed training effects from post-hoc hypotheses. Always restore
unmodified source matrices after the TRAIN search.

## NEW independent-corpus diagnostic

First-time CALIBRI science-question evaluation corpus:
`allenai/sciq`, **validation** split, with fixed original ordering
and deterministic first **2,048 tokenizer tokens** concatenating
science-explanation `support` and `question` strings.
Record public dataset revision and exact token SHA. The `SciQ`
validation text is not read by the allocator, only by the final
locked model-quality evaluation. WikiText2 validation 1,024 tokens
remains a familiar development control. Evaluate full-model
perplexity with all twelve MLPs replaced by decompressed dense
weights from the physical byte stream, NOT the byte-streamed runtime.
This different domain limits accidental narrative bias but 2,048
fixed tokens do not prove task-level accuracy or generalization.

Two independently allocated GitHub-hosted Linux CPUs with fixed
PyTorch 2.5.1 CPU, Transformers 4.46.3, selected ATen AVX2 and
requested oneDNN maximum AVX2, 2 threads, deterministic algorithms
ON. Require exact source model, tokenizer/dataset token SHA,
per-module original weight SHA, real physical bundle SHA/size and
reproduced PPL across hosts. Report any cross-host differences, source
corruption or physical size failures; **never call negative PPL a
software failure**.

## Decision gate and scope

Success requires actual TRAIN NLL to decline without size growth
and an improvement on the precommitted NEW SciQ validation versus the
LOCAL TRAIN-calibrated same-byte control. Even success would be
exploratory, not evidence of new multidimensional tensor mathematics,
reproducing QMoE 0.8bpw, a usable compressed full-language model,
native packed inference speed, mobile/ARM execution or compressed
whole-model disk storage.

The physical CB36 artifacts are ONLY the twelve Pythia MLP weight
matrices; embeddings, attention and norms are NOT serialized.
One-time test data exposures and model selection are documented.
No merge to main, Render, Supabase, APK, production Jessica, secrets
or normal Jessica chat test is requested.


## Actually observed v39 scientific measurements — GitHub Actions 41361149476

**All three GitHub Actions jobs SUCCESS**: independently hosted
Ubuntu 22.04/24.04 CPUs ran real pretrained Pythia-14m. The
third-source checker verified exact original pretrained checkpoint,
original matrix source SHA, WikiText TRAIN and DEV and *first-use*
SciQ validation token SHA, exact 12-module physical CB36 binary SHA
and lengths, frozen 75%-125% per-matrix row-precision limits, and
**bitwise identical PPL** across the two hosts. No ordinary Jessica
chat or external owner-run test was needed.

### Actual matched physical storage

Every tested 12-MLP rowwise INT3/INT4 binary container —
LOCAL uncalibrated, LOCAL calibrated, v38 floor-protected GLOBAL
uncalibrated/calibrated, and new v39 JOINT NLL-selected
uncalibrated/calibrated — occupies **exactly 343,840 B**
(**3.497721 bits per MLP coefficient**). This includes actual
per-module signed physical bitstreams, FP16 row scales, CRC,
outer CB36 descriptors and SHA256 footer; all were written and
verified. The above-budget rowwise INT4 reference is **401,224 B**
(~4.08146 bpw). There are **786,432 real source MLP weights**
across 12 matrices; the binary is **not** a full-model checkpoint.

### Actual full-model TRAIN loss optimization

Starting from the LOCAL train-calibrated 343,840B baseline,
two frozen rounds independently scored up to twenty byte-neutral
candidate row swaps EACH against the **joint model-level WikiText
TRAIN NLL** with all 12 MLPs simultaneously quantized, **not** summed
local activation MSE or held-out evaluation PPL:

- Baseline measured joint TRAIN NLL **6.3518228531**.
- Round 1 accepted a *within-module* 3↔4-bit row precision swap
  in `gpt_neox.layers.3.mlp.dense_h_to_4h`, lowering TRAIN
  NLL to **6.3475291729**.
- Round 2 accepted another *within-module* swap in
  `gpt_neox.layers.1.mlp.dense_h_to_4h`, lowering TRAIN
  NLL to **6.3440327644**.
- Total observed TRAIN NLL reduction **0.0077900887**
  (about **0.123%**), using exactly **0 additional physical bytes**.

**Critical nuance**: although cross-matrix and 4-short↔1-long
byte-neutral swaps were INCLUDED in the predeclared proposal pool,
the two accepted proposals both swapped rows INSIDE a single
up-projection matrix. Thus v39 demonstrated genuine *joint
model-level TRAIN-loss evaluation* but **NOT an observed successful
cross-layer precision transfer** or broadly optimized all-layer
bit allocation.

### Full model PPL after all 12 MLPs replaced together

| Real measured variant | WikiText2 DEV PPL | First-use SciQ science support/question validation PPL |
|---|---:|---:|
| Source uncompressed teacher | **173.38149** | **59.74490** |
| LOCAL uncalibrated, 343,840B | **1597.66884** | **425.40615** |
| LOCAL TRAIN-calibrated, 343,840B | **443.63432** | **150.60845** |
| v38 GLOBAL protected uncalibrated, 343,840B | **1666.13674** | **382.01090** |
| v38 GLOBAL protected TRAIN-calibrated, 343,840B | **501.85659** | **161.25250** |
| **v39 joint TRAIN-NLL selected uncalibrated, 343,840B** | **1580.44543** | **423.81818** |
| **v39 joint TRAIN-NLL selected & calibrated, 343,840B** | **443.87060** | **150.89525** |
| Rowwise INT4 above-rate quality control, 401,224B | **372.07758** | **116.11262** |

**NEGATIVE predeclared primary result:** Joint TRAIN-loss optimization
slightly worsened the *calibrated* held-out PPL versus exactly
same-physical-byte LOCAL TRAIN calibration:
WikiText DEV **+0.23628** PPL and first-use SciQ **+0.28680** PPL.
Thus the measured TRAIN NLL gain **did not transfer** to the held-out
text in this small trial. The uncalibrated v39 variant had
slightly better PPL than uncalibrated local, but this was
NOT the designated calibrated quality gate. The whole 3.5bpw
all-12 model is still substantially inferior to source pretrained
quality on both corpora, and to the >budget INT4 anchor.

### Scientific interpretation / next gate

Avoid iteratively tuning swap count or row choices on the now-seen
SciQ validation subset. This result suggests that **256 TRAIN tokens
are too narrow to reliably select globally effective bit transfers**
(or the rowwise codebook itself is rate-distortion limited), but
this is only a plausible hypothesis, not a proven causal diagnosis.
The next independent research experiment should increase
training-only calibration breadth **with all parameters frozen
before opening a separate, previously unused evaluation corpus**,
and compare to strong **rate-matched calibrated 3.5bpw** baselines.
Do not call a smaller TRAIN NLL an improvement in validation quality.
No QMoE 0.8-bpw trained Switch MoE result, new multidimensional
tensor architecture, physical full-model checkpoint, or native packed
throughput has been established. PPL uses restored dense weights
and does not demonstrate compressed execution speed.

Main, Render, Supabase, APK, production Jessica and ordinary
Jessica chat were unchanged; all work was in PR #168's isolated
research branch.
