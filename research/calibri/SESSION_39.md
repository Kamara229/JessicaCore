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
