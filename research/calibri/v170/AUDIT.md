# CALIBRI v169/CB161: source and behavior audit (2026-11-05)

## What v169 actually is

`research/calibri/v169/{protocol,sources,grade,run}.py` is an evaluation
harness. It never creates or refits weights. Its arms are original FP32
SmolLM2-135M-Instruct, original packed CB59, and the original physically
serialized CB161 (CB59 with seven paid 1-bit first-block residuals).
Therefore editing a string scorer cannot restore lost internal knowledge.
Changing response quality requires a new **physical source codec/runtime**.

## Audited source bytes and execution

Actual GitHub v161 artifact: physical complete CB161 size **11,292,174 bytes**,
**0.6715785349 bpp** over 134,515,008 original unique parameters.
Inner original CB59: **11,229,303 bytes**, **0.6678 bpp**. This file is NOT
a full FP32 checkpoint and contains NO original dense 2D matrices in
native inference. Inspecting its manifest found exactly **211 original 2D
trained operators** as 5bit/V8 K32 source codebooks, **61 raw 1D tensors**,
and a single physical tied embedding/output-head tensor, referenced by
`lm_head.weight` as an alias. The original archive is SHA-locked and
immutable in the experiment.

The 5-bit/V8 groups consume one 5-bit dictionary index for **8 source
weights**. Each trained 2D matrix uses only **one global K32 × 8-dimensional
FP16 centroid table** (512 bytes) and FP16 RMS scaling per row. This very
aggressive approximator uses the same 32 vector patterns across an entire
trained operator or the 49,152-row tied embedding/output head. Original
CB59 refit sampled just 4096 globally evenly spaced groups per tensor.

Code-audit of C `direct_kernel_v59.c`: native kernel unpacks indices,
calculates dot products of the 32 group-centroids with each input group,
and scales by the original row factor. Prior unit smoke compared against
explicit dense reconstruction of **the compressed** weights, not
FP32 pretrained. No C-index unpacking error has been demonstrated; a
separate exact-bit toy test is in v170. Correct compressed-matrix GEMV
does NOT imply the compressed matrix represents FP32 well.

Actual bin-level code-index entropy diagnostic on all 211 2D matrices
showed ALL 32 code indices in use; mean empirical index entropy about
**4.98 out of 5 bits**. So there is no credible hidden reservoir of
unused per-vector index entropy to simply turn into language capability.
This doesn't preclude more efficient representation families but rules
out the simplistic explanation of unused codewords.

Prior actual original input embedding normalized rRMSE versus teacher
was ~0.72. v165 found only ~10% pretrained FP32 top1 recovery for a
perfect output-only top128 reranker, so simply rearranging output text
or adding posthoc masks cannot reconstruct missing internal states.

## Behavioral evaluation flaws identified

- v167 SciQ/BooIQ force-choice label accuracy was biased; v167 BoolQ
  answered **Yes to all 96 examples** in all models. A ~58% hit rate on
  that set is a constant-class majority baseline, not proof of comprehension.
- v167 posthoc exact string grading incorrectly called
  `7 + 5 = 12.` and `The capital of France is Paris.` wrong.
  v168 content audit regraded **30/40** FP32 clear factual/arithmetical
  response examples correct versus **0/40** for both compact variants.
  These are small posthoc natural output reviews, NOT a locked TEST.
- v169 improves content diversity and balanced positive/negative answers,
  but its `v169/grade.py` deliberately leaves name/word answers as
  `candidate_gold_present` rather than confirmed correct. It therefore
  CANNOT support an automatic claim of FP32-level correctness, whether
  a candidate score looks good or bad. Read literal response texts.
- v169 `gen` called `install_paid_block` once per CB161 question. That
  recompiles and overwrites a shared C library and swaps 7 operators
  **192 times**. This is a performance/test harness bug, not a codec
  improvement. Fixed separately in isolated v171 using a dedicated
  persistent packed CB161 model with a single C compile and retained
  rotated evaluation order.

## v170 concrete source correction

The old 4096-group center fitting is a reproducible point to improve
while keeping exact original physically paid bits. The isolated v170
candidate re-fits **all 211 original pretrained FP32 2D matrices** using
32768 weight groups each, 6 fixed Lloyd passes, and writes **new 5-bit
indices and FP16 K32/V8 centroids** within the ORIGINAL codebook payload
boundaries. Original row scales, all 61 original 1D tensors, manifest,
aliases and all other bit fields are unchanged. New CRC+SHA verifies
the independently decoded archive. Candidate source bytes MUST equal
11,229,303 exactly; its bpp cannot be higher than original CB59,
therefore lower than v169's tested CB161.

This replaces an actual trained weight representation; it does not
alter the existing QA question grader to fake better results. It
reuses only source pretrained FP32 WEIGHT values for fitting; no gold
answer, SciQ/BoolQ source label, GT controller, NLL, KL or teacher
logit data enters fitting. The physical new model runs via unchanged
real direct C compact GEMV/embedding; 40 exact frozen new free-response
probes compare original FP32/CB59 and actual changed CB170, measure
real generated tokens and CPU time and retain raw answers.

**The candidate's real answer performance is UNKNOWN until the workflow
finishes.** Even better source-weight error is not answer quality and
does not establish parity with FP32. Log all failures; do not promote
the source to live production unless a separate and much broader
answer-quality test proves it.

## Stop conditions and follow-up

1. Rate: newly saved whole physical source <= previous 11,229,303B
   (not just <=0.7bpp), alias integrity, decoder integrity.
2. Behavior: real final answer correctness and meaningful generation
   across diverse questions; human-review ambiguous/creative answers.
   Do not judge by NLL/KL or matching FP32 token probability distributions.
3. Runtime: native full-answer CPU throughput >=90% of original CB59.
4. Scientific status: no claims of FP32 parity from 40 diagnostic prompts;
   require new precommitted, untouched prompts and broader model scale.
5. Deployment: no changes to LIVE Jessica, main, Render, Supabase or APK.
   Green Tunnel experiments remain separate.

If v170 lacks substantive improvement on real text, report FAIL and move
to operator-importance-aware bit allocation/quantization-aware re-training
within a literal fixed byte budget; do not silently repeat a source-MSE
optimizer as a claimed language recovery.
