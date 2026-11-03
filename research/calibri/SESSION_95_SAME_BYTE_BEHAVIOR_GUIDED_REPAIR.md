# CALIBRI v95 — start quality improvement from frozen compressed native baseline

## Non-negotiable engineering baseline

This is the first **quality-only** intervention after freezing previous
real-weights research. Nothing in this experiment changes the pinned original
trained weights, already achieved packed weight archives, or native C kernel.

- Source model: SmolLM2-135M-Instruct, pinned original revision.
- **CB59** 11,229,303 bytes / **0.667839 bpp** including whole archive.
- **CB88** 11,413,683 bytes / **0.678805 bpp** including 4D correction.
- Historical CPU native speed: CB59 **22.7315 tok/s**, CB88 **22.4345 tok/s**
  in earlier 4-prompt run; never retune on that already-opened TEST.
- Hard guard: **zero added bytes** over the selected CB59 source and unchanged
  native 5-bit/8-vector GEMV kernel. Require real new-prompt full CPU speed
  preservation: aim >=98% of immediately compared original CB59 on same runner.
- Correct answers, class-balanced accuracy, useful decoded content and genuine
  generation repetition are primary measurements. No KL/NLL or FP32
  probability matching in candidate fitting or acceptance. FP32 is consulted
  ONLY for correctness against external actual ground truth.

## Experimental treatment (small facade, single native block)

Frozen original source layer: model.layers.13.self_attn.v_proj.

Candidate family comprises original CB59 (no change), previous v92
same-byte whole-block rewrite, and three *new* fully native complete 5-bit
re-encodings of the same learned projection. They are built from existing
original FP32 source weights as **source material**, not as a teacher loss:

1. Uniform per-input dimensional weighting, re-fit every group index and
   32-entry shared eight-dimensional centroid codebook.
2. Root activation-aware weighting of original public TRAIN-only inputs.
3. Winsorized activation-energy weighting suppressing outlier dominance.

Source 2D matrix geometries, physical packed-index/FP16 scale and codebook
byte sizes, CRC32/SHA256, native block runtime and ALL other layers stay
exactly unchanged. No extra correction memory or additional shared runtime.

The local weighted-reconstruction metric may drive the internal index/codebook
construction, but it **never decides** which candidate is better than another.
Only actual original labeled answers and generation outcomes decide ranking.

## How automated Quality Doctor selects

- TRAIN FIT: BoolQ train[1350:1382] + SciQ train[1100:1124]; genuine
  original public TRAIN labels. Within training-only, a small separate
  activation group from training is used for early internal encoding.
- Evaluate each actual native full model on exact same FIT questions
  and two fixed generation prompts. Rank by count of correct answers first,
  then class-balanced accuracy, repeat concentration and diversity.
- Freeze candidate name BEFORE loading independent VALID data in process.
- Independent TRAIN_VALID: BoolQ train[8350:8414] and SciQ
  train[9100:9132]. Both are disjoint from all earlier known v90/v91/v92/v94
  fit and measurement slices; untouched original v89 official independent TEST
  is forbidden throughout.
- Compare CB59, selected compressed candidate and original FP32 *only on
  true-answer correctness*, separately in both labeled domains. Also run
  eight distinct real KV-cache greedy prompts, preserve full decoded response
  for later external semantic/human checks, and measure **full packed CPU
  processing time**, not microbenchmark.
- Require source bytes IDENTICAL to parent, no loss of accuracy by domain,
  no generation degradation, no significant speed penalty. Need genuine
  semantic response review, repeated full-speed tests and larger validation
  before any scientific PASS or adoption. If no variant improves, keep
  locked original unchanged, record negatives and propose another operator
  family instead of adding unbudgeted corrective bits.

## Architecture: small modular facades

spec.py — pinned fit/valid ranges, target and prompt sets.
qa.py — original pinned real labeled FIT and VALID answer tasks.
features.py — original public FIT-only activation statistics for source
             codebook candidate generation.
variants.py — exact same-byte 5-bit native full-block rewrites.
scoring.py — real correctness-first FIT ranking and decoded greedy probes.
gates.py — correctness/generation/size/full-model CPU invariants.
pipeline.py — orchestrates a single finite bounded search and restores
               original source in finally.
test_v95_quality_repair.py — test no leakage, class collapse, CRC
                              and no paid bits.
CLI/workflow — research-only original source artifact references.

This branch is stacked on PR #226 for isolated research and must remain
draft. No writes to production Jessica, main, Render, Supabase or APK.
Scientific failures MUST remain in the report rather than be hidden.
