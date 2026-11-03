# CALIBRI v93 — measure correct, useful answers and generation, not teacher likeness

## Change in research doctrine

The reconstructed neural operators are intended to be **functionally better**, not
forced to imitate the pretrained FP32 output distribution. Therefore from v93:
- **REMOVE KL, NLL, FP32-output similarity from candidate objectives, gates and
  new reports.** Earlier experiments stay immutable for reproducibility; their
  old KL values do not determine whether a candidate is worth developing.
- Direct outcomes take priority: correctly answered original public questions,
  balanced accuracy by label/topic, robust instruction following, consistency
  across prompts, and nonrepetitive **meaningful** generation.
- Model storage <=0.700 fully paid bpp and actual CPU decode throughput >=90%
  of CB59 remain hard engineering constraints, not proxies for correctness.
- Good diversity is **not** enough: randomly different tokens can be incoherent.
  Real answers must be relevant, factually correct, coherent, readable and
  properly constrained. Simple unique-token ratio is just a collapse sentinel.

## Rereading the existing v92 attempt (not a new science test)

The single rewritten whole native block at no additional storage moved original
BoolQ TRAIN_VALID correct answers **5/12 to 8/12**, and mean greedy unique-token
ratio across three fresh prompts **0.0521 to 0.1146**. Both trends are encouraging.
But 12 questions/three prompts are grossly insufficient for stable conclusions,
generation remains beneath diversity goal 0.25, quality/coherence of complete
answers is not yet rated, and full-model speed is unmeasured.

Thus this candidate should be labeled **PROMISING BUT NOT YET PROVEN** rather than
rejected solely due to its mismatch with FP32 probabilities. It remains isolated,
not deployable, and no new official TEST may be opened yet.

## Facade interfaces

- evidence.py: consumes only original correct-answer labels and genuine
  same-prompt greedy outputs, model archive bytes and speed; integrity-checks
  public TRAIN-only sources and their provenance; excludes old KL/NLL fields.
- answers.py: paired before/after correct-answer counts, macro class recall,
  accuracy delta, uncertainty interval; does not use FP32 teacher probabilities.
- policy.py: hard bounds on source size, actual full model speed and coherent
  useful answers; detects majority-answer collapse, repetition and insufficient
  sample sizes; no distribution-distance argument exists in its public API.
- pipeline.py: deterministic report reconstruction from the immutable v92
  diagnostic artifact and explicit status; does not train or write weights.
- Future facade: functional benchmark suite of >=100 truly disjoint VALID items
  spanning BoolQ/SciQ, arithmetic, instructions, factual retrieval, reasoning,
  semantic consistency, and helpful long-form generation. Blinded human or
  separately validated rubric marks factual accuracy, relevance, coherence,
  repetition and completion, without rewarding random novelty.

## Next experiment after v93

Expand independent *train-heldout* quality validation first, with task-level
macro accuracy and paired intervals. Optimize complete block alternatives
against actual response correctness plus generation quality on TRAIN, selecting
on independent TRAIN_VALID only. Compare Pareto trade-offs between correct
answers, meaningful response generation, bpp and full runtime speed. Never
select on v89 frozen independent TEST (already examined). A completely unused
new TEST can be opened once only after source/selection criteria are frozen.

No code modifications in Jessica main, Render or Supabase. A future autonomous
self-repair facade may generate research drafts and validation reports but
cannot silently install revised model weights into a live system.
