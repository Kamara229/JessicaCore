# CALIBRI v55 — frozen four-family paired bootstrap TRAIN gate and first real answer-quality test

**One isolated research next stage, continuing verified v54 draft PR #184.**

The v54 four-family/8-window jackknife selected an original-model FP16 error-repair correction at 6% per trained tensor in a complete real pretrained Pythia14m physical INT6 checkpoint. It improved new ARC-Easy and Yelp PPL, but new HellaSwag PPL was **0.02487 worse** than plain INT6 PRE — demonstrating that tiny language PPL excerpts cannot guarantee non-regression. Later tuning against that revealed HellaSwag validation subset is forbidden. Instead v55 tests actual heldout *correct-answer selection* with fully preregistered prompt, model, data and training-only choice.

## Frozen physical source and candidate archive

Identical pretrained real Pythia14m model revision, v46 CB46 physical entire trained checkpoint **10,803,661 B** with calibration on the first **2,048 WikiText TRAIN-A tokens**, and exactly the source-only v51-v54 deterministic 6%,12%,24% per-source-tensor SSE/physical-byte CR47 FP16 row residual corrections: **610,751 / 610,715 / 610,875 B**. All source CRC, SHA256, parent hash, tensor geometry, row count and exact byte budgets verified before training-only candidate selection. Source + optional repaired rows + 15,480 B **reserved** future POST capacity **<=6.50 real bits per model parameter**. Cannot claim compressed runtime memory: all real Pythia weights expand into dense FP32 hot RAM at cold start, then CR47 applies once.

## New eight TRAIN windows, never exposed to previous CALIBRI experiments

Two new disjoint 2,048-token *training* excerpts per four equal-weight CORPUS FAMILIES:
- WikiText TRAIN tokenizer positions [16384:18432] and [18432:20480].
- AGNews TRAIN record [1536:1664] and [1664:1792].
- SciQ TRAIN [512:640] and [640:768].
- SuperGLUE BoolQ TRAIN [256:384] and [384:512].

All split names/revisions pinned. The **only innovation in this step for selection** is a deterministic family-cluster paired bootstrap: for each physical candidate, take window NLL gain `NLL_plain_INT6_PRE−NLL_repaired`. Resample four families with replacement, then one of two disjoint windows per sampled family, 1,024 deterministic draws (seed 55). Mark candidate eligible iff >=3/4 true training family means positive, worst family gain >=−0.0080 NLL, and sorted bootstrap macro-gain lower decile at fixed 0-index position 102 is >=+0.0005 NLL. Choose best *lower decile*, then larger macro, then smaller source-tensor repair cap as tie-break; fallback plain PRE. This is **an exploratory sensitivity heuristic, NOT a calibrated 90% population confidence bound**. Selection frozen BEFORE opening any heldout dataset, no task accuracy tuning.

## First genuine QA task-answer accuracy, all labels never read until candidate choice frozen

**Zero-shot fixed scoring** on original pretrained Pythia14m teacher and all 4 actual restored physical INT6 variants (plain + 6/12/24% CR47), NOT a Jessica production model. Exactly 32 novel disjoint heldout examples per task:

- SciQ TEST[128:160] (earlier SciQ TEST[:128] previously exposed): `Support, Question, Answer:` and four answer candidates, deterministic question-hash order; correct-answer label not included in prompt.
- ARC-Easy TEST[:32] (different split from prior ARC-Easy VALIDATION[:128]): `Question, Answer:` plus candidates; evaluate answerKey by label, never provide it to model.
- SuperGLUE BoolQ validation[256:288] (previously evaluated validation[:128]): `Passage, Question, Answer:` with two possible answer strings "yes", "no"; hold labels only for postdecision accuracy.
- HellaSwag validation[256:288] (previous HellaSwag validation[:128]): `ctx` as prompt, each `ending` independently scored, label used only in metric.

For every candidate, actual causal LM teacher-forced *mean per-candidate token log probability* using the same prompt, cap 128 prefix Pythia tokens and 32 option tokens, no answer tuning or prompt search. Choose max option normalized log-likelihood (tie low option index). Compare **correct-count /32 per task**, teacher and all four physical INT6 variants, preserving every negative. No model training on heldout. Only 32 items per task so neither statistical significance nor real production Jessica task quality can be claimed; this is the first **true answer accuracy** evidence rather than causal text PPL. Independent checker recomputes 1,024 TRAIN bootstrap draws in a separate implementation, validates task sample SHA and correctness sums and cross-CPU prediction reproducibility.

## True full-model speed constraint

On Ubuntu22.04 + Ubuntu24.04 pinned AVX2 CPU, full original Pythia vs TRAIN-selected actual physical CB46+CR47 runtime: **128-token prefill +32 autoregressive KV-cache forward steps**, 2 warmups/7 measured repeats, selected/original <=1.20 latency ratio (user's hard first-stage goal). No claim of native 6-bit packed matrix multiplication, lower active RAM or Jessica production acceleration. Every physical source/patch byte and CRC/SHA checked; third independent GitHub job recomputes selection and task-scoring metadata. Integrity CI PASS != guarantee of quality improvement or speed gate.

**No changes to production Jessica, main, Render, Supabase, APK, user chats, settings or secrets.** PR kept draft, research branch only.
