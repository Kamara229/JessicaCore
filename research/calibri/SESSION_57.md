# CALIBRI v57 — paired task-answer margin calibration under INT6 physical bytes and 20% speed

Research-only, predeclared next step after completed v56 draft PR #186, GitHub Actions run 41492767498 (3/3 PASS): v56 required at least ONE extra TRAIN-correct answer before spending ~611kB in FP16 residual repairs; all three corrected candidates FAILED this strict coarse 0/1 metric despite improved TRAIN language perplexity. v56 chose smaller plain INT6 PRE 10,803,661 B = 6.143806 bits/parameter, and no change in some heldout correct-answer counts. The user explicitly cares about practical quality-speed compromise, therefore v57 tests a more sensitive *continuous* but still fully supervised **TRAIN-only** gold-vs-best-wrong answer likelihood margin. No v55/v56 previously revealed TEST record may influence the rule, chosen correction, or calibration.

## Physical storage — unchanged / reference quality and speed
Pretrained EleutherAI/pythia-14m complete physical INT6 CB46 source exactly 10,803,661B and genuine deterministic source-verified physical CR47 FP16 teacher residual row repairs, quotas cap6 610751B, cap12 610715B, cap24 610875B including CRC, SHA, model parent binding and true headers. Exact source+patch+15480B future POST capacity <=6.50 bits per ALL pretrained source parameters. CB46 SHA and all CR47 SHA required identical across Ubuntu 22/24 AVX2 independent CPUs; no JSON-only or shape-only counting. Fits priority <20% warm CPU penalty (batch1 128-token original context prefill+32 KV autoregressive decode, 2 warmups+7 samples), measured original FP32 teacher vs physically restored TRAIN-chosen model on each host. Physical INT6 weights remain *at rest*; whole model restored to dense FP32 active hot RAM at cold load, so no native packed/RAM savings claimed.

## Fixed primary TRAIN gate
Exactly v55/v56 four-family 1024-draw paired bootstrap NLL gate on new-to-that-protocol WikiText TRAIN token slices [16384:20480], AGNews TRAIN[1536:1792], SciQ TRAIN[512:768], BoolQ TRAIN[256:512], two nonoverlapping 2048-token windows/family. Gate accepts each source-only physical patch if >=3/4 corpus family NLLs positive, worst family >=-0.008, deterministic 10th percentile 102-of-1024 NLL macro gain >=0.0005. It is a heuristic, NOT formal confidence. Keep identical to v56.

## Fixed secondary TRAIN-only task answer **margin** gate
Never-used task TRAIN records (different from v56's earlier 32 TRAIN cases) 64 items/family:
- SciQ TRAIN[800:864]; previous CALIBRI exposed TRAIN up through 800.
- ARC-Easy TRAIN[32:96], disjoint from v56 TRAIN[:32].
- BoolQ TRAIN[544:608], disjoint from v56 TRAIN[512:544].
- HellaSwag TRAIN[32:96], disjoint from v56 TRAIN[:32].

For each task record, use the **unchanged v55/v56 zero-shot scoring** with actual Pythia causal-token `mean log P(answer | prompt)`, prefix<=128 and continuation<=32 tokens; per-task option-set independent each candidate. Define `gold_margin = log P(correct answer) - max_{wrong answers} log P(wrong answer)`. Gold labels used ONLY for TASK TRAIN margin optimization of model-choice acceptance (never fed to model, never to source-only row rank). Evaluate plain INT6 PRE and every physical FP16 repair alternative, 64 labeled TRAIN items each. Compute `gain = candidate_gold_margin - PRE_gold_margin`, mean separately per four families, equal-family macro average. Use 1024 seeded (57) paired bootstrap draws: sample four families with replacement then one of four contiguous **16-question TRAIN blocks** per drawn family; draw's mean of mean option gold-margin differences. Frozen candidate eligible only if ALL:
- identical v56 primary NLL family bootstrap approves candidate;
- >=3/4 TRAIN task family mean gold-margin differences **strictly positive**;
- equal-family mean margin gain >=+0.001;
- worst single family margin gain >=-0.020;
- sorted lower decile index 102 across the 1024 TRAIN block bootstrap mean gains >=+0.0005;
- no TRAIN task family loses **more than one correct** answer among 64.

Choose eligible candidate highest lower decile then macro mean then smaller correction cap; if none, choose plain no-extra-byte INT6 PRE. A continuous margin can indicate changes missed by strict integer threshold on 32 questions; it is NOT proof of unseen answer correctness. Do not retune any thresholds after reading newly heldout results; if sensitivity fails, that negative is real.

## NEW blind heldout accuracy — read only AFTER BOTH TRAIN gates
128 first-use disjoint labeled records per actual QA family, same frozen zero-shot scoring:
- SciQ TEST[224:352], disjoint from prior TEST[:224].
- ARC-Easy TEST[96:224], disjoint from prior TEST[:96].
- BoolQ validation[352:480], disjoint prior validation[:352].
- HellaSwag validation[352:480], disjoint prior validation[:352].

For each original pretrained teacher, physical plain PRE, and all three CR47 repairs, report correct/128, per-item correctness, SHA256 of record prompts+choices, mean gold margin and every regression. No labels in prompts, no adaptation, and no retroactive TRAIN threshold tuning. Only 128 per family, not meaningful proof of real task quality or significance, and raw Pythia14M is NOT Jessica's deployed model.

Independent third GitHub Actions verifier must recompute v56 1024 NLL family bootstrap and v57 1024 TRAIN block margin bootstrap independently of the selected model code; cross-host SHA all models and task records and heldout per-item correctness; count actual physical bits and full model 2+7 warm prefill+decode. Data/training protocols fixed in this document before first run. No main, production Jessica chats, Render, Supabase, APK, credentials, settings or user data changed.

## Input normalization amendment before TEST scoring (CI source-schema failure)

The first CI attempt failed while loading the *new TRAIN* SciQ[800:864] (before task TRAIN scoring, before TEST access) because the public SciQ dataset contains some records with exactly duplicate option text. This is a dataset schema issue, not a signal to tune the model or labels. The frozen protocol now **collapses exact duplicate SciQ answer strings** in deterministic question-hash order before scoring, and locates the gold correct answer by its original string; applies identically to TRAIN and TEST data without inspecting predictions or correctness. Every included record retains a valid uniquely identified correct answer and >=2 distinct options; if not, CI fails. No population quality threshold, candidate rate, selection cutoffs, sample ranges or storage budget changed. This amendment is clearly disclosed and occurred BEFORE any heldout task data could be opened.
