# CALIBRI 2.0 S102 — scientific transfer audit of original CALIBRI V research

**Date:** 2026-11-10. **Research only.** The old CALIBRI V series
compresses an original **pretrained SmolLM2-135M-Instruct** (134,515,008
unique weights). CALIBRI 2.0 S102 is a **36,380,480-parameter scratch
model** with a separate S101 frozen 3-profile policy. Neither its tokenizer
matching nor equal bpp licenses directly importing old 135M learned
vectors, teacher hidden activations, answer templates or old QA samples.

See `V_LEGACY_INVENTORY.md` for the 144 V source directories and 118 root
SESSION documents; this file extracts falsifiable **techniques**, not an
automatic selection of historical frozen sources.

## Classification of evidence (highest-value facts)

| Original V study | Actual measured evidence | What to do for S102/S103 | Do NOT infer |
| --- | --- | --- | --- |
| V59/V64/V179, physical coding | Original 135M K32V8 native CB59 = **11,229,303 bytes / 0.667839 bpp**. V179 211-index entropy 4.974866/5 bits; LZMA saves only 1.36% index bytes. | Keep physical full archive+sidecars+manifest integrity; change representation/learning, not generic zlib on already high-entropy 5-bit IDs. S101 leaves only **~1,940 bytes** headroom at 0.699573 bpp. | Compression of already packed indices cannot pay for a dense universal 10MB controller. |
| V27/V42, reproducibility | Same physically identical model SHA did not guarantee bitwise identical FP32 PPL between different CPUs; V41/V42 source quality remained far below higher-rate controls despite tiny training gains. | Pair each S102 CPU speed and generated tokens **same-run**, pin torch/OS/CPU, compare numerical outputs per identity and retain negative results. | Tiny PPL/KL deltas are not answer quality. |
| V120–V123, Green architecture | Standalone V120 byte-Transformer **8,223,732 bytes** and V122 **8,228,664 bytes**; V123 ran exact same frozen controller on real 14M GPTNeoX and packed 135M, but changed bad continuations rather than fixing them. V122 shuffled question produced identical generation in its 16 probes. | Optional **separate research track**: normalized-depth/role shared 6-coefficient control, question-shuffle intervention, measure real answer delta. **Do not graft these weights into S102** (not same as two internal rank8 layer-3/9 residuals). | Shared file ABI across hosts != task-conditioned improvement or proven universal recovery. |
| V124/V125/V126, learned critic | Hard-negative V124 rejected **38/40** actual bad outputs but accepted only **3/20** true answers. Pairwise V125 reached **70% balanced classification**, true acceptance **14/20**, plausible wrong rejection only **8/20**, gold-given offline oracle rank1 **11/20**. On third frozen GPT2 host V125 rejected **16/16** wrong outputs but only **4/16** plausible false references. | If introducing verifier, train **balanced same-question gold / plausible-wrong / real-model-wrong contrasts** and validate both false-positive and false-negative rates. Require candidate-generation mechanism; a binary critic alone cannot create answers. | Offline oracle correct candidate being supplied is NEVER real generative improvement. |
| V127/V128/V129, incorrect first-token signal | Original 'correct gold first token' often meant just **whitespace** ID 216. V128 corrected autonomous content-token top32 on original v119 from historical apparent **10/24** to **4/24**; real complete answer only **1/24**. | S102 metric must score content token, EOS, genuine autoregressive prefix, actual token IDs; compare both variants of answer tokenization without treating an injected prefix as generated. Read outputs, not just first-token success or confidence. | Leading space or forced prefix is not model understanding. |
| V130, V186, V206, V228, paired counterfactuals | Literal words identical with swapped owners should reverse correct answer. V186 produced **3/12** first-token opposing pairs both right despite weak general language. V228 trained gold answer pairs while preserving **31/48** frozen correct whole TRAIN replies. | Add **pair-level both-right** guard and semantic binding switch tests on authorized TRAIN and, later, once-only independent new questions. No free pass from always-'Yes', always-'red' or one predominant token. | A single fixed answer can accidentally score on one half of a pair. |
| V165, top-K information ceiling | Teacher-top1 among compressed CB161 top128 only **11.5% Wiki / 9.5% News** (historical FIT); even top2048 **60.7% / 37.3%**. | Diagnose **what the actual correct content token is reachable from before intervention**. Prefer upstream representation/actual context-based copying if ranking cannot produce the missing token. S98's prompt-token copy is a genuine, domain-specific mechanism to preserve. | 'A better reranker' cannot choose truth not in any candidate set. |
| V166/V182/V183, causal localization | Original FP32-teacher hidden-state transplants show cumulative behavior and cross-domain effects; replacing only one early/middle/late hidden stream is NOT uniformly sufficient. V182 located divergence already at shared input/head; V183 individual block repairs sometimes reduced early hidden error but worsened final logits. | On S102 analyze physical S93 layer-3/9 activations **and token-level downstream effects**, contrast shared input/head and first/second/EOS, then consider **placement/routing**, not blindly make rank8 adapters in every layer. Matched 36M FP32 needed for valid teacher-oracle diagnostics. | 135M FP32 hidden vectors are incompatible with 36M width and cannot be deployed as free neural weights. |
| V169–V175, V178/V181/V217–V219, source-MSE fits | Source-only weight MSE, re-clustering, same-rate codebook fit, higher-rate 2.54/5.04 bpp can improve RMSE or diversity while still yielding **0/24** correct; V175 **0/96** after millions of code flips. | Audit source fidelity only as *diagnostic*; accept by **generated correct answers** on frozen TRAIN and independent unseen, plus byte and speed guard. | Similar weights/low KL do not imply useful language. |
| V176/V184/V185, whole-system QAT | V176 trained 211 paid fields and collapsed to **one repeated token in all 96 DEV prompts** despite falling TRAIN objective. V185 changed **10,240** physical indices, got **1/24** correctly answered vs 0/24 original CB59, but only one distinct first-token ID. | Always monitor output *diversity, exact EOS, first divergence, full rollouts,* and first-token collapse. Codebook mutations physically real != quality improvement. | Low loss and native numerical parity alone are inadequate. |
| V188/V189/V192, baseline & scoring | Genuine original 135M FP32 produced **66/192 machine-confirmed**, 57 ungraded; 360M original FP32 **105/192** on same public DEV. V192 froze a **correct/wrong/ungraded** conservative grader; verbose valid answers were falsely rejected by strict bare-token rules. | Future CALIBRI 2.0 FP32 comparison MUST use a **matched 36M scratch FP32 ancestor/config**, when available, plus graded meaningful sentence-level answers. Treat ungraded ≠ wrong; publish raw real text. | Do not claim parity/failure using 135M pretrained FP32 as mathematically matched control for scratch 36M. |
| V197/V198/V200/V202/V204/V206, no-forgetting and rollout | V196 trained 211 dictionaries but collapsed from **2/16 to 0/16**, repeating 'red'. V202 sampled per-step protection missed global prefix regression, resulting in full rollback; V204 moved to full global guard per step. V200 trained on **student-own greedy trajectories**, not only gold teacher-forced prefixes. | Port J7 **whole 353/full+387/first** protection, real S101 own-prefix rollout and early error; do not accept a checkpoint based on sampled replay or synthetic anti-repeat metric. Revert Adam moments too. | Greedy first token preserved != whole answer preserved. |
| V212/V213/V215/V216, unsuccessful teacher paths | FP32 teacher trajectory V212 scored **2/16** vs v206 **3/16**; native output correction V214 **2/16**, V215 full-vocab teacher-fit **0/16** despite TRAIN token gains. V216 1-bit tied-head residual **2/16** and only **59% CPU**. | Use teacher data only as TRAIN diagnostics, not a hidden decoder; avoid slow output head sidecars, preserve S101 cached path; optimize **actual S98+S95-selected discrete policy** not just teacher logits. | Teacher logit similarity cannot replace correct autonomous answers; larger correction can destroy CPU budget. |
| V220–V225, late-token diagnosis | V221 found v206 and v220 had all **48 identical first tokens** and both had **40/48 correct first tokens**, yet full answers diverged: **12 second-token** differences; V224 found 5 late adjacent repetitions at positions 3/4 after earlier divergence. | Prioritize first **incorrect content token** by position, EOS/timing, and inspect **late-loop causal ancestry**, rather than only first-token color score. | First-token equality does not certify sequence equality. |
| V226–V230, don't suppress valid repeats | V226 removed all repetitive sequences but degraded full TRAIN score **31→29/48**; **four of five repeats matched the gold answer**. V227 then still failed to improve; V228 step12 preserved 31 full and reduced harmful repeat count **5→3** but did not exceed 31 full. V230 lost seven protected full despite repairing six wrong at one step. | Classify repeated tokens as harmful **only when they disagree with the true TRAIN target or prevent EOS**. Optimize full correct with strict no-forgetting; do not force anti-repetition unconditionally. | 'Fewer repeats' alone is not quality. |
| V231/V232, independent collapse | Held-out V231 pretrained 135M FP32 **326/960** correct; compressed V228 **0/960** using choice-letter prompts, with **901/960 'Yes'** in V231 report. V232 was post-hoc diagnostic of natural-format outputs, not new independent evidence. | Add per-answer dominant-token/mode-collapse alarms, balanced choice/Yes-No, prompt-format ablations and never post-hoc tune on revealed heldout questions. | Old original 135M compress path is **not** a proven high-quality sub-bit LLM. |

## **Critical architecture distinction: two different things called Green Tunnel**

**Original CALIBRI V120–V126**: separate 8.2MB fixed-size
byte-based neural critic / question-conditioned gain controller that emits
six cross-layer coefficients. Its hypothesis is reusable weights across
different host architectures (GPTNeoX, GPT2, Llama-like), but its
useful transfer failed or was very weak. The shared checkpoint is
physically separate; user-specified host-only bpp may exclude it ONLY if
actual combined bytes and cost are stated.

**CALIBRI 2.0 S87–S102**: two *internal* rank8 residual adapters
in layers 3 and 9 of a **12-layer scratch 36M** core, jointly trained
with two existing FP16 centroid dictionaries. S101 also has a separate
three-profile bank, prompt-only S95 router and S98 source-token copier.
S102 accounts for all runtime files **inside** the physical inclusive
0.7bpp. The original standalone 8MB V neural network cannot be inserted
without breaking current S102 budget and ABI. Transfer **science and
tests**, not incompatible numeric checkpoint.

## What S102 already established before this audit

- Stage-A real S101 CPU 456 identity reconstruction CI
  [#42845071504](https://github.com/Kamara229/JessicaCore/actions/runs/42845071504):
  **353/456 full+EOS**, **387** historically correct first, profiles
  S87:15/15 S91:5/5 S93:333/436, physical **0.699573397602 bpp**,
  **~1,940 physical bytes margin**, numeric Adam rollback selftest PASS.
- Stage-B REAL MPS 8-step S93 core+internal Green Tunnel, final reported in
  [#42847155291](https://github.com/Kamara229/JessicaCore/actions/runs/42847155291):
  proposed **348/456**, **five** old correct lost, **zero** gained,
  first-token losses **0**, physical **0.699573837398 bpp**,
  +2 source/sidecar bytes, guarded **rollback verified**. Relative
  S101 frozen candidate CPU ~1.02x in paired full456; previous concurrent
  run had ~0.845x, so report this ratio as a **noisy diagnostic**,
  not native matched 90%-of-S82 proof.
- GitHub Actions success means the **safety code ran**, not that a
  proposed model passed. Original S101 numeric weights remain the
  development reference; S102 research-only rejected prototype is
  not a scientific quality win.

## Concrete changes to the hybrid S103 protocol (priority order)

1. **READ-ONLY frozen S101 + S102 candidate TRAIN trajectory audit**:
   record first wrong token position for all 456 (first vs second content
   vs EOS vs late), old-correct-loss cause for five, repeat classification
   against gold, and frozen S95 router/S98 override activity. Use
   existing signed research artifacts; do not consume new independent QA.
2. **No blind CE on the selected S93 core**: S93 handles 436/456
   TRAIN but many S98 outcomes are determined by a 0.95 **hard threshold**.
   The teacher-forced `s91.policy_loss` optimizes the S82 differentiable
   approximation, **not the full final S101 choice**. Study whether the
   five losses arose in no-override routes, or near S98 threshold/first
   token instability. Choose a gradient objective aligned to the active
   route and valid EOS, *not* original 135M KL.
3. **Falsifiable narrow MPS intervention**: use smaller ≤2-step
   atomic transactions and supervised paired wrong examples chosen
   exclusively from already authorized S93-routed TRAIN. Every candidate
   repack/freeze and check **all 353 correct full+EOS and 387 first**;
   if no safe gain, **do not keep training by sheer step count**.
4. **Gold-aware repetition and lexical correctness**:
   log target and prediction token IDs incl EOS; distinguish a valid
   repeated name/number from wrong self-loop (V226/V228); never count
   whitespace-only or default 'Yes'/'red' as semantic success.
5. **Causal placement study BEFORE all-layer growth**:
   compare original 36M matched FP32 and S101 compressed features on
   authorized TRAIN; test gradient/activation relevance of layers 0,
   3, 9, 11 and tied embedding/head as *diagnostic* only. Do not port
   old 135M hidden tensors or claim oracle repair is deployable.
6. **Genuinely independent final quality**: only after freezing
   enhanced hybrid candidate SHA, run ONE new pre-registered heldout
   domain-balanced unseen sample; preserve all S100/S96/V231/V232
   previously inspected sets as permanently embargoed. Evaluate full
   real answers (with ungraded uncertainty and human review where
   needed), swapped binding, no default-token collapse, correct EOS,
   compared with matched scratch 36M FP32 and speed budget.

## No-go rules

- No rank8-in-all-12-layers expansion without first proving S102
  baseline and finding **actual** ≤0.7bpp storage and full-CPU time.
- No 8MB original Green V critic hidden in the source budget.
- No direct copying of V-vintage original pretrained 135M tensors to
  scratch 36M; no private QA leakage.
- No merging to `main`, no Jessica tests, no Render, Supabase, APK or
  LIVE changes. No automatic promotion of negative prototypes.

## Traceable source evidence

Permanent docs:
`research/calibri/SESSION_59.md`,
`SESSION_120_GREEN_TUNNEL.md`,
`SESSION_123_SAME_GREEN_DIFFERENT_LLM.md`,
`SESSION_124_SHARED_GREEN_REAL_NEGATIVES.md`,
`SESSION_125_REAL_CONTRASTIVE_GREEN.md`,
`SESSION_126_UNSEEN_HOST.md`,
`SESSION_127_ORACLE_STEER_CAPACITY.md`,
`SESSION_128_TOKEN_AUDIT.md`,
`SESSION_129_CALIBRI_PRO_MAX.md`,
`CALIBRI_FROZEN_BASELINE_2026-11-04.md`,
`results/v176_20261105/README.md`,
`results/v181_20261105/README.md`,
`results/v185_20261105/README.md`,
`results/v186_20261105/README.md`,
`results/v188_20261105/README.md`,
`results/v189_20261105/README.md`.

Signed GitHub sources and experiments: V165 PR #302; V166 PR #303
CI #41925484368; V200–V231 PR #339, #342–#371 and V231
PR #372 CI #42344548617; V232 PR #373 CI #42347056055;
V228 PR #369 CI #42336914724; S102 hybrid PR #506
CI #42847155291.

**Audit conclusion:** The most valuable V legacy exports are **methodological**:
early divergence localization, exact first CONTENT token/EOS, source+runtime
parity, TRAIN-only atomic rollback, correctly grounded paired counterfactual
quality, and mode-collapse/repetition diagnosis. Importing the
old standalone Green weights is scientifically unjustified and cannot
fit S102's current physical runtime budget.
