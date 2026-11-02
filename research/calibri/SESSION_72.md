# CALIBRI v72 — measured answer correctness, not merely next-token KL

**2026-11-03; preregistered before CI measurements. Research-only.**
No main, Jessica live chat, Render, Supabase, APK, settings or secrets changes. All 3 full model variants execute in independent processes with source SHA validation and original pretrained source never present in packed student processes.

## Evidence requiring this step
v70 whole-model TEST mean KL and NLL improved modestly for a TRAIN-only-selected GQA-safe 2D Hadamard V/O representation at *identical real whole trained archive bytes*. v71 independently repeated the **direction of mean KL/NLL improvements** on WikiText and AG News, but both paired 95% bootstrap intervals included zero, model absolute language NLL remained extremely bad and native CPU execution did not become faster. These do not establish competence or actual correct question answering. Avoid mistaking source geometry for model intelligence.

## Fixed actual pretrained source / exact physical accountability
- Pinned original `HuggingFaceTB/SmolLM2-135M-Instruct`, revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`, unique source N=134,515,008.
- Load exact original CB59 file from verified GitHub v59 action 41507201766. Regenerate v70 predeclared sources **without changing** quantizer/source-selection rules. Only `baseline` (identical original CB59 source) and frozen, original TRAIN-selected `l0_hadamard` are used; source archives each **11,229,303 actual bytes / 0.667839413 bpw including metadata, alias tables, CRC/SHA, FP16 row scales/codebooks and uncompressed 1D source tensors**.
- Create packed models from config with native C GEMV/embedding consuming original 5-bit indices and paired source-only 2D coordinate representation; no original trained 2D dense weight materialized in packed runtime.
- Separate FP32 pretrained teacher reference process. All sources run with exact same dataset examples/prompt hashes/label token IDs.

## Preregistered labeled QA protocol
- **SciQ** (pinned original official dataset snapshot, TEST rows **480–519**, 40 questions, four alternatives). Use question and options only, no supplied `support` paragraph; options shuffled by immutable `720000 + record_index`. Score ONLY the four actual one-token answer-letter candidate completions (A/B/C/D), one forward per question. Chance benchmark: 25%. This is a closed-book diagnostic, not official reproduced benchmark with fine-tuned answer policy.
- **BoolQ** (pinned official source, VALIDATION rows **480–519**, 40 yes/no questions). First 128 tokenizer tokens of passage + full question; score one-token constrained ` Yes` versus ` No` next-token completion. Chance benchmark: 50%. Truncation can omit evidence; **do not claim full-passage official BoolQ benchmark**.
- Both tasks are **new unseen labeled evaluation sources never used to train, fit, tune, select or compare the candidate transform**. Fixed prompt format and range; no post-hoc prompt tuning.
- Measure per-question correctness, mean correct-answer conditional candidate probability, teacher agreement, baseline vs selected accuracy and paired 3,000-sample bootstrap interval. Also exact McNemar p-value on discordances; with n=40, sensitivity is low. Distinguish **teacher** from **both real compressed** candidates and record if teacher performs close to random.
- This is **constrained-choice** answer correctness, not free-form response generation or multi-turn Jessica chat. No attempt to perform tests through working Jessica.
- Record actual CPU prefill time separately from the answer-accuracy metric; source runtime might be slower than original FP32. No speed SLA claim.

## Expected scientific interpretation
A positive result demands a reproducible higher answer accuracy against the intact CB59 baseline **without larger source bytes**; even then the model can be useless if both compressed accuracies remain near chance. Do not assert a statistical effect when exact paired tests and intervals are inconclusive. Negative answer accuracy is a crucial result and would require shifting main algorithmic effort from mathematically elegant V/O gauges to genuinely task-trained, direct-executable, ultra-low-rate multidimensional latent representations and rate allocation.

## Original user-provided constraints
CALIBRI only on isolated GitHub research branches; no main/production changes. At most one text-only Jessica test when directly requested, **none required for this stage**.

## Observed result
CI verification pending at preregistration. Results MUST be appended only after actual job completion; no estimated score counts.

## VERIFIED real model outcome — GitHub Actions 41535003201 (2026-11-02 UTC)

**Infrastructure/PASS, labeled ANSWER QUALITY/NEGATIVE OR INCONCLUSIVE.** Real pinned 134,515,008-unique-parameter SmolLM2-135M-Instruct, exact checksum-validated original CB59 and frozen l0_hadamard (the v70 TRAIN-selected 2D operator transform), both whole physical 11,229,303 B / 0.667839413 bpw. All trained 2D source weights absent from each teacher-free packed native C-GEMV student process. FP32 original teacher evaluated in its own process. Every QA sample was fixed in the preregistration; no label-based training or selection.

| Frozen actual constrained-answer benchmark | Full original FP32 teacher | Packed unchanged CB59 | Packed layer0 Hadamard |
| --- | ---: | ---: | ---: |
| SciQ official TEST questions 480–519 (40 four-choice, no support) | 8/40 (20.0%) | 8/40 (20.0%) | **8/40 (20.0%)** |
| BoolQ official VALID questions 480–519 (40 yes/no, passage first 128 tokens) | **29/40 (72.5%)** | 20/40 (50.0%) | **21/40 (52.5%)** |
| SciQ native CPU prompt-evaluation seconds (40 cases) | 6.62s | 51.49s | 51.73s |
| BoolQ native CPU prompt-evaluation seconds (40 cases) | 11.34s | 107.79s | 107.96s |

Signed selected-minus-original accuracy: **SciQ 0.0 percentage points; BoolQ +2.5 percentage points (1/40 extra correct answer)**. Both exact paired McNemar p values **1.0**, so absolutely no evidence of a statistically meaningful improvement at this sample size. Paired 3000-resample bootstrap accuracy delta 95% intervals: SciQ **[-0.20,+0.20]**; BoolQ **[-0.175,+0.200625]**, both include 0 by a huge margin.

**Negative finding:** Although the frozen multidimensional GQA source gauge improved mean next-token KL/NLL in v70/v71, that did NOT establish improved real answer correctness in v72. Original FP32 teacher achieved only 20% on this PARTICULAR one-token letter SciQ template (worse than 25% random chance), making this SciQ prompt particularly uninformative for model intelligence and demonstrating importance of evaluating the teacher. On BoolQ, original teacher scored 72.5%, both compressed sources only about chance, and one extra correct constrained-choice prediction cannot justify a quality claim. Both compressed CPU runs ~8.8x slower than original teacher on these prompt forwards; speed still fails practical requirement.

**Critical methodological limits:** Only 40 questions per task, 1 prompt style each, forced one-token letter/yes-no decision, SciQ without reference support, BoolQ evidence artificially truncated to 128 tokens, no generative/free-text answer evaluation, no robust answer-ability claim, no second model checkpoint, no multi-machine latency test. The publisher/record lists are heldout data sources, but benchmark protocol is explicitly a diagnostic, NOT an official leaderboard evaluation. Label/teacher corpora were not used to optimize compressed weights.

**Decision:** Do NOT promote v70 Hadamard into a working LLM based on the small KL gain. Shift research emphasis to genuine functional/task-aware multi-dimensional source optimization and paid correction under the <=0.700bpw total physical source ceiling; preserve direct-native inference and measure wider out-of-distribution QA plus actual language generation. A candidate's accuracy must be compared with ORIGINAL FP32 teacher and baseline CB59; a tiny marginal apparent gain over near-random CB59 does not matter.

Official CI run https://github.com/Kamara229/JessicaCore/actions/runs/41535003201 ; research PR #202 (draft).
