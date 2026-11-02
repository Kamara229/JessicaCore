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
