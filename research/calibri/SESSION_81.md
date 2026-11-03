# CALIBRI v81 — projected answer-conditioned learning with explicit two-corpus KL ceiling

**Experiment pre-registration, 2026-11-03, before v81 TEST.** Isolated GitHub draft research branch and no production Jessica, main, Render, Supabase, APK, deployments or user settings modified. Not a Jessica ordinary-chat test.

## Empirical motivation from successfully completed real training
- CALIBRI v79 original pinned pretrained 135M source GitHub run `41550969128` successfully built full source-complete CB76 **11,725,487 B / 0.697348923 bpw**, including entire original CB59 pretrained source (11,229,303 B), paid fp16 output rank-4 U/V and bias and checksums. Source 5-bit original pretrained 2D matrices stay C native in memory; paid rank-4+logit bias executes via CPU Torch.
- **No v79 answer-supervised source was promoted**: SciQ TRAIN VALID [400,412) and BoolQ TRAIN VALID [800,816) plus two language KL safety guards selected unchanged v76 parent step0.
- Verified separate no-TEST-leak diagnostic v80, run `41554768160`, independently reconstructed all source-only TRAIN VALID gates. v79 step18 **improved both small QA TRAIN VALID slices** (+1/12 SciQ; +4/16 BoolQ), but increased teacher→packed language KL in WikiText by +0.26440 and AGNews +0.50674 nats, exceeding the fixed +0.25 ceiling on both. Earlier steps regressed SciQ and AGNews. Thus simple low-weighted language CE regularization fails to enforce constraints in practice.
- On independent v79 labeled TEST, original teacher scored supported SciQ 23/27 vs original compact CB59 7/27 and paid CB76 9/27; truncated BoolQ teacher21/32, original compact13/32, paid12/32. Generated paid tokens highly repetitive (unique ratio ~0.0694). KL improvements are not restored semantic ability. Never reuse those heldout v79 TEST labels to select a v81 source.

## True new v81 hypothesis
**Can hard-projected every-step updates, not only fixed soft teacher CE penalty, improve answer correctness while maintaining the real original teacher→compressed full-model language KL ceiling?** This is not a new compression codec: the same physically paid output rank4 source is trained with a new constrained functional optimizer. It might still fail due to severe early-layer corruption. If so, do not promote.

### Frozen source and actual model
- Pinned true original trained `HuggingFaceTB/SmolLM2-135M-Instruct`, revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`, **134,515,008 original unique pretrained parameters**. Download original source artifact from Github Actions run 41507201766, checksum validated CB59 original **11,229,303 physical B**.
- Recreate original paid CB76 snapshot from original v76 true packed+teacher WikiText/AG News TRAIN FIT and TRAIN VALID pipeline, before any v81 QA TEST. Preserve its independently fitted SHA, full pretrained 5-bit source and all source alias metadata.
- **All physically runnable snapshots** are exactly **11,725,487 bytes = 0.697348923 bpw** per original unique pretrained parameter. The complete original pretrained CB59 source, paid FP16 `U[49152,4]`, `V[576,4]`, vocab bias `b[49152]`, CB76 header, CRC and SHA are in the source. No hidden teacher, feature cache or extra side metadata in inference file. Native C direct 5-bit GEMV for original pretrained 2D source, paid rank4 factor operator in CPU Torch. No claim fully native C, speed superiority, or assistant-level competence.

### New actual TRAIN FIT and independent QA TRAIN VALID
Use genuine original compressed-model final hidden trajectories and token logits from source-specific teacher-free native 5-bit model, not original pretrained teacher hidden states; original pretrained teacher used separately for language KL. Fixed input SciQ source support truncated to first 320 tokenizer tokens; exact full-answer-string token likelihood. BoolQ original passage first256 tokenizer tokens then exact Yes/No label token. Official original rows (disjoint from prior v79 and each other):
- **QA FIT:** SciQ official `train[1050:1082]` (32 source questions); BoolQ official `train[2500:2564]` (64).
- **QA TRAIN VALID:** SciQ official `train[1180:1204]` (24); BoolQ official `train[2700:2732]` (32).
- Uniform validity exclusion (duplicate SciQ options and empty/very long answers) based on original source **before any model answer scoring**, exact source row IDs logged and denominators adjusted; no selecting cases for performance.
- Initial paid v76 factors copied from frozen real TRAIN-only source. Adam original FP32 optimization workspace, seed 81, lr 0.003, 18 steps, gradient clip norm5, loss `0.5*SciQ full answer-choice CE + 0.5*BoolQ yes/no CE + 0.10*mean pretrained teacher original TRAIN FIT full-vocab CE (32 source positions each for WikiText/AGNews) + 0.003*factor deviation MSE`.
- **Hard constraint after EVERY optimizer step:** compare candidate fp16-rounded factors against parent source on the original v76 `wiki_valid` and `news_valid` **TRAIN-CAL** sets, entire original teacher→native compressed full-vocab KL. Each must have `KL_candidate − KL_parent ≤ +0.25 nats`. Instead of accepting an infeasible update, predefined backtracking alphas `[1,0.5,0.25,0.125,0.0625,0.03125,0.015625,0]` along previous-feasible-to-Adam-proposed factor vector. Select FIRST feasible fp16-snapped update, else revert. Note that the two language TRAIN-CAL sets have been used in v76-v80 and are **NOT** independent external validation. Constraints are only claims about these old TRAIN sets, not about new unseen TEST; no cheating by calling them fresh test. This is projected line-search over factor updates, not mathematical orthogonal projection onto a convex feasible KL manifold; do not mislabel.
- Freeze actual physically quantized checkpoints step **[0,4,8,12,18]** before QA TEST. On the new independent QA TRAIN VALID, candidate eligible only if both SciQ and BoolQ correctness non-regressing, at least one absolute accuracy gain **≥0.05** (subject to float tolerance), and both full-source TRAIN-CAL KL constraints still hold. Among eligible choose maximal sum of task accuracy gains, lowest step ties. If none, retain exact bitwise original paid v76 source. Never change criteria or promote due to TEST. All QA FIT/VALID questions remain original `train` split.
- Run the same optimizer twice in independent Python processes on same frozen original native source feature arrays, seed and CPU host. Record chosen source full SHA, 3×FP16 factor max-diffs, exact source byte budget. Existing prior SHA differences across separate CI hosts remain unresolved; same-host match does not prove cross-host repeatability.
- Freeze source including complete-original-file SHA **BEFORE any TEST is opened**. Never train or adjust rank/steps, alphas, gating or source storage on labeled TEST.

### NEW independent heldout TEST task/word data, frozen in advance
Four **actual** full pretrained model processes: original FP32 teacher, original packed CB59, original paid v76 parent CB76, new v81 TRAIN-selected or reverted CB76.
- **SciQ official `test[500:528]`** — original-support-assisted full-length answer-option scoring, 28 before source-validity exclusion. Original pretrained teacher task accuracy is essential protocol sanity control.
- **BoolQ official `validation[2350:2382]`** — first256 passage tokens, yes/no, 32 before validity checks; not the official full-passage benchmark.
- **WikiText2 official test tokenizer tokens `[115000,115512)`** and **AG News official `test[7200:7300]`** text only first512 tokens, each 8×64 contexts /504 next-token positions. Disjoint from all previous v79 heldout source spans. Report signed teacher→student KL, actual next-token NLL and CPU prefill time.
- Three NEW fixed prompts `One result from the research is that`, `When the traveler returned to the village,`, `A careful proof begins by explaining`. True autoregressive **KV-cached 24-token generation**, post-prefill tokens/sec, mean distinct ID share and modal token fraction. Continue fixed load after EOS in microbenchmark to keep runtime comparable; not free-form assistant correctness.
- Report exact QA correct counts, paired McNemar bootstrap confidence caveats from original established evaluator, physical source SHA, every training failure, and whether v81 actual new source was promoted by TRAIN. If frozen TRAIN winner is parent step0, TEST metrics from v81 and original paid source MUST be identical; do not call it a model improvement.
- Scientific success requires actual improved independent teacher-sane QA while preserving acceptable KL/NLL and reducing catastrophic output repetition, not simply passing physical rate.

## Scientific limitations and exit
Even if constrained update succeeds on 24/32-question VALID, it might fail true generalization. A 0.697bpw model whose generated outputs repeat and whose supported SciQ is near guessing is **NOT comparable to useful original FP32 assistant**. Failure after this iteration suggests paid correction needs to target early/central transform layers under explicit novel source-geometry storage and native decoder with all bytes counted, rather than endlessly tune the final output 4D factor. The user-facing Jessica must stay unchanged.

## CI outcome
**Verified real pretrained 135M CI run 41555699743 SUCCESS**; measured truthful gains and regressions appended below. Stage remains draft research, not live Jessica.

## VERIFIED v81 actual independent pretrained source experiment — 2026-11-03 UTC

**Scientific result: MIXED — a genuine newly TRAIN-selected modified source significantly increased raw BoolQ heldout correct count in this small task slice, while SciQ barely changed, both independent text TEST KL/NLL regressed vs unmodified v76, and greedy token repetition remained severe. Do NOT promote as useful assistant.** Actual successful GitHub Actions run **41555699743**, all 23 steps successful. Research PR #211 isolated; main/Render/Supabase/APK/live Jessica unchanged.

### Real physical original complete pretrained source and frozen TRAIN selection

Original pinned complete CB59 source **11,229,303 bytes / 0.667839413 bpw**, pretrained 134,515,008 unique original parameters. Physically paid CB76 with entire exact original source, FP16 `U[49152,4]`, `V[576,4]`, `b[49152]`, 20B versioned header+CRC/SHA = **11,725,487 bytes / 0.697348923 bpw**, below 0.700 inclusive of all source 2D trained tensors and tied weights. Original pretrained source remains C direct 5-bit codebook kernels; paid output correction CPU Torch; **not** all-native or proven CPU speedup.

NEW QA TRAIN FIT from actual native-packed source: SciQ original TRAIN 1050:1082 **32 valid**; BoolQ original TRAIN 2500:2564 **64 valid**; 96 TRAIN FIT QA instances, **343 original-answer token features**. NEW independent QA TRAIN VALID: SciQ original TRAIN 1180:1204 **24 valid**, BoolQ original TRAIN 2700:2732 **32 valid**; 56 QA VALID instances and 213 original-answer token features. Neither the v81 SciQ/BoolQ TEST nor new test text was used to select source; all source snapshots frozen first.

**Projected hard two-corpus KL guard enforced at EVERY optimizer step on physically FP16-rounded factors** using previously seen original WikiText/AGNews TRAIN-CAL source; bounded each KL against original v76 CAL by +0.25 nats. This is a constraint on reused language TRAIN-CAL, **NOT** an independent bound on unknown TEST. Fixed optimizer trial snapshots 0,4,8,12,18, no TEST tuning. Candidate step18 was the **ONLY eligible** step: **QA TRAIN VALID** SciQ Δaccuracy **0/24**, BoolQ **+7/32** correct (∆accuracy +21.875pp); raw checkpoints steps4/8/12 slightly regressed SciQ (−1/24) and were rejected. Largest per-step enforced language TRAIN-CAL KL regression was **+0.249829 nats**, below +0.25 within original numerical precision. Both same-host independent optimizer repetitions selected step18 and produced exactly same frozen physical source SHA256 `94282c492713b053ccabc76510bad5f45791142866aa81abad62b35081e752c0`. Original complete source archive same length. Cross-host bitwise consistency is NOT proven; previous v76/v77 difference remains.

### Truly independent NEW TEST data (frozen only after TRAIN source selection)

Source real FP32 teacher, original native CB59, unmodified v76 paid rank4, newly selected v81 paid rank4 all independently executed as real pretrained complete models. SciQ original TEST [500,528): 28 rows, **27 source-valid**; support-truncated first320-token original context and full-answer-string normalized log-likelihood (not one-letter prompt). BoolQ original validation [2350,2382): **32** rows, first256-token passage, Yes/No forced choice. Neither is an official leaderboard or a large test.

| Actual heldout QA | Original FP32 pretrained teacher | Native complete CB59 baseline | Parent v76 complete CB76 | NEW projected QA v81 CB76 |
|---|---:|---:|---:|---:|
| **SciQ supported TEST (27 valid)** | **25/27 = 92.6%** | 3/27 = 11.1% | 6/27 = 22.2% | **7/27 = 25.9%** |
| **BoolQ TEST slice (32)** | **18/32 = 56.3%** | 13/32 = 40.6% | 9/32 = 28.1% | **17/32 = 53.1%** |

**Key:** v81 paid improved BoolQ from parent 9/32 to 17/32 (**+8 correct**, +25 percentage points) and from original CB59 13/32 to 17/32 (+4 correct), on this **32-question** truncated-passage heldout slice. This is a genuine *raw-count* improvement for this one pinned checkpoint under a fixed post-TRAIN test; statistical confidence is NOT claimed. The source QA teacher's BoolQ 18/32 is only moderately above chance, so caution warranted. SciQ gain from paid parent 6/27 to 7/27 is just +1; teacher far superior **25/27**, so useful scientific-answer competence still NOT recovered.

Independent actual text TEST source windows, 8×64 original sequences / 504 scored next tokens per domain:
- WikiText2 official TEST tokens [115000,115512): original CB59 teacher-KL **11.26366**, true next-token NLL **14.92902**; parent paid v76 KL **7.93840**/NLL **11.9475**; NEW v81 KL **8.30190**, NLL **12.33116**, i.e. **regression vs parent +0.36350 KL and +0.38366 NLL**.
- AG News original TEST rows [7200,7300): original CB59 KL **10.73905**/NLL **14.39932**; parent v76 KL **8.47333**/NLL **12.19657**; NEW v81 KL **8.60851**/NLL **12.31240**, i.e. **regression vs parent +0.13518 KL and +0.11583 NLL**.
- Both new models still substantially reduce these numerical errors relative to original CB59, but that *inherited parent improvement* does not cancel the **new regression**. The original TRAIN-CAL guard of +0.25 did NOT generalize to WikiText external TEST (KL +0.3635 > +0.25). **Do NOT claim hard universal teacher-KL guarantee.**

### Native-source CPU token speed and generative degeneration

Actually generated 3 new ×24-token greedy KV-cached outputs per model on one Ubuntu CPU; post-prefill generation:
- Original FP32 teacher **26.0643 tokens/s**; mean distinct generated token ratio **0.8611**.
- Original native source CB59 **53.3308 tokens/s**, distinct ratio **0.25**.
- Original paid v76 CB76 **54.3670 tokens/s**, distinct ratio **0.1389**.
- NEW constrained QA v81 CB76 **53.9555 tokens/s**, distinct ratio **0.1389**.
Thus raw cached decoding speed v81 ≈v76 within single-run noise, but the actual output is **still massively repetitive** (13.9% unique token IDs vs 86.1% original teacher). No QA/generation-speed generalization, no coherent-answers benchmark and no claim of useful assistant quality. Prefill and runtime paid Torch factor overhead also require independent controlled benchmarking.

### Decision and v82 next experimental gate

Keep v81 source as a **research candidate only**, not production-approved. Unlike v79 negative null, v81 shows hard projected guard *can* improve heldout BoolQ raw correct count while preserving unchanged 0.697349bpw physical source and C-native 5-bit trained base, but teacher-sane SciQ remains near guessing, generative repetition not fixed, and language KL on untouched WikiText TEST **violated its TRAIN-CAL ceiling**. As a result **the proposed objective is not yet a complete model-quality solution**.

Next v82 should pre-register genuinely NEW independent QA TEST and language corpora, larger per-task samples and uncertainty; assess run-to-run/host source SHA, *explicit two-domain heldout-like QA+KL and entropy/argmax preservation objectives* using only TRAIN, and objective that addresses repetition without destroying language KL. Investigate earlier-layer native 2D/8D source correction and heterogenous bit-rate allocation if output-only U/V capacity demonstrably inadequate. Do not tune to any existing v79/v81 disclosed TEST examples. Keep native C+paid Torch split accurately described and all physical bytes paid. No Jessica ordinary-chat test required.

Verified CI: https://github.com/Kamara229/JessicaCore/actions/runs/41555699743
Research DRAFT PR: https://github.com/Kamara229/JessicaCore/pull/211
