# CALIBRI v77 — strict paid-factor ablation, unseen answer correctness, true cache decoding speed

**Pre-registered before v77 TEST evaluation; isolated research branch / draft PR.** Neither live Jessica, main, Render, Supabase, APK nor production configuration is altered; no ordinary Jessica chat test is required.

## Why this stage
The verified **v76** experiment was the first true learned paid-correction source at a physically complete <0.700 bits per original parameter to show **large independent two-domain full-model KL/NLL improvements**. A pretrained SmolLM2-135M-Instruct original CB59 file (11,229,303 bytes) plus a fully paid rank-4 output operator correction and vocabulary bias (49152×4 FP16 U, 576×4 FP16 V, 49152 FP16 output bias), a versioned 20-byte header and CRC/SHA footer totaled **11,725,487 bytes = 0.697348923 bpw** for 134,515,008 source parameters. v76 independently selected the step20 **rank4+bias** snapshot from two TRAIN FIT and two TRAIN VALID corpora. WikiText heldout ΔKL −2.13015, ΔNLL −1.96659; AG News heldout ΔKL −3.13409, ΔNLL −3.22857, yet absolute language competence, quality of answers, and native autoregressive generation speed remain unproven.

**Questions v77 must falsify:** (a) Is it mainly the vocabulary bias, rather than multidimensional rank-4 learned U/V, which explains v76 improvement? (b) Does reduced teacher-KL actually increase **correct answers** on unseen labeled tasks? (c) Can the *same source-complete executable model* generate tokens at useful speed on CPU versus original FP32 and CB59?

## Fixed same-cost ablation; no new training or TEST-based selection
Re-run the original unchanged v76 algorithm *using its original frozen TRAIN input ranges* to regenerate exactly v76's step20 paid source. Require original v76 source SHA provenance, `chosen_source_mode=rank4`, `chosen_source_step=20`, original CB59 prefix identical and every source CRC/SHA valid. If replication diverges, FAIL rather than invent a new selected model.

Generate four **genuine whole physical CB76 model sources**, all exactly **11,725,487 bytes (0.697348923 bpw)**, with original full CB59 pretrained model unchanged, original packed C GEMV/embedding and physically stored paid FP16 values:
1. `full`: bit-for-bit equal to original v76 TRAIN-selected rank4 U/V **plus** bias; source SHA equal to original v76.
2. `bias_only`: identical bias, but all same-size rank4 factors U and V set to zero.
3. `rank4_only`: identical U and V, but output bias set to zero.
4. `zero_paid`: paid U/V/bias all zero, as a physically same-cost null-control.

Compare also `baseline`: original CB59 **11,229,303 bytes = 0.667839413 bpw**, no paid bytes. An original pretrained FP32 `teacher` (separate process) anchors quality. **Do not compare ablations by stripping unused paid bytes**; all four CB76 variants must be equally expensive, native runnable and complete, not a hypothetical rate. The paid low-rank factor operation is executed on CPU by Torch; only the original compressed K32V8 5-bit pretrained linear/embedding is native C. Do NOT mislabel this hybrid as native C end-to-end.

## New disjoint language TEST (not seen by the v76 trainer)
- WikiText-2 original TEST tokenizer tokens **[80000,80512)**, 8×64 windows, 504 next-token positions. Distinct from v76's previous [70000,70512).
- AG News original TEST records **[4000,4100)** text only, first 512 model tokens, 8×64 windows, 504 positions. Distinct from v76's TEST [3000,3100).

Evaluate all physically distinct source models against the **same true FP32 pretrained teacher** on full-vocabulary teacher→student KL and actual heldout next-token NLL; compare signed deltas and bootstrap 95% paired eight-window intervals. Original teacher logits temporarily saved FP16 in CI to control storage and its approximation disclosed. No FIT/VALID/TEST re-selection on these results.

## New unseen heldout labeled QA, preregistered formats
- SciQ official `test` original rows **680–727**, 48 items, closed-book four-choice question/options only (no support passage). Options shuffled with seed `770000+row_index`; score the actual next-token A/B/C/D label candidates. Chance ≈25%. This template yielded poor FP32-teacher performance in previous work, so explicitly record teacher competence and do NOT claim official SciQ benchmark or general science ability.
- BoolQ official `validation` original rows **960–1023**, 64 questions, choose next-token Yes/No conditional on original passage's **first 128 model tokens only** and the unabridged question. Chance ≈50%. A truncated-passage forced-choice diagnostic, NOT official full-passage leaderboard BoolQ.
- Fixed prompt templates for every exact original question, dataset revision SHA and prompt-token SHA. No fitting or tuning on labels. Measure correct counts, percent, teacher accuracy, paired exact two-sided McNemar and 3,000-resample accuracy CI vs original CB59. Distinguish apparent improvements of 1–2 examples from statistically meaningful gains; note small sample and task limitations.

## Real autoregressive speed gate
For each original FP32 teacher and each five packed source models, run fixed-source 3 short English prompts, true **greedy KV-cache token-by-token generation** with 16 tokens per prompt. Measure initial prefill and separate 15 subsequent one-token decode steps, aggregate tokens/sec; checksum generated token sequences and record fraction of distinct generated token IDs. To keep 16-token workload comparable this microbenchmark intentionally continues after EOS, if any; do not imply coherent chat or valid post-EOS generation. This is one Ubuntu CPU runner, not a controlled thermal-averaged deployment/SLA benchmark. Measure speed regressions instead of hiding them.

## Acceptance and interpretation
- All 4 paid **complete** on-disk files and their SHA/CRC must parse identically, with all original trained source matrices still packed native C; zero trained 2D dense source in paid runtime.
- Real pinned training reproduction must select v76 rank4+logit bias step20 before any of these TEST sources are opened.
- CI PASS means correct numerical measurement and physical integrity, NOT that semantic QA or speed succeeded.
- A stronger result than v76 requires full correction improve KL over bias-only on both unseen domains, improve at least one actual labeled QA task while not degrading the other **with uncertainty considered**, and not catastrophically worsen CPU generation. If bias-only performs comparably, acknowledge the 4D factor may not justify its 397,824 physically stored bytes; simple bias calibration may be the explanatory mechanism.
- If practical competence remains low, the project needs a new **global, direct-operable compact neural weight representation and learned rate allocation**, not incremental hidden-output head tweaks. No production deployment on these experiments.

## v77 CI result
Pending before initial PR run. Actual negative outcomes and measured CPU token/sec will be appended after checking independent GitHub Actions run logs.


## Protocol correction before any scored v77 answer evaluation
First CI run **41543185133** passed source recreation, physical archive integrity (4 equal-cost paid models, 11,725,487 B each) and teacher's independent full-model TEST text logit collection, but **FAILED before any SciQ/BoolQ QA score** due to repeated SciQ distractors in the official dataset rows (invalid distinct-4-option assumption). This is a **source-data validity failure**, not evidence about model quality. In a corrective commit, all original precommitted SciQ row indices 680–727 remain the designated sampling frame; any source row with non-unique four options is now uniformly excluded for **every** model *before any model answers are scored*, with exact excluded row indices logged as source metadata, actual denominator reduced, and all physical source candidates and TEST corpus ranges unchanged. No model accuracy was inspected to make the rule. V77 is therefore not a perfect original pre-registered protocol: openly report the data-quality correction and its excluded counts. CI must rerun and verify, not claim that the failed first run measured QA.

## VERIFIED real 135M trained-source evaluation — second CI run 41543812501, 2026-11-02 UTC

**GitHub Actions SUCCESS for source integrity and actual complete-model inference; answer correctness remains MIXED/NOT RECOVERED.** The first CI run 41543185133 failed due to one SciQ official TEST row containing duplicate answer choices, and was corrected by an explicit uniform data-validity exclusion *before any QA score was computed*. Excluded original SciQ row **718**, leaving 47/48 preregistered rows. No answer-performance-based sampling occurred; acknowledge deviation from the original 48-row intended test.

### Exact physical rate, original trained source and control
- Original trained CB59 archive source: **11,229,303 bytes / 0.667839413 bpw**, all 134,515,008 original unique pretrained parameters and tied aliases. Original TRAIN-selected v76 algorithm was rerun with the same pinned source/corpora/optimizer settings; original pretrained source SHA256 is **50eac24ba7d7a452c495f8adced2cdef3f9cc44e024217fc21d8f2cebb87413c**.
- Reproduced TRAIN-only choice: rank4 + bias, step20, worst two-domain TRAIN VALID original→corrected KL improvement **3.60648 nats**, consistent with prior v76 **3.60649**. All four paid variants (`full`, `bias_only`, `rank4_only`, `zero_paid`) are **real separately serialized complete files of identical 11,725,487 bytes = 0.697348923 bpw** including **496,184 bytes paid above the entire original CB59 source** (all FP16 latent matrices/bias/header/integrity checksums). Original trained 2D tensors stay in original native K32V8 5-bit direct C kernel; learned lowrank operation is Torch CPU, not falsely claimed fully native.
- **REPRODUCIBILITY LIMITATION (important):** V76 original experiment's trained full CB76 file SHA256 from its retained audit was `bf5eed8a06fb5d01843fca1bd23d62066ed04813403e71f9cba0ed80ebd1ee62`; identical algorithm and original TRAIN data re-run in v77 produced `4d5c1258924b77a458b1d08d962914f6c8845632f8604a0a04e929cd43ef759c`. The original CB59 prefix, corpus revision provenance, selected TRAIN step/mode, source size, and all factor shapes/nonzero counts MATCH, but **trained sidecar numerical bytes are not bitwise identical across the two CI runs**; worst VALID KL differs by only ~0.000015 nats. This is a real reproducibility defect/host numerical variance, NOT an identical frozen-weight model replication. The v77 `full` equals its OWN prior-to-TEST recreated CB76 source exactly, and all v77 ablations are internally controlled and selected without v77 TEST access. Need fix deterministic cross-machine factor training and retain a source checksum/artifact for exact future transfer.

### Truly new heldout original TEST language text, full native model and paired uncertainty
Each genuine source: 8×64-token contexts = **504 next-token positions per domain**, v77 TEST independent of earlier v76. Mean teacher→compressed full-vocabulary KL and actual next-token NLL (lower is better):

| Source | WikiText TEST [80000,80512) KL / NLL | AG News TEST records 4000–4099 KL / NLL |
| --- | --- | --- |
| Original CB59 | **10.98966 / 14.33392** | **12.11661 / 15.89241** |
| `zero_paid` (same-bit negative control) | **10.98966 / 14.33392** | **12.11661 / 15.89241** |
| `bias_only` | 10.58467 / 13.93874 | 11.69588 / 15.47018 |
| `rank4_only` | **8.09018 / 11.52590** | **9.06832 / 13.00589** |
| `full` rank4+bias | **8.03370 / 11.47922** | **8.99261 / 12.92868** |

Signed `full` minus original CB59: WikiText KL **−2.95596**, NLL **−2.85470**; AG News KL **−3.12400**, NLL **−2.96373**. Paid rank4-only versus baseline: WikiText KL **−2.89948**, AG News KL **−3.04829**, i.e. **MOST of the paid quality gain is supplied by the rank4 learned factor, not by the vocabulary bias alone**. Bias-only gains are only ~0.405 and 0.421 KL nats. The full combination yields an additional ~0.0565 WikiText and ~0.0757 AG News KL gain over rank4-only.

Paired eight-window full-vs-original KL bootstrap 95% (exploratory, non-independent texts) WikiText **[−3.6384,−2.2301]**, News **[−4.1913,−2.1659]**, both below zero. This supports real heldout cross-corpus next-token improvement on THIS 135M trained source, **NOT** usable model competence, statistical generality across models, task reliability or proof of novel weight theory.

### New heldout labeled SciQ and BoolQ: scientific negative/mixed outcome
Both source models and original full FP32 teacher answered EXACT SAME fixed prompt-token SHA and label token candidates; SciQ duplicate row718 uniformly skipped before scoring.

| Actual constrained QA | Genuine original FP32 teacher | Original packed CB59 | Paid bias-only | Paid rank4-only | Paid full rank4+bias |
| --- | ---: | ---: | ---: | ---: | ---: |
| SciQ TEST original rows 680–727, one duplicate rejected (**47 valid**) | **7/47 (14.9%)** | **10/47 (21.3%)** | 9/47 (19.1%) | **17/47 (36.2%)** | **17/47 (36.2%)** |
| BoolQ VALID rows 960–1023, first 128 passage tokens (**64**) | **39/64 (60.9%)** | **26/64 (40.6%)** | 26/64 (40.6%) | **22/64 (34.4%)** | **22/64 (34.4%)** |

**No robust answer correctness success:** SciQ paid full vs original CB59 **+7/47 correct**, exact paired two-sided McNemar **p≈0.2295**, 95% paired bootstrap accuracy-delta **[−0.0638,+0.3617]** includes zero. Worse, the original trained FP32 teacher achieves only **7/47 (below 25% chance)** under this nonstandard single-letter prompt, making this particular SciQ format unsuitable as a reliable competence test. BoolQ paid full vs original CB59 **−4/64 correct**, McNemar **p≈0.5413**, paired accuracy-delta interval **[−0.2031,+0.0938]** includes zero, also not a reliable effect but **negative direction**. The fully trained original teacher does better than either compressed model on BoolQ. These were forced single-token choice answers, SciQ sans support, BoolQ truncated passages: **not official benchmark scores nor free-form text quality**. Distinguish significant numerical KL improvements from unproven QA competence.

### Measured real autoregressive KV-cached token generation and repetition
Each model performed genuine greedy cached next-token **3 fixed prompts ×16 generated tokens**; 45 post-prefill token steps per source, one Ubuntu22.04 GitHub CPU runner. This is NOT controlled throughput across hardware or coherence.

| Real model | Cached post-prefill greedy decode tokens/sec | Mean distinct-token ratio across 3×16 generated tokens |
| --- | ---: | ---: |
| Original FP32 teacher | **27.93** | **0.771** |
| Original packed CB59 | **52.51** | **0.104** |
| Zero-paid control | 51.10 | 0.104 |
| Bias-only | 51.42 | 0.104 |
| Rank4-only | **48.88** | **0.083** |
| Full paid rank4+bias | **55.95** | **0.083** |

**Crucial:** the compressed model produced **extremely repetitive greedy tokens** (~8% distinct IDs, vs 77% original teacher), despite high measured tiny post-prefill cached tokens/sec. This is **generation degeneration**, so faster cached decoding does NOT mean the model responds well or meets the desired quality–speed target. Prefill over the test windows was also slower than FP32 teacher (~5.7 seconds per domain packed vs ~0.96 seconds teacher, with small run variation); one-number tokens/sec does not demonstrate overall accelerated inference. Keep separate prefill/decode metrics and do not promote.

### Research decision
The v76 trainable rank4 residual is **real and functionally effective** for KL/NLL at paid <0.700bpw; vocabulary bias alone cannot explain that numerical improvement. **Yet the hypothesis of useful QA/generative function is not verified**: BoolQ correctness drops, SciQ prompt teacher below chance, outputs collapse/repeat, and direct CPU prefill still slow. Next v78 should target **global hidden-state error propagation and output entropy calibration with stronger TRAIN-only objectives** instead of further maximizing teacher KL alone, plus a corrected independent QA prompt with original-teacher sanity checks and exact cross-run reproducibility of trained factors, before attempting larger pretrained checkpoints. Treat compact storage, KL quality, QA accuracy and prefill/greedy throughput as separate metrics that all must pass.
 
Verified successful CI: https://github.com/Kamara229/JessicaCore/actions/runs/41543812501
First invalid-source-input CI: https://github.com/Kamara229/JessicaCore/actions/runs/41543185133
Research-only PR #207 remains DRAFT. Main, Jessica, Render, Supabase and APK unchanged.
