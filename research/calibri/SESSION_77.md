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
