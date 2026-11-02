# CALIBRI v75 — two-domain TRAIN worst-case KL selection, pinned real full-model source

**Predeclared before any v75 TEST measurement.** Isolated draft research PR only. No changes to main, Jessica's live chat, Render, Supabase, APK, weights settings, deployment credentials, or user data. No Jessica chat test.

## Why change v74
v74 built **12 physically complete trained 135M CB59 checkpoints**, each **11,229,303 bytes = 0.667839413 bpw** with original native compressed C inference. Selection solely by WikiText TRAIN full-model teacher KL favored `cov_all`, but heldout WikiText TEST KL improved −0.17274 and AG News TEST KL **regressed +0.92334** (NLL regressed +0.96135). This is a real negative generalization finding. The v72 constrained-answer result also failed to establish practical model competence.

## Frozen candidate sources and physical budget
Use exact original pinned `HuggingFaceTB/SmolLM2-135M-Instruct` revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`, 134,515,008 unique original parameters, original checksum-correct full CB59 source from CI run 41507201766. The source-only v70 2D Hadamard transform and previously predeclared v73 8D activation metric on 4 real tensors are rebuilt with unchanged scripts. Compose the same twelve v74 variants: `baseline`, `hadamard`, `cov_all`, `hadamard_cov_all`, `cov_q`, `cov_down`, `cov_gate`, `cov_up`, `hadamard_cov_q`, `hadamard_cov_down`, `hadamard_cov_gate`, `hadamard_cov_up`.

Each physically serialized *entire original-source checkpoint* includes all 1D tensors, 5-bit per 8D-group indices, FP16 codebooks and row scales, full tensor manifest, tied aliases and CRC/SHA. **No metadata side bits or native kernel changes**; a separate process loads ONLY compressed trained 2D tensors and uses unchanged native C LUT-GEMV. `baseline`, `hadamard`, `cov_all` must byte-match original source archives exactly.

## New robust TRAIN-only model-level objective
Two disjoint official TRAIN data sources, both frozen before running:
- WikiText2 `train` tokenizer tokens **[8192,8448)**, four windows of 64, 252 next-token scores. Distinct from v73's source-fit first 256 tokens, v74's Wiki TRAIN [4096,4352) and v70 Wiki TRAIN [2048,2304).
- AG News original `train[5000:5100]` *text only*, first **256 tokenizer tokens**, four 64-token windows, 252 next-token scores. No classification labels used.

For each corpus: original full pretrained FP32 teacher separately; all twelve complete real packed CB59 models in independent teacher-free native C processes. Original trained source model revision and dataset revisions recorded. **No TEST data read in this phase.** Score whole-model original teacher→candidate full-vocabulary next-token KL and actual TRAIN token NLL.

Define per-corpus TRAIN gain for each candidate `g_c = KL_baseline,c − KL_candidate,c`. Choose the candidate that **maximizes min(g_WikiTrain,g_NewsTrain)**, with fixed tie-breaking by the frozen list. Promote that candidate only if **both corpus gains ≥0.05 nats**; otherwise choose baseline and treat that as a legitimate negative outcome. This is discrete per-source **minimax selection**, not differentiable full-parameter training or an original theorem. Set threshold and candidate pool BEFORE any TEST.

## Untouched independent TEST
Only after saving frozen winner JSON, evaluate original baseline and selected candidate (plus FP32 teacher in separate process) on:
- WikiText2 official TEST tokenizer tokens **[60000,60512)** = 8×64 windows / 504 next-token positions, never used in prior CALIBRI experiments.
- AG News official `test[2000:2100]` text only, first **512 tokenizer tokens** = 8×64 windows / 504 next-token positions, disjoint from v71/v74 news-test records.

Report signed paired teacher KL, next-token NLL, teacher top-1 token agreement, paired 8-window fixed-seed bootstrap percentile interval, physical source bytes/bpw, and actual exploratory native CPU prefills. No choice, threshold adjustment or source re-encoding after TEST. Only a directional **both-domain KL and NLL gain** counts as a small exploratory improvement; any bootstrap CI crossing zero is explicitly inconclusive. A baseline selected on TRAIN produces exactly 0 paired deltas, not an invented positive result.

## Scientific limits and next decisions
Absolute original packed KL is ~10 nats and accuracy is near chance on many answer tasks. The <0.7bpw storage/native decoder gate alone is NOT practically useful; broad language competence and inference speed parity are not proven. This experiment only asks whether multi-source TRAIN guards prevent obvious v74 out-of-domain regressions; 252 TRAIN / 504 TEST next-token points per domain are small and correlated. A success would still require a genuinely new source format with paid global error correction and evaluation on *different* pretrained models, larger heldout text, QA, and native generation speed. A failure means selection among existing source geometries is insufficient, motivating more fundamental learnable compact operators and error-aware bit allocation.

## CI outcome
Pending prior to real workflow measurements. Append measured signed results and negative findings after CI.
