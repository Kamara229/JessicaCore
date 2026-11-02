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

## Verified actual trained 135M experiment — GitHub Actions run 41539372478 (2026-11-02 UTC)

**CI PASS for genuine pretrained 135M source encoding, two separate TRAIN corpora, 12 actual native-packed complete model variants and two untouched independent TEST domains. Scientific result: NO candidate qualified under the frozen robust TRAIN rule.**

All twelve original-source CB59 full trained checkpoints physically reassembled at **11,229,303 bytes = 0.6678394131307638 bits per 134,515,008 unique parameters** each, including every trained 1D tensor, 5-bit code stream, FP16 centroids/scales, manifest, tied aliases, CRC/SHA. Identical native C direct-packed kernel, no original trained dense 2D source present in student execution.

### Measured TRAIN — genuine full native model teacher-KL
Each row mean teacher→packed next-token KL (nats; smaller is better). Training windows: WikiText TRAIN [8192,8448); AG News TRAIN records 5000..5099, first 256 model tokens, four 64-token contexts each.

| Fixed original-source candidate | WikiText TRAIN KL | AG News TRAIN KL |
|---|---:|---:|
| **Original CB59 baseline** | **9.72563** | **9.78943** |
| Layer0 V/O Hadamard | 10.03944 | 9.56855 |
| 8D covariance all 4 source matrices | 10.70344 | 11.29891 |
| Hadamard+all covariance | 9.85806 | 10.15111 |
| Covariance q only | 9.75404 | 9.73273 |
| Covariance down only | 10.63500 | 11.37519 |
| Covariance gate only | 9.68224 | 9.80502 |
| **Covariance up only** | **9.69764** | **9.76561** |
| Hadamard+q | 10.11019 | 9.52944 |
| Hadamard+down | 9.74939 | 10.13274 |
| Hadamard+gate | 9.97028 | 9.53515 |
| Hadamard+up | 10.07247 | 9.45402 |

**Frozen minimax rule:** maximum over candidates of minimum improvements across TWO TRAIN corpora. The raw best was `cov_up`: Wiki KL improved **−0.02799** nats and AG News KL **−0.02382** nats. Its *worst* corpus gain **+0.023819** nats is **below preregistered +0.05-nat required gain in both corpora**, so the chosen candidate is **UNCHANGED BASELINE**, not `cov_up`. This decision was physically saved BEFORE any TEST examples were opened. `cov_all`, previously chosen on Wiki-only TRAIN in v74, regresses badly in BOTH new TRAIN domains (KL 10.70344 and 11.29891). This is further evidence that naive compositions and narrow TRAIN objectives are unstable.

### New independent TEST — honest null selection
The frozen multi-domain TRAIN rule chose original baseline. Therefore TEST correctly compared **baseline to itself**, with no re-selection, no manufactured source improvement and no unjustified promotion.

| Real original native compressed model metric | WikiText TEST [60000,60512) | AG News TEST records 2000..2099 |
|---|---:|---:|
| Original CB59 teacher→student KL (nats) | **11.32578** | **10.29115** |
| Original CB59 token NLL (nats) | **14.92305** | **14.46329** |
| Frozen chosen source | baseline | baseline |
| Signed chosen-baseline KL / NLL | **0.0 / 0.0** | **0.0 / 0.0** |
| Native C packed direct full pretrained inference | PASS | PASS |

**Scientific result: NEGATIVE / null improvement.** The strict two-corpus TRAIN guard successfully **prevented unsupported promotion of a possibly distribution-specific transformed source**, but it did **not** produce a better model. These newly heldout TEST values are characteristics of unchanged CB59, NOT a test of raw runner-up `cov_up`; no inference about `cov_up` TEST performance is possible from this run. No absolute competence: native compressed model still has KL ~10–11 nats and NLL ~14–15 nats on genuine text. No measured speed parity, full QA transfer or next-token correctness recovery.

**Next stage v76:** rather than repeatedly test parameter-free representation changes while semantic quality remains catastrophic, prototype **TRAIN-conditioned low-dimensional multi-module residual corrections with explicit paid physical bits** (remaining complete-source budget to 0.700bpw), a directly executable sparse/low-rank sidecar whose model-level function objective optimizes teacher→packed KL. Require source-complete archive, entire on-disk correction/decoder bytes counted, no hidden dense weights or deployment, and genuinely disjoint tasks/corpora for independent quality/speed gating. Pre-register tradeoffs *before* opening additional TEST data. Compare against original CB59 and robust null baseline.

Verified research CI https://github.com/Kamara229/JessicaCore/actions/runs/41539372478 ; PR #205 draft isolated, main/Render/Supabase/APK/live Jessica untouched.
