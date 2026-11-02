# CALIBRI v78 — physical reproducibility, frozen TRAIN-only correction calibration and real answer quality

**2026-11-03: preregistration before v78 TEST.** This study changes only isolated branch `research/calibri-qa-repro-v78` and draft GitHub PR. It does **not** alter Jessica, main, Render, Supabase, APK, secrets, or original pretrained checkpoint.

## Why v78 is necessary
Actual v77 paid rank4 + bias source **11,725,487 bytes (0.697348923 bits per original unique trained parameter)** improves mean teacher→packed full-model KL and heldout NLL substantially on two new corpora at exactly paid physical source rate. Factor-only contributes most of the improvement vs bias-only. But **SciQ original trained teacher performed 7/47 under a one-letter answer template, below 25% chance**, whereas packed paid factor was 17/47, showing the prompt was unreliable as a proxy for intelligence; **BoolQ 26/64 baseline fell to 22/64 paid**, so no demonstrated answer gain. Greedy generation v77 full only ~8.3% distinct generated token IDs vs teacher ~77.1% despite ~56 cached tokens/s on one CPU. Further, v76 and v77 repeated identical training settings but produced different trained FP16 source SHA files (very similar TRAIN VALID KL), so exact cross-run weight reproducibility is **not** established.

## First gate: reproduce genuine v76 original fully paid source, twice
- Use the unmodified real pinned pretrained `HuggingFaceTB/SmolLM2-135M-Instruct`, original revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`, and exact physical real original CB59 source artifact from GitHub Actions v59 run 41507201766. All 134,515,008 original unique pretrained source parameters, original FP16 codebooks/scales, tied aliases, 5bit codes and CRC/SHA physically preserved.
- Repeat the **unchanged v76 TRAIN-only rank4/bias optimizer** with the same already validated real WikiText and AG News TRAIN FIT/VALID source logits / *real compressed-model hidden states* **twice in independent Python training invocations on the SAME GitHub CPU host**. Before running tests, save first and second source file SHA256 and real FP16 factor max absolute differences U/V/bias. Record explicitly whether bytes are identical; **do not declare a reproducibility pass merely because both runs pick rank4 step20**. Compare both SHAs against v76/v77 prior runs and report genuine cross-host discrepancy. A difference within one CPU runner is a real determinism failure even if KL is close.
- Freeze the original first TRAIN-selected CB76 source before any TEST. Both trials must reproduce v76 selected `rank4` at optimizer step20, full original source and correct physical single-file versioned CRC/SHA structure. If not, stop rather than reselect using TEST.
- Keep both original complete source rates: CB59 **11,229,303 B = 0.667839413 bpw**; paid CB76 **11,725,487 B = 0.697348923 bpw**. These include ALL original pretrained trained weights and paid FP16 rank4/bias. The model still uses C direct 5bit source GEMV, **paid correction executed on CPU Torch**, not all-native-C and not a deployed fast assistant.

## Frozen scalar calibration: learnable output-weight geometry without new stored bits
The original v76 rank4 corrected head `y = CB59(W,x)+U(VᵀRMS(x))+b` is too repetitive in greedy generation. Explore **fixed source-absorbed scalar factors** `α∈{0,0.5,0.75,1.0}`: replace FP16 stored U with `αU`, FP16 bias with `αb`, keep V identical. ALL four candidate files remain byte-identical in length at **11,725,487 B**, no additional scalar metadata or updated pretrained source, no new inference operation. Zero `α` is a physically paid null control.
- Freeze list and threshold now. Before opening any TEST, score the actual original pretrained teacher vs modified source *full-vocabulary KL* using native packed-model hidden activations on TWO independent **TRAIN VALID** corpora from v76, WikiText TRAIN VALID and AG News TRAIN VALID. Rank candidates by **worst-domain full-KL gain** against original original CB59. Require **≥0.10 nats in both TRAIN VALID domains** or choose α=0.0. This tests whether reducing correction intensity can improve generalizability; it is **not direct training for task QA nor generator entropy**. Explicitly avoid calling it a solved anti-repetition algorithm.
- Serialize the chosen α in the already-paid FP16 U/bias tensors and freeze final CB76 SHA before touching any new TEST examples. Do **not** modify scalar α in response to TEST accuracies or generation diversity. Original uncalibrated α=1 frozen v76 model is evaluated as a fixed control.

## New truly unseen TEST language and QA, unchanged after source lock
- WikiText2 official `test` original tokenizer tokens **90000:90512**, 8×64 windows / 504 next-token positions.
- AG News official `test[5000:5100]` text only, first **512 model tokens**, 8×64 windows / 504 next-token positions.
- New **SciQ TEST original rows 840–863** (up to 24, each with context `support` truncated to first 320 tokenizer tokens). This addresses previous one-letter prompt failure by scoring **all tokens of each actual answer string** using teacher-forced, length-normalized mean token log probability, instead of demanding a single letter. Four different answer contents are scored for each question; duplicate answers or empty/inordinately long options uniformly excluded **before evaluation**, source row IDs logged. This is a custom *support-assisted* answer selection diagnostic, NOT an official SciQ leaderboard or free-form language understanding test. The original pretrained FP32 teacher must be scored to check that the protocol itself is meaningful.
- New **BoolQ VALID original rows 1200–1227** (28 cases): use the original passage **first 256 tokenizer tokens** instead of previous first128, all question tokens, single Yes/No conditional next-token decision. Chance 50%. Still truncated, not official BoolQ benchmark.
- Each model evaluated using exactly identical prompt token hashes, source dataset revision SHA and fixed original answer-label identities. Count original teacher correct answers, original complete CB59, unmodified complete v76-paid correction, and new TRAIN-scaled complete CB76. Record paired accuracy differences, exact two-sided McNemar p and paired bootstrap 95% CIs; small sample cannot establish competence. Do not select α on these QA data.
- Also genuine greedy **KV-cached** autoregressive **3 fixed prompts ×24 tokens** per source. Keep first-token prefill separate from subsequent decode tokens/sec; record mean distinct-token ratio, max repeated-token share, exact generated token hashes. Compare original trained FP32, CB59, unscaled paid and TRAIN-scaled paid, without claiming coherence just from diversity. The benchmark deliberately continues after EOS for fixed workload; not a conversational response quality measure.
- Measure genuine full-model teacher→packed KL, NLL, native compressed model prefill CPU times on both untouched text domains and actual complete file rate; report sign of every regression. The original trained dense 2D source remains **unmaterialized** in each compressed student runtime.

## Decision gate
Science passes only when numerical output, physical integrity, TRAIN→TEST separation and reproducibility auditing are valid, regardless of objective outcome. If both paid models remain repetitive or QA is not improved with credible teacher anchor, conclude that hidden-output adjustment alone is insufficient. Further stages must optimize model internals and *task-conditioned error propagation* under strict complete ≤0.700 bpw storage and performant native decoding, with genuinely new neural architectures/weights, not just output bias. A KL improvement is **not** evidence that the compressed model can answer questions well.

## CI results
Pending actual workflow. A failed test must be fixed transparently with original TEST/data protocol changes disclosed. Record verified results, including negative ones, only after run completes.

## VERIFIED actual pretrained experiment — GitHub Actions 41547079249 (2026-11-02 UTC)

**GitHub Actions SUCCESS** for actual source integrity, full genuine trained 135M direct-native packed inference with paid rank4 correction, repeated independent TRAIN optimizer invocations and untouched TEST language, QA and autoregressive generation.

### Weights and reproducibility — important qualified progress
- Original full source CB59: **11,229,303 physical bytes / 0.667839413 bpw**. Paid CB76: **11,725,487 real bytes / 0.697348923 bpw**, including all trained source coefficients, original metadata/aliases, correction FP16 U/V/bias, header, CRC and SHA. Original pretrained trained dense 2D source matrices are NOT loaded in the compressed runtime; original 5-bit tensor GEMV executes via C native decoder, small rank4 paid correction via CPU Torch (not all-native C). No production modifications.
- **TWO independent v76 training repetitions on the SAME CPU runner exactly matched byte-for-byte:** both complete CB76 SHA256 `bf5eed8a06fb5d01843fca1bd23d62066ed04813403e71f9cba0ed80ebd1ee62`, max absolute FP16 U/V/bias difference **0.0 for all three arrays**. **Also exactly matches the original v76 PR #206 source SHA**. BUT v77's different CI execution previously produced SHA `4d5c1258924b77a458b1d08d962914f6c8845632f8604a0a04e929cd43ef759c`; thus **cross-run / cross-host reproducibility remains UNPROVEN** and SHA divergence is explicitly retained. Do not say trained weights are reproducible on all environments.
- Frozen two-TRAIN-VALID scalar test `α={0,0.5,0.75,1}` (absorbed in U,b FP16, identical physical bytes), selection minimum improvement ≥0.10 nats across both TRAIN corpora. TRAIN VALID full-model teacher-KL for α0: Wiki 12.17645, News 12.26084; α0.5: Wiki 8.39623, News 9.38721; α0.75: Wiki 7.71471, News 8.86114; **α1: Wiki 7.52793, News 8.65434 (best both)**. The frozen source selected **α=1.0**, i.e. **unchanged original v76 paid rank4+bias weights**, NOT a new weight optimization improvement from scaling. The so-called calibrated source therefore has the SAME SHA, same logits and QA as full v76 by design; do not imply independent new performance from its duplicate TEST execution.

### New disjoint TEST language (eight 64-token contexts per domain, each 504 scored next-token positions)

| True teacher-vs-full packed model metric | Original CB59 | Frozen v76 paid rank4+bias (α=1) | Signed change |
|---|---:|---:|---:|
| WikiText-2 TEST tokens 90000–90511, mean KL (nats) | 10.81293 | **7.94323** | **−2.86971** |
| WikiText-2 TEST next-token NLL (nats) | 14.95724 | **11.94340** | **−3.01384** |
| AG News TEST records 5000–5099, mean KL (nats) | 10.64970 | **8.15847** | **−2.49123** |
| AG News TEST next-token NLL (nats) | 14.74117 | **12.08575** | **−2.65542** |

Paired eight-window 95% bootstrap TEST KL signed intervals for α1 vs baseline Wiki **[−3.7750, −1.7880]**, News **[−3.5580, −1.5044]**; correlation and small sample caveats apply. **Absolute KL/NLL remain very poor**; the new TEST only confirms numerical correction at 0.697349bpw, not useful language or global robust generalization.

### ACTUAL answer correctness with protocol sanity: SciQ answer *strings* (not letters), original support

New original SciQ TEST rows 840–863, n=24 valid, original `support` text limited to first320 tokens; each of 4 entire original answer strings scored by teacher-forced mean log probability. This is a custom, supported closed-choice metric. Unlike v77's unsupported one-letter prompt that made even original FP32 teacher score below chance, **the actual original pretrained teacher now scores 20/24 = 83.3%** (above 25% chance), so the new protocol has a meaningful teacher anchor.

| Heldout QA | Original trained FP32 teacher | Original physically compressed CB59 | Paid original CB76 rank4+bias (α=1) |
|---|---:|---:|---:|
| **SciQ support-assisted full answer content, n=24** | **20/24 = 83.3%** | **6/24 = 25.0%** | **7/24 = 29.2%** |
| **BoolQ first256 passage tokens, n=28** | **14/28 = 50.0%** | **16/28 = 57.1%** | **10/28 = 35.7%** |

Paired CB76 minus CB59 SciQ correctness **+1/24**, exact McNemar p=**1.0**, 95% paired bootstrap signed accuracy-delta **[−0.125,0.2083]** (includes zero). Paid BoolQ **−6/28**, McNemar p≈**0.2632**, paired bootstrap accuracy-delta **[−0.5357,0.1071]** (includes zero). Original teacher was itself only at chance on this fixed and truncated BoolQ slice. **Science outcome: NEGATIVE — the newly teacher-validated SciQ prompt decisively reveals the compressed model's absence of original factual competence despite much better KL than the original CB59; one extra correct SciQ is not evidence of recovered answers.** BoolQ direction worsens, not statistically established. Both tasks are small, constrained answer-choice diagnostics rather than official leaderboard task evaluations.

### TRUE greedy cache generation and speed (one GitHub Ubuntu22 CPU)
Three fixed prompts ×24 greedy output tokens, 69 measured native-model cache-decode tokens after each first-step prefill.

| Actual complete model | Post-prefill KV-cache tokens/s | Distinct generated token ratio | Mean largest-token frequency within output |
|---|---:|---:|---:|
| Original FP32 trained teacher | **34.81** | **0.7917** | 0.1250 |
| Original CB59 | 33.86 | 0.0694 | 0.7500 |
| Paid rank4+bias CB76 | 32.99 | 0.0972 | **0.9444** |
| Frozen α1 duplicate paid source | 32.62 | 0.0972 | **0.9444** |

**Critical failure:** CB76 greedy generation is *almost entirely the same repeated token* (most common emitted token ≈94.4% of 24-token output), even more collapsed than original CB59 by top-token share. This is NOT solved by lower KL; cached token throughput (~33 tok/s) is not evidence of usable or faster ChatGPT-like assistance, FP32 teacher is ≈34.8 tok/s in this single CPU run. True compact runtime speed/quality target is not met.

### Decision: v79 research direction, not deployment
1. **Stop equating KL/NLL gain with recovered answer competence.** A teacher-sane SciQ prompt scores 83.3% correct for original FP32 and only 29.2% for CB76; this is concrete semantic failure.
2. Train next source representation against *genuine answer-conditioned TRAIN functional objectives*, intermediate activation propagation and teacher next-token top1/sequence behavior, not merely teacher probability KL at the output head. Explicitly address one-token greedy collapse and try control over logit repetition/entropy without hiding original model degradation; use paid bit allocation across internal 2D/8D tensor representations and direct-runtime CPU throughput. Keep whole source including correction ≤0.700bpw.
3. Diagnose **cross-GitHub-run floating-point determinism** with CPU math runtime/seed/thread/torch version/environment capture and compare exact original TRAIN inputs and source SHA; v78 proves same-host perfect repeat and match to v76, but v77 SHA differs. Store pinned full source SHA for a truly bitwise portable model if future archive release is explicitly authorized, not only JSON.
4. Test the learned source in a much larger / different pretrained model and more robust labeled QA/greedy generation before claiming novelty, speed or practical deployment. Continue in isolated research branch. No Jessica ordinary-chat test required.

Verifiable completed workflow: https://github.com/Kamara229/JessicaCore/actions/runs/41547079249
Original source+protocol: draft PR #208.
