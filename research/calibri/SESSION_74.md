# CALIBRI v74 — frozen global TRAIN-KL optimization of physically composited multidimensional neural weights

**2026-11-03. Preregistered before CI TEST. Research only; no production Jessica / Render / Supabase / APK / main changes.**

## What recent genuine measurements demand
v73 retained all real trained source parameters at **0.667839413 bpw**, and 8D activation-covariance source-only index reassignments reduced all four tested original TRAIN layer response errors. Yet untouched WikiText TEST **whole packed model** teacher→student KL **10.6538→12.1336** and heldout NLL **14.1629→15.4635**, significant regression. Thus isolated original-teacher local layer error is a poor proxy for cascading losses inside an *already compressed* transformer. The earlier pairwise Hadamard v70 alternative slightly improved heldout KL in 2 domains, but v72 SciQ/BoolQ actual answers showed no compelling quality gain.

## New v74 research hypothesis, defined in a falsifiable way
**Do not optimize only one teacher-conditioned matrix.** Select among **combinations of already measured physically stored multivariate 2D/8D source fragments using the actual ORIGINAL pretrained teacher→ENTIRE native packed-model TRAIN KL**. The objective is global output behavior of compressed model, not artificial source distance.

This is explicitly **discrete candidate-level, TRAIN-only joint KL selection**, NOT end-to-end continuous differentiable full-model weight training, NOT a new mathematically original compression theorem. Its virtue is that each candidate is a TRUE complete independently executable packed 135M source, without any hidden floating-point correction or additional inference-time operation.

### Candidate pool, frozen *before* TEST
Source checkpoint original pinned `HuggingFaceTB/SmolLM2-135M-Instruct`, revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`, **134,515,008 unique trained parameters**. Real existing original CB59 archived source:
- `baseline`: every tensor original K32×8, 5-bit per original 8D codebook index, FP16 centroids/scales.
- `hadamard`: exact v70 2D function-preserving paired V/O Hadamard transformation on only GQA layer0.
- `cov_all`: exact v73 TRAIN-only eight-dimensional covariance index reassignments in four predefined pretrained tensors (layer0 q, layer0 MLP down, layer15 MLP gate, layer29 MLP up).
- `hadamard_cov_all`: independent byte-level composition of the two above, modifying different source tensors.
- `cov_q`, `cov_down`, `cov_gate`, `cov_up`: isolated v73 modifications only.
- `hadamard_cov_q`, `hadamard_cov_down`, `hadamard_cov_gate`, `hadamard_cov_up`: isolated covariance changes added to v70 Hadamard.

**12 fixed candidates**. Every one must literally reconstruct as a full original CB59-format 11,229,303-byte archive: unchanged trained tensor names/shapes, 1D weights, alias tables, 5bit native CPU packed operator ABI, CRC32/SHA256. All 5-bit codes/FP16 centroids/scales are physically in that file. No *paid* bits, secret sidecars, expansion to original trained dense hot runtime. Physically exact original/gauge/all-cov controls must byte-match their prior source fragments.

### TRAIN-only whole-model selection
Use WikiText-2 `train` token indices **4096:4352**, 4 windows ×64, 252 next-token positions. This window is distinct from v73 local TRAIN 0..255 and v70 earlier TRAIN 2048:2304. Original FP32 teacher in separate process; 12 native C direct-5bit student complete LLM processes. Score each on full-vocabulary KL against original teacher, *not* local error or TEST labels. All logits/hidden operations from original packed model and genuine trained source. Teacher logits temporarily stored FP16 for memory; disclose numerical approximation. Select lowest TRAIN whole-model KL if it beats original baseline by at least **0.05 nats**; otherwise select unchanged original baseline. Frozen selection written to JSON **before** any official TEST data opened. No special preference for previous Hadamard, and all covariance-candidate failures reported.

### New TEST data and metrics
- WikiText-2 original official `test` tokens **50000:50512** (distinct from 40000-40256 v73 and 32768-34816 v71).
- AG News official `test[1000:1100]` *text only*, first **512 tokenizer tokens**, distinct from v71 first 300 examples.
- Each set eight 64-token windows = 504 scored next-token positions; 1008 combined.
- Original FP32 teacher separately, original CB59 and **only frozen TRAIN winner** as separate teacher-free native C full 135M executions. Report KL teacher→packed, observed next-token NLL, teacher top1 match, paired eight-window bootstrap 95% CI, actual warm CPU 8x64 prefill duration. Record BOTH directions and CI even if significance weak. If TRAIN picked baseline, TEST remains baseline; report zero difference honestly. All TEST remains immutable; NO tuning on TEST.

### Decision criteria
- CI PASS means complete source integrity, actual direct packed full-model execution and valid genuine scores, not that the candidate succeeded.
- Scientific success requires lower TEST KL and heldout NLL in **both** domains and uncertainty evidence; tiny 504-position domains cannot establish semantic quality, language model competence or speed parity.
- Genuine trained-language absolute quality at ~0.7bpw has remained very poor: structural/source selection alone does not solve this. If the original global TRAIN objective still overfits or selects baseline, move on to actual paid global sparse corrections / mixed binary low-rank sources within <=0.700 physical bpw with a new native decoder, and evaluate multiple genuine task benchmarks.
- No ordinary-chat Jessica test requested or needed. This is an isolated research-draft PR.

## Verified CI outcome
Pending at preregistration. Record literal signed global TRAIN and TEST results only after independent workflow completes.

## VERIFIED real CI findings — 2026-11-02 UTC, run 41537733285

**INFRASTRUCTURE PASS, scientific full-model cross-domain generalization FAIL.** The completed Ubuntu22.04 GitHub Actions workflow built and integrity-checked all 12 *genuinely source-complete* trained 135M native-5bit CB59 candidate checkpoints. Every actual physical source measures EXACT **11,229,303 bytes = 0.6678394131307638 bits/unique pretrained parameter** including all 1D weights, FP16 centroids/scales, manifest, tied aliases and checksums. The original native C direct-packed full model executed teacher-free; no additional runtime ABI or original trained dense 2D source weights loaded.

### Pre-committed TRAIN-only whole-model selection: genuine trained source
4×64 WikiText TRAIN source contexts, indices [4096,4352), mean FP32 teacher→native packed full-model KL:

| Original-source candidate | Real full-model TRAIN KL (nats) |
|---|---:|
| Original CB59 baseline | 10.61790 |
| 2D V/O Hadamard only | 10.61441 |
| **8D covariance edits in all four tensors** | **10.06250** |
| Hadamard + all four 8D | 10.72880 |
| 8D q only | 10.63858 |
| 8D down only | 10.20577 |
| 8D gate only | 10.59860 |
| 8D up only | 10.50220 |
| Hadamard+8D q | 10.63430 |
| Hadamard+8D down | 10.94949 |
| Hadamard+8D gate | 10.64182 |
| Hadamard+8D up | 10.62045 |

The frozen minimum `cov_all` improves genuine whole packed-model TRAIN KL by **−0.55540 nats**, above the predeclared 0.05-nat threshold, and was selected and stored **before opening either TEST dataset**. TRAIN NLL original 14.21246 vs selected 13.65977 (−0.55269 nats). This is *not* proof of unseen performance.

### Independently evaluated two genuine TEST domains, frozen before TRAIN selection
Each official heldout test domain has **8 ×64-token contexts (504 next-token predictions)**; both physical checkpoint files and CPU compressed runtime unchanged from TRAIN.

| Actual TEST corpus metric | Original packed CB59 | Frozen `cov_all` 8D | Change |
|---|---:|---:|---:|
| WikiText-2 TEST tokens [50000,50512), KL teacher→packed (nats) | 10.66333 | **10.49060** | **−0.17274** (smaller is better) |
| WikiText-2 TEST NLL | 14.14146 | **13.85621** | **−0.28525** (improved) |
| AG News TEST records [1000,1100), KL teacher→packed (nats) | **9.66904** | 10.59238 | **+0.92334** (worse) |
| AG News TEST NLL | **13.69314** | 14.65448 | **+0.96135** (worse) |

**DECISION: NEGATIVE for robust quality.** Whole-model TRAIN-KL optimization mitigated the isolated-teacher-output proxy problem for the *training distribution* but chose a source variant that does NOT transfer to independent AG News. The news regression (+0.923 KL, +0.961 NLL) is large relative to the WikiText improvement; this is a mixed result, not robust compression quality success. The original baseline and 8D candidate both remain far below original trained language-model usefulness (teacher-relative KL ~9.7–10.7 nats; candidate absolute heldout NLL ~13.9–14.7). Native CPU time was logged but is NOT shown as a speed improvement: source code and C operator are unchanged, no low-latency performance claim.

The naive GQA pairwise Hadamard composition is not additive with all covariance edits: TRAIN KL `hadamard_cov_all` (10.72880) was worse than each individually, a real **negative interaction** among two techniques supposedly helpful in isolation. This supports measuring full-model interactions rather than presuming methods can be stacked.

Scientific limitations: small frozen data slices; v73 covariance component was originally TRAIN-fitted using WikiText early tokens; optimizing with WikiText TRAIN further risks distribution-specific specialization; AG News records are distinct from v71's first 300 records but still English short news; teacher logits temporarily FP16 in metric; no label-based QA or free-form correctness, no second trained model, no multi-host inference-speed control.

### Proposed next gate v75
Do **NOT** promote `cov_all` into Jessica or assert ultra-low-bit model competence. Pre-register a genuinely **cross-domain TRAIN** optimization objective with a *worst-domain/generalization penalty*, and compare against baseline and earlier Hadamard under untouched independent per-domain TEST sets. Consider input distribution robustness, final full-model teacher→student KL, per-layer residual propagation, and an actual bit-allocation/stronger 8D or binary factor codec only with fully counted source metadata and native decoder speed. A quality improvement must exceed candidate variance on unseen QA and generation, not only conditional-language KL. Preserve all unsuccessful variants in the research record.

Verified GitHub Actions run: https://github.com/Kamara229/JessicaCore/actions/runs/41537733285. Draft research PR #204 remains isolated. Live Jessica, Render, Supabase, APK and main were not modified.
