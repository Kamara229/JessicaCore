# CALIBRI v65 — Task-aware same-rate latent geometry after a mixed v64 result

**Preregistered study; status on initial commit: not yet run.** No main, live Jessica, Render, Supabase, APK, or weights refresh is allowed.

## Trigger and evidence

After original CB59 showed lossless index entropy near maximal (v63 H=4.9749 of 5 bits/index, 59,971 B actual saved), we evaluated latent binary matrix factors informed by LittleBit and NanoQuant on four original trained SmolLM2-135M-Instruct matrices (v64, GitHub Actions 41525887887). Real independent WikiText TRAIN/VALID teacher activations showed mixed errors:
- attention q layer0: 0.4661 (CB59) vs 0.3198 (binary) on VALID, **ratio 0.686**;
- MLP down layer0: 0.6257 vs 0.8018 (ratio 1.2815);
- MLP gate layer15: 0.6624 vs 0.6960 (ratio 1.0507);
- MLP up layer29: 0.7978 vs 0.7175 (ratio 0.8993).
Mean ratio 0.9794; more work needed. Crucially v64 capped the rank at r≤192 although true per-matrix CB59 bytes permitted r≈256 for all three MLPs. No whole-model loss/QA claim.

## Single preregistered hypothesis

**H65:** For a fixed single-tensor **source-equivalent physical storage budget**, increasing binary latent rank to the maximum allowed (r≤256) AND source-only iterative orthogonal latent alignment may lower heldout layer-output distortion compared with v64, and may beat CB59 on additional layers. Test all six candidate variants at the same rank and compare under heldout teacher activations. Distinguish rate saturation from geometrical alignment in the JSON ablation. Reuse the identical model source revision, dataset family, two disjoint 128-token TRAIN/VALID partitions and four original modules (early attention, early MLP down, middle MLP gate, late MLP up). This is explicitly a **local matrix study** only.

The exact candidate source is sign(U sqrt(S) R), sign(V sqrt(S) R), learned **FP16 per-factor gains**, and the original CB59 **FP16 row scales**. Both sign matrices are bit-packed (r*(m+n) real bits plus padding); count two actual arrays, gains, scales and 64-byte tensor header. Rank is chosen solely from the actual original CB59 per-matrix bytes, before inspecting validation. Candidate byte count must not exceed CB59 local packed codes+FP16 rows+FP16 dictionary.

R is either identity, a fixed-seed random orthogonal matrix, or the source-only ITERATIVE Procrustes rotation: after eight alternations, assign signs to stacked balanced singular factors [U sqrt(S); V sqrt(S)], solve the closest orthogonal rotation using SVD, repeat. This is a simplified ITQ-inspired source-only geometric preconditioner, not a claim to reproduce published LittleBit-2 QAT or NanoQuant ADMM.

For every rotation, fit gains by source-weight LS or real WikiText TRAIN activation-output LS. Choose the per-tensor candidate solely by TRAIN output RMSE. Then report all 6 heldout VALID errors, the TRAIN-chosen heldout error, ratio to original CB59, rank, real source bytes, and source-only sign alignment objective. No TEST labels, no external learned correction sidecar and no cheating by selecting the best validation model.

## What constitutes a result?

- Hard smoke and no-data tests: packbits matching measured source cost; 8 alternating Procrustes steps yield finite orthogonal matrices and non-worse source sign-quantization objective.
- REAL source physical CB59 SHA+CRC: original pretrained unique 134,515,008 trained parameters and genuine weight tensor shapes.
- All candidate factor local archives <= original CB59 matrix payload. **No full-model 0.700bpw claim**, since a full factorized archive, tied embedding, 1D tensors, dictionaries and runtime module descriptors are not yet serialized.
- Heldout validation error below CB59 on most or all four matrices supports deeper exploration. A non-improvement means geometric alignment and rank reallocation are insufficient *without* more sophisticated iterative factor/response training.
- Runtime is not benchmarked. No complete functional packed model exists at v65. Prior complete CB59/CB61 archive remains poor in actual language output. A low layer-output RMSE is not a language competence measure.

## Follow-on if the pilot provides a credible improvement

First develop a global physical bit allocator and source-to-factor mixed-format experiment, including a *single* new <=0.700bpw full archive (all tensors, tied weights and CRC) and direct binary source kernel. Then train/validate on disjoint 2048-token windows and downstream QA, compare against v61 and exact teacher. If v65 remains mixed, use analytic source covariance/task-loss error shaping C01/C17 and true iterative binary factor adaptation rather than stage another near-uniform index entropy coding experiment.

References: LittleBit (NeurIPS 2025) https://proceedings.nips.cc/paper_files/paper/2025/hash/a917c1d57088897beba47f96b4fd7f3c-Abstract-Conference.html ; LittleBit-2 (ICML 2026) https://proceedings.mlr.press/v306/lee26y.html ; NanoQuant (ICML 2026) https://proceedings.mlr.press/v306/chong26a.html ; CALIBRI v64 record research/calibri/SESSION_64.md .


## VERIFIED RESULT — run 41526555083, 2026-11-02

The real source artifacts, real pinned pretrained weights and independent teacher TRAIN/VALID activations all passed; 4/4 tensor comparisons completed in GitHub Actions, source-only Procrustes toy mathematical check passed. This is a **measured local tensor improvement**, not a functioning whole-model language result.

| Tensor | CB59 VALID response rel-RMSE | v65 TRAIN-selected factor VALID rel-RMSE | v65/CB59 | Packed factor bytes / original CB59 tensor bytes |
| --- | ---: | ---: | ---: | ---: |
| layer0 attention q (r176) | 0.4661 | **0.2853** | **0.6120** | 26,912 / 27,584 |
| layer0 MLP down (r256) | **0.6257** | 0.7219 | 1.1538 | 69,312 / 70,784 |
| layer15 MLP gate (r256) | 0.6624 | **0.6383** | **0.9636** | 71,232 / 72,704 |
| layer29 MLP up (r256) | 0.7978 | **0.6761** | **0.8475** | 71,232 / 72,704 |

**Mean of four VALID error ratios = 0.89421; 3 of 4 tensors improved.** All four TRAIN-selected candidates used source-only Procrustes ITQ with TRAIN activation LS fitting. The 8-iteration ITQ source sign-error objective decreased for each matrix. Going from v64 to saturated v65 reduced the mean error ratio from 0.9794 to 0.8942 (different candidate choices, same original tensors/validation family). This does not prove statistical significance, downstream language competence, or reproducibility across seeds. Importantly, the early MLP down tensor remains **worse than CB59 by 15.38%**, so a uniform substitution of all matrices would be unwarranted.

**Decision:** advance to a **mixed-codec, task-aware allocator**, with CB59 retained in matrices where factorization loses. Test C17/C19 source-code noise shaping on the remaining weak matrices. A complete <=0.700bpw physical model must count all 2D matrices, tied embeddings/head, 1D trained parameters, metadata and native compact operators before proceeding to independent whole-model NLL/QA and speed. The local improvement is a promising direction, **not** a solved 0.7bpw language model.
