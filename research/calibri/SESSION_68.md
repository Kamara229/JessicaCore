# CALIBRI v68 — real pretrained GQA V/O gauge at unchanged physical source rate

**Pre-registration before run. Research branch ONLY.** The synthetic v67 experiment did not show any real-model benefit. This stage tests that hypothesis on the actual pinned 135M trained dense source, not a constructed mathematical example.

## Objective
Real Llama grouped-query attention (SmolLM2 135M Instruct) has 9 Q heads and 3 KV heads of dimension 64. Attention probabilities depend on Q,K, not on V or output projection O. A per-KV-head diagonal positive gauge D may multiply each V coordinate and divide all corresponding THREE repeated O input channels by the same D coordinate. Exact FP32 attention output should be unchanged. The algebra has no new stored model parameters; D is absorbed into both transformed real source matrices prior to encoding.

Test whether this **function-preserving reparameterization** makes original CB59 K32V8 5-bit-per-8 compression more faithful on genuine attention outputs. Important: avoid falsely transforming Q/K with noncommuting RoPE matrices and avoid applying an ordinary 576-channel V/O pairing that ignores GQA repeats.

## Frozen experiment
- Pinned `HuggingFaceTB/SmolLM2-135M-Instruct` revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`.
- Exact pinned original CB59 physical source retrieved from successful v59 run `41507201766`, source SHA verified by original parser.
- Test source ONLY: WikiText-2 **TEST** tokens 1024–1151, two 64-token windows; no TEST activations used to select D or codebooks. Note: a single heldout split is not a blind claim of statistical significance.
- Two predetermined attention layers: 0 and 15, weights `v_proj.weight` and `o_proj.weight`. Analytical L2 source-only balance, clamp D range [1/8,8]; no validation tuning.
- Re-encode FOUR source matrix tensors by exact existing `direct_codebook_v59.compress_matrix`, including row FP16 scales, K32V8 FP16 codebooks, original integer-lattice code index quantizer. A recompression-of-unmodified-source identity control must reproduce CB59 bytes exactly.
- Cross-check measured manual GQA attention output with HuggingFace original `o_proj` pre-hook using ORIGINAL EAGER attention probabilities. Verify uncompressed transformed V/O gives same outputs as original teacher in FP32.
- Measure **actual local attention output** discrepancy after original CB59 paired matrices versus transformed same-rate CB59 matrices on the same frozen teacher attention maps, not just raw individual weight MSE.
- Physically patch new codes/centroids/scales into entire ORIGINAL CB59 135M container, recompute CRC/SHA, ensure unchanged global **11,229,303 bytes / 0.6678394131 bpw**, original complete unique parameter count and tied aliases. Do not count a hypothetical full archive, physically patch and reparse.
- Archive is NOT a native executable trained model quality result: no full-model logits/KL/QA, prefill/token speed benchmark or end-to-end accuracy is measured in this stage.

## Pass/negative criteria
The independent test first must PASS exact uncompressed GQA invariance, original CB59 same-source re-encoding, byte-preserving full-source pack and integrity. **Performance hypothesis** requires heldout paired layer attention response error ratio <1 to support further research. A ratio >=1 is a negative scientific result, not a CI failure; report it and abandon naive balancing if repeated.

## Context and prior measured state
v67 entirely SYNTHETIC V/O pair showed 0.7185 relative balanced-to-baseline quantized error when factors were deliberately unbalanced; this is not pretrained evidence. v66 real source guided index shaping had mean 0.84784 ratio on **selected rows only**, with a MLP down-layer regression 1.0758, so no general claim of whole-model quality. CB59 initial whole-model next-token quality was catastrophic even though archive rate is under cap. This v68 stage is genuine pretrained GQA-specific falsification, not a solution claim.

## Relevant research
Wang & Wang, gauge symmetries in transformers (PMLR 2026) https://proceedings.mlr.press/v282/wang26a.html ; Nagel et al. 2019 Data-Free Quantization Through Weight Equalization and Bias Correction https://openaccess.thecvf.com/content_ICCV_2019/html/Nagel_Data-Free_Quantization_Through_Weight_Equalization_and_Bias_Correction_ICCV_2019_paper.html ; Merkulov et al. Kashin basis https://proceedings.mlr.press/v244/merkulov24a.html .

## CI outcome
Pending at preregistration. No quality claims until run logs are verified. If positive: test more layer pairs, new disjoint data, then full native end-to-end KL, LM NLL/PPL, QA and speed. If negative: pivot to gauge search with task-conditioned but TRAIN-only objectives or constrained multidimensional operator coding; maintain budget honestly.

## VERIFIED real pretrained experiment — GitHub Actions 41530736611 (2026-11-02)
**SUCCESS** for reproducible math, trained source matching, original CB59 re-encode identity, real GQA 9Q/3KV exact invariance (<4e-7 relative), and physically rebuilt full original source container preserving exact 11,229,303 B / 0.667839413 bpw, tied aliases and checksums.

Real WikiText TEST tokens [1024,1152), source-only chosen scales, 2 × 64 token teacher-eager attention windows. Pretrained teacher-conditioned local attention layer L2 absolute error (lower is better):

| Real trained GQA layer | Unchanged CB59 error | Balanced re-encoded V/O error | Balanced / CB59 | Result |
| ---: | ---: | ---: | ---: | --- |
| 0 | 23.05442 | 23.89723 | **1.036558** | **NEGATIVE**, 3.66% worse |
| 15 | 349.72285 | 341.96419 | **0.977815** | positive, 2.22% lower |

**Verdict: MIXED / NOT GENERALIZABLE.** The large gain in synthetic deliberately unbalanced sources did **NOT** transfer uniformly to pretrained GQA; first layer regressed and another only slightly improved. Neither layer errors are end-to-end LM quality metrics. Do NOT replace all V/O pairs or claim competent language performance. We did NOT benchmark inference speed or whole-model KL/PPL/QA. The complete CB59 archive was patched in-memory and reparsed with exact byte count/integrity; the new blob itself was not uploaded as a checkpoint, only the audit JSON was stored.

### Decision / next falsifiable gate
The naive analytic norm-balance gauge alone is insufficient. Next investigate different mathematical degrees of freedom (per-head geometry, task-aware TRAIN-only gauge, block-diagonal/orthogonal isometries respecting GQA) ONLY if they can be absorbed into the source and recoded without uncounted side metadata and without degrading realistic compressed execution. Test a large set of PREDECLARED real layers on disjoint heldout TEST without selecting hyperparameters by TEST, and measure true whole-model output metrics before any performance claim. Prioritize quality over a large synthetic percentage.

CI: https://github.com/Kamara229/JessicaCore/actions/runs/41530736611
