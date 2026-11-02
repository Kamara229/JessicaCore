# CALIBRI v67 — mathematical functional equivalence / gauge-canonicalization pilot
**2026-11-03; isolated research branch; no change to main, live Jessica, Render, Supabase, APK or secrets.**

## 0. Scientific status and the real starting point
The user requested a genuine interdisciplinary research program drawing on number representations, algebra, spectral geometry, operator theory, information theory and signal coding, not merely an additional quantizer.

Known VERIFIED previous results:
- v63 original trained 135M CB59 index entropy H=4.9749/5 and best lossless chunk net saving 59,971 B (0.534%). This route cannot alone fix catastrophic model error.
- v65 4-tensor trained-source binary factor geometry improves heldout **layer response** in 3/4 matrices, average ratio 0.89421; still no complete factor codec, native runtime, or language quality.
- v66 workflow run 41527576396 **SUCCEEDED** on pinned real 135M CB59 source. 17,084 physical 5-bit indices changed in 64 rows of four matrices, exact complete 11,229,303 B (0.667839bpw) remains. Selected-row WikiText VALID ratios vs original CB59: attention q 0.6583, MLP down **1.0758 REGRESSION**, gate 0.9612, up 0.6961; arithmetic mean 0.84784. This is NOT evidence of end-to-end model competence. Run: https://github.com/Kamara229/JessicaCore/actions/runs/41527576396

## 1. A new synthesis with an explicit mathematical invariant
**Hypothesis G67: before quantization, seek a functionally equivalent coordinate system (gauge) that is easier to encode.** This changes the representation of numbers but not the exact source function; we then quantize *both* paired factors at an identical physical bit cost.

For a paired linear value and output projection, in column convention:
    h = V x,  y = O h.
For any invertible D:
    V' = D V,    O' = O D^{-1},
hence O' V' x = O V x identically in exact arithmetic.

**First pilot uses D diagonal positive, not a dense rotation.** Balance the two corresponding factors analytically using:
    d_j = sqrt( ||O[:,j]||_2 / max(||V[j,:]||_2, eps) )
subject to an offline clamp. D is *absorbed* into V' and O' before encoding, so it need not be stored separately and adds no inference-time multiply. The compressed archive must still account for all new codebooks, indices, scales, headers and file integrity.

Why this might help: K32V8 source distortion is NOT invariant under paired channel rescaling when each stored matrix has independently trained centroids and row-wise FP16 scales; changing hidden-coordinate geometry changes which 8-tuples are easy to describe. In particular changing O columns is not generally canceled by per-row scales. This is an operator-level analogue of equalization/noise shaping, not FEC and not a claim to invent gauge symmetry.

This simple value/output operator omits softmax and grouped query attention; for real GQA, a value-coordinate transform must respect repeated key/value heads and the mapping into O. Never assume a global GL transformation is valid on Q/K with RoPE: the rotary commutant restricts it. First VERIFY exact teacher pre-quantization logits for every transformed real block before any quality claim.

## 2. Independent reproducible numerical falsification: SYNTHETIC, NOT real weights
File: `research/calibri/gauge_pilot_v67.py`. Numpy only. Seed=6701, two paired 576x576 matrices, initially artificially unbalanced latent channels; 128 independent normally distributed test input vectors. Real 5-bit-per-8 coefficient index *counts* with fitted 32x8 FP16 codebooks, FP16 per-row scales and 64B headers. All variants use identical 55,296 bytes for the two local tensors = 0.666667 **local** bpw (does NOT include a complete trained LM's other tensors / global archive).

Command: `OPENBLAS_NUM_THREADS=1 python research/calibri/gauge_pilot_v67.py 2.3`.

| artificial channel log-spread | Baseline response rel error | Balanced response rel error | Balanced/baseline | Interpretation |
| ---: | ---: | ---: | ---: | --- |
| 0.0 | 0.882693 | 0.882624 | 0.999922 | essentially no gain in already balanced source |
| 1.0 | 0.947056 | 0.882624 | 0.931966 | moderate gain in constructed imbalance |
| 2.3 | 1.228408 | 0.882624 | **0.718510** | 28.3% relative reduction in synthetic function error |

Additional synthetic diagnostic at spread 2.3: half balance rel error=0.959583; inverse-balance negative control=24.311117. Original-to-gauge uncompressed response discrepancy <=5.4e-7 relative in FP32; the exact identity is algebraic. These are from a container-executed NumPy pilot, not GitHub Actions and NOT pretrained weights. No inference or language-quality improvement has been shown. The baseline synthetic quantizer is itself poor (rel error >1), so don't overinterpret the ratio.

### Falsification and limitations
- **Novelty**: equalization and transformer gauge symmetry exist in prior research. The proposed combination with direct 0.7bpw K32V8 is a research hypothesis, not an established novel method.
- Some gauges can worsen codec quality dramatically (inverse-balance negative control). GQA/RoPE restrict valid symmetries. Balancing a tensor in isolation is NOT sufficient.
- Synthetic deliberately unbalanced factors; substantial gains may vanish on actual pretrained sources with already conditioned layers.
- Local physical 0.666667bpw cannot be reported as whole-model <=0.700bpw; the full model must be serialized, hashes/aliases/1D params counted.
- Model functional quality, speed, RAM, task generalization **unmeasured** for this technique.

## 3. Interdisciplinary mathematical program, not just one fixed algorithm
Each domain carries a *testable representation hypothesis*:
1. **Gauge group / Lie theory:** equivalent parameter orbits; verify exact equivalence under RoPE / GQA / gating restrictions before quantizing.
2. **Operator factorization / numerical linear algebra:** paired operator and tensor network decompositions, quantize composite function rather than individual weights.
3. **Information geometry:** use activation covariance/Fisher-like response sensitivity for error objective, with TRAIN/VALID separation.
4. **Spectral theory / random matrices:** measure stable effective ranks and singular-value tails before allocating binary latent rank.
5. **Frames / Kashin:** trade coefficient concentration against dictionary/index and extra-factor rate.
6. **Lattice geometry:** E8 or structured lattice coding when 8D source vectors support it at the actual rate.
7. **Rate-distortion / waterfilling:** allocate rate based on activation-weighted uncertainty rather than equal bits for every coefficient.
8. **Harmonic analysis / noise shaping:** Hadamard transforms, deterministic trellis and task-space residual feedback; time any added transforms.
9. **Structured sparsity / compressed sensing:** only where *measured* residual structure supports compact support encoding.
10. **Algebraic coding / FEC:** integrity and genuine erasure recovery only, never lossless restoration of deliberately dropped arbitrary trained weights.

## 4. Proposed real-weight stage (not executed here)
Use original pinned `HuggingFaceTB/SmolLM2-135M-Instruct` source, and CB59/CB66 original complete archives. Validate per-head V/O gauge invariance for its actual GQA mapping in FP32; test paired changes on 2–4 fixed pretrained layers. Only TRAIN data may set gauges, codebooks or selection. Compare same total physically serialized source rate, not extra offsets/hidden matrices. Use genuinely untouched VALID / task data to measure paired outputs and whole-model next-token KL, LM NLL/PPL and QA on sizeable independent sets. Evaluate channel allocation, gauge search cost and native compressed speed; keep a negative result if balanced real pretrained layers show no improvement. Do not propagate to live Jessica.

## 5. Primary scientific sources
- Gauge symmetries for canonical Transformer families and RoPE restriction (Wang & Wang, PMLR 2026): https://proceedings.mlr.press/v282/wang26a.html
- Nagel et al., Data-Free Quantization Through Weight Equalization and Bias Correction, ICCV 2019: https://openaccess.thecvf.com/content_ICCV_2019/html/Nagel_Data-Free_Quantization_Through_Weight_Equalization_and_Bias_Correction_ICCV_2019_paper.html
- Ordentlich & Polyanskiy, Optimal Quantization for Matrix Multiplication, IEEE TIT 2026: https://doi.org/10.1109/TIT.2025.3649596
- Lifar et al., WaterSIC, ICML 2026: https://proceedings.mlr.press/v306/lifar26a.html
- Merkulov et al., Kashin / overdetermined basis quantization, UAI 2024: https://proceedings.mlr.press/v244/merkulov24a.html
- Tseng et al., QuIP# / E8 lattice quantization, ICML 2024: https://proceedings.mlr.press/v235/tseng24a.html
- Chong et al., NanoQuant, ICML 2026: https://proceedings.mlr.press/v306/chong26a.html
- Lee & Kim, LittleBit-2, ICML 2026: https://proceedings.mlr.press/v306/lee26y.html

**Decision:** gauge-canonicalization survives a constructed fixed-budget synthetic probe, especially when original coordinate system is badly unbalanced. It has NOT passed the critical real-weight/whole-model evaluation gate. Next stage must test real GQA V/O transformations and discard the idea if quality or speed fails.
