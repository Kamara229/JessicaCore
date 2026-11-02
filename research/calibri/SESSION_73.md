# CALIBRI v73 — TRAIN-only 8D activation Mahalanobis source-code assignment

**Status: preregistered before CI. Research branch only.** No main, live Jessica, Render, Supabase, APK, secret or deployment changes. Original pretrained 135M model and original physical CB59 source pinned by exact full-model archive from v59 GitHub Actions 41507201766, model revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`.

## Scientific trigger / limits of earlier work
v70 selected a GQA-safe pairwise 2D Hadamard gauge by independent TRAIN only and v71 observed small mean KL/NLL gains on WikiText and AG News, but uncertainty intervals crossed zero. v72 showed **no demonstrated answer-quality gain**: SciQ teacher/baseline/gauge all 8/40, BoolQ FP32 teacher 29/40 vs original packed 20/40 and gauged 21/40, exact paired McNemar p=1; both compressed CPU markedly slower. A metric gain on weights or language KL is NOT usable language capability.

## Falsifiable alternative: 8D task-induced metric without any side bits
Stop concentrating solely on arbitrary source coordinate gauges. Use **activation-weighted code-index choices** while freezing every original K32×8 FP16 codebook, FP16 row scale, format, 5-bit index stream length, alias, 1D weight and native C direct kernel.

For original group of 8 source weights `w_g`, centroid `c_k`, row scale `s_r`, and *actual pretrained teacher TRAIN* activation block `X_g`:
```
C_g = X_g^T X_g / T                    (8x8 real Gram; only TRAIN)
D_diag(k) = (w_g/s_r - c_k)^T diag(C_g) (w_g/s_r - c_k)
D_full(k) = (w_g/s_r - c_k)^T [0.75 C_g + 0.25 diag(C_g)] (w_g/s_r - c_k)
```
For each 8D group, assign the best of EXISTING 32 original indices with `argmin D`. This is a **local 8-dimensional covariance metric** and does NOT yet minimize cross-group error correlations. Native decoder still sees exactly the original 5-bit indices and scales, so no source metadata increase and no new runtime multiply. **This is analogous to known Hessian/activation-aware quantization, NOT a new proven theorem or a reproduction of full GPTQ.**

Frozen original pretrained matrix selection: layer0 attention q, layer0 MLP down, layer15 MLP gate, layer29 MLP up. Exactly **256 uniformly spaced output rows per tensor**, source-independent. Each tensor has three fixed candidates: unchanged CB59, diagonal Gram nearest-centroid, and full 8D shrunk Gram nearest-centroid. No other codebook learning, clipping, rank fitting or post-hoc parameters.

Use genuine **256 WikiText TRAIN** tokens (original pretrained teacher full forward, hook inputs only). Choose the per-tensor candidate that reduces the teacher TRAIN *full row response* relative error by **>0.5% vs unchanged CB59**, otherwise keep baseline codes. Selection is independent of all WikiText TEST data. The exact original 5bit packed stream lengths are unchanged; check all 134,515,008 unique pretrained trained parameters, original alias map, original manifest, physical CRC/SHA, no extras. No extra runtime ABI. Explicitly record if all modules choose unchanged; a zero-gain result is scientifically valid.

## Independent whole-model quality and time test
After the source is frozen and physically written, run original FP32 teacher, original CB59 direct native C full packed model and new v73 complete CB59 8D-selected full packed model as **distinct processes**. All tested on WikiText **TEST tokens 40000..40255**, four 64-token windows (252 next-token scored positions). Original pretrained teacher weights NEVER loaded in compressed student processes. Measure real teacher-to-packed KL nats, heldout next-token NLL, FP32 teacher top1 agreement, normalized logits error, exploratory CPU full prefill times and exact physical bpw.

This TEST set is frozen before build, never accessed to choose matrix candidate, codebook/index or hyperparameters. Absolute original model language quality, cross-domain generalization and task QA are **not** established by 252 positions. Even a positive result must be replicated on new longer text, new model, and independent task labels before any deployment assertion. If TEST NLL/KL regresses, retain the negative finding and move to joint cross-group 8D/16D operator factorization or learned functional correction under a genuinely full physical budget.

### Hard acceptance gates
- Actual complete source length **11,229,303 bytes / 0.667839413 bpw** for both original and candidate (including all checksum, manifest, aliases, scales, codebooks and original 1D tensors).
- Real model revision, source integrity and reparse checked; all 4 source matrix dimensions verified. No dense original pretrained 2D weights in either student runtime.
- CI passes only when source/index/format integrity and real TEST score calculations pass; quality improvement **NOT** required, negative results must be recorded.
- Stage does **not** measure user-facing Jessica; no text test requested.

## References
Ordentlich & Polyanskiy, *Optimal Quantization for Matrix Multiplication*; GPTQ (Frantar et al. 2022); HAWQ-V2 (Dong et al. 2020); original CALIBRI v66 signal-guided code-index reassignments. The new engineering test is fixed-rate, original-source-index-compatible and uses a true 8D TRAIN-derived metric; novelty is unproven.

## CI result
Pending at preregistration. Log actual metrics and any failures here after verification.

## VERIFIED REAL numerical outcome — GitHub Actions run 41536784680 (2026-11-02 UTC)

**Integrity PASS; full language-quality HYPOTHESIS FALSIFIED.** Exact original full source plus new trained-activation 8D quadratic-code candidate: **both 11,229,303 physical B = 0.6678394131307638 bpw**, trained 134,515,008 unique params, all original 5-bit K32×8 centroids/FP16 scales/1D tensors and aliases preserved, no additional side bits; original C direct-5-bit GEMV full-model kernel and trained-2D-free student execution verified.

**Actual pretrained TRAIN selection** on WikiText first 256 TRAIN tokens, 256 frozen source output rows per matrix: all four selected full 8D activation covariance (not diagonal or original) by the predeclared 0.5% layer-output threshold. Original vs TRAIN-selected **real teacher layer response** normalized errors:
- Layer 0 q: 0.47002 → **0.40317**, 9,782 modified 5bit source symbols.
- Layer 0 MLP down: 0.62452 → **0.53965**, 22,153 modified 5bit symbols.
- Layer 15 MLP gate: 0.67197 → **0.51084**, 5,486 modified 5bit symbols.
- Layer 29 MLP up: 0.82268 → **0.46033**, 6,925 modified 5bit symbols.
- Total: **44,346 modified EXISTING five-bit indices**, **zero additional source bytes**, no new executable format.

**Independent entire native pretrained LLM evaluation (TEST tokens [40000,40256), four 64-token windows, 252 next-token positions):**

| Whole-model metric | Original physical CB59 | TRAIN-selected 8D covariance codec | Result |
|---|---:|---:|---|
| FP32 teacher→packed mean KL (nats; lower better) | **10.6538** | **12.1336** | **NEGATIVE: +1.4798 nats** |
| Real next-token NLL (nats; lower better) | **14.1629** | **15.4635** | **NEGATIVE: +1.3006 nats** |
| Native C full 256-token context prefill CPU time | **5.359 s** | 5.366 s | Slightly worse; not accelerated |
| Physically complete stored original pretrained model | 11,229,303 B | 11,229,303 B | Same exact size |

**Interpretation:** Local source TRAIN activation-covariance 8D assignments yield substantial reductions in module output error on the teacher-forced TRAIN activations, but **harm actual auto-regressive full-model logits/NLL on independent TEST**. This is an important negative result—TRAIN layer-input response error is NOT a reliable proxy for the strong error propagation across quantized layers. Quantization mismatch may alter internal activation distributions and later-layer interactions; do not assert causality without further ablation. No general QA improvement or speed improvement. Moreover, both original and candidate output distributions remain extremely poor relative to original teacher. Do **NOT** promote v73 covariance-coded source to Jesssica or claim the sub0.7 problem solved.

**Next decision:** stop adding isolated layer-output metrics as if they were model competence. Next session should explicitly model the **joint** quantization errors across modules using true packed-model trajectories, with a global TRAIN-only output KL objective, cautious source edits and comprehensive TEST/QA/CPU measurement. Better candidate class: small low-rank binary/2D block factor + task-aware residuals under a fully serialized global bit allocator. A new method should earn its complexity in direct packed inference at ≤0.700bpw, not only improve teacher-forced local error. Keep prior negative results visible.

Verified CI: https://github.com/Kamara229/JessicaCore/actions/runs/41536784680 ; draft research PR #203, production untouched.
