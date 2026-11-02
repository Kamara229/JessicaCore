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
