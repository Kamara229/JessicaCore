# CALIBRI — session 23: residual re-centering at fixed size

Date: 2026-11-01. Research branch only, based on v22 PR #146. All v22 scores remain exploratory and should not be rebranded as a new hold-out benchmark.

## Problem and causal hypothesis

v22 changed only the FP16 tensor-train (TT) cores, while the original packed INT5 residual stayed fixed. On a REAL pretrained Pythia-14m 128×512 MLP matrix, teacher-train KL improved, but WikiText validation PPL worsened. That is a negative result. A plausible, untested explanation is mismatch between updated TT cores and a residual encoded around the *old* TT cores.

Make a controlled comparison, holding the teacher, 16-step learning procedure, token windows, tensor rank/shape and byte budget identical. Re-center the residual *after* TT training around the learned physical FP16 core:

- Original geometry: `W_0 ≈ TT(A_0,B_0) + Q5(W_0 - TT(A_0,B_0))`.
- v22 frozen-residual learned geometry: `W_fixed = TT(A_t,B_t) + Q5(W_0 - TT(A_0,B_0))`.
- v23 re-centered learned geometry: `W_recenter = TT(A_t,B_t) + Q5(W_0 - TT(A_t,B_t))`.

`Q5` is the existing signed row-scaled FP16, physical 5-bit packer. The post-training encoding can use original FP32 pretrained W0; inference must read the identical compact physical TT+INT5 format without a full dense residual buffer.

## Predeclared design and checks

- One unchanged real `EleutherAI/pythia-14m` projection (128×512).
- 1280 trainable TT-core scalars, TT geometry rank2 m2=16 n2=32.
- Teacher TRAIN = first 2048 WikiText-2 train tokens; two v22 methods: KL and KL + 0.1 local MSE; AdamW, 16 updates, learning rate 0.0005.
- TRAIN SHADOW = next 512 non-overlapping train tokens, used only to report teacher KL and local output error. They are not a new dataset.
- WikiText-2 validation 1024 tokens + TinyStories validation 2048 tokens: both previously inspected; **diagnostics only**.
- Compare original FP16, **plain physical INT4, INT5, INT6**, untrained TT+INT5, both fixed-residual trained TT variants and both re-centered trained TT variants. INT4/5/6 are restored dense **only for the quality-control calculation**, not claimed as direct packed compute.
- Check equality of no-op re-center, exact unchanged byte size (**43832 B**), CRC and pack/unpack, direct packed-C versus restored dense layer max-abs error <=0.001, layer reconstruction RMSE, full-model PPL and non-training shadow KL.
- Store only metrics; no pretrained model weights, residual streams, or personally identifying data in artifacts.

**Decision**: a size-neutral re-centering method is useful only if it improves diagnostic error or PPL consistently without worsening a second domain compared with the untrained physical geometry **and** has a defensible quality/size trade-off versus plain physical INT4/INT5/INT6. A TT+INT5 structure is **larger** than plain INT5 at this matrix size, so quality must justify the extra TT bytes. A negative result must be recorded honestly. Even a positive result is NOT proof of novel compression, robust generalization, fast ARM decoding, real MoE routing, or a QMoE 0.8-bit reproduction.

## Next session if results support it

Optimize TT cores together with the quantized residual under the physical byte budget, adding activation-aware quantization; then evaluate real SwitchTransformer routed experts with a truly independent corpus and native CPU/ARM throughput. Respect size/quality/latency jointly. Keep main, Render, Supabase, APK untouched.

## Observed real-run result (GitHub Actions 41342279169 — SUCCESS)

Real full-Pythia forward evaluations were completed and the C packed INT5 kernel matched the independently decoded layer within **3.58e-7** max-abs in the shadow check. No-op re-centering and physical 43,832-byte invariant passed. Original FP16 is measured with its 131,072 weight bytes; other components of Pythia remain uncompressed.

| Single 128×512 projection | Bytes | WikiText-2 dev PPL | TinyStories dev PPL | Train-shadow teacher KL |
|---|---:|---:|---:|---:|
| Original FP16 | 131,072 | 173.382 | 37.890 | ≈0 |
| Plain INT4 | 33,042 | 179.677 | 39.965 | 0.0760 |
| Plain INT5 | 41,234 | 175.705 | 38.639 | 0.0166 |
| Plain INT6 | 49,426 | **173.272** | 38.068 | **0.0040** |
| Untrained TT+INT5 | 43,832 | **173.482** | 37.785 | 0.0171 |
| Trained KL, frozen residual | 43,832 | 175.300 | 37.731 | 0.0138 |
| Trained KL, re-centered residual | 43,832 | 175.381 | **37.676** | 0.0188 |
| Trained KL+local, frozen residual | 43,832 | 175.160 | 37.696 | 0.0138 |
| Trained KL+local, re-centered residual | 43,832 | 174.831 | 37.719 | 0.0186 |

**Decision: mixed / no release gate.** Re-centering improves WikiText for KL+local against its frozen variant (175.160→174.831), but makes it worse for KL-only (175.300→175.381). TinyStories changes in the opposite direction for KL+local (37.696→37.719). Training the TT cores gives worse WikiText PPL than using the untouched TT geometry in every run above. Lower teacher KL on training examples does not establish generalized quality.

Physical storage is the major obstacle: TT+INT5 is **5.351 effective bits per original weight**, not 0.8, and exceeds *plain* INT5 bytes (41,234) by 2,598 bytes (~6.30%). The geometry can justify extra bytes only if it offers a better rate–distortion–runtime frontier; more aggressive graph sharing and a lower-entropy residual will be needed for the long-term 0.8-bpw target. Plain INT6 has better WikiText PPL on this sample, while TT+INT5 has better TinyStories PPL; neither is universally superior.

These are small, repeatedly inspected validation portions; no untouched independent test, ARM/NEON benchmark, or MoE result is proven. CI result: https://github.com/Kamara229/JessicaCore/actions/runs/41342279169 . Work is restricted to the research PR/branch.
