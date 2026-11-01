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
- Compare original FP16, untrained TT+INT5, both fixed-residual trained TT variants and both re-centered trained TT variants.
- Check equality of no-op re-center, exact unchanged byte size (**43832 B**), CRC and pack/unpack, direct packed-C versus restored dense layer max-abs error <=0.001, layer reconstruction RMSE, full-model PPL and non-training shadow KL.
- Store only metrics; no pretrained model weights, residual streams, or personally identifying data in artifacts.

**Decision**: a size-neutral re-centering method is useful only if it improves held-out diagnostic error or PPL consistently without worsening a second domain compared with the untrained physical geometry. A negative result must be recorded honestly. Even a positive result is NOT proof of novel compression, robust generalization, fast ARM decoding, real MoE routing, or a QMoE 0.8-bit reproduction.

## Next session if results support it

Optimize TT cores together with the quantized residual under the physical byte budget, adding activation-aware quantization; then evaluate real SwitchTransformer routed experts with a truly independent corpus and native CPU/ARM throughput. Respect size/quality/latency jointly. Keep main, Render, Supabase, APK untouched.
