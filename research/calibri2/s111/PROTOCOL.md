# CALIBRI 2.0 S111 — prompt-conditioned shared Green Tunnel

**Status:** research-only proof of architecture. This does not constitute improved generative ability. Never merge to `main` or change LIVE Jessica, Render, Supabase, or APK without explicit approval. Jessica chat tests are not authorized.

## Evidence motivating this architecture

- S101 actual physical compressed cached baseline: 353/456 full+EOS authorized TRAIN correct, 387/456 first tokens correct, bpp 0.699573397602.
- S103 / S104: a **single shared GT3 donor** with ten global scalar gains preserved 353/456; two genuine MPS steps caused no gains.
- S105: in 8 wrong × 12 protected TRAIN gradient pairs, 46/96 directions opposed. The strict local gradient projection from the old task to all 12 protected constraints returned a zero direction. This is **not** a theorem that gradient learning cannot improve CALIBRI.
- S106 actual alpha-scaling test (Linux CPU #42878106022): at alpha=1 no gained/lost; alpha=2 **one newly correct and one previously correct forgotten**; alpha=4 +1/-4; alpha=8 +2/-5, with first-token forgetting at 8. All unsafe variants rejected. This is direct evidence of potential new responses and old-response interference.
- S107: four sparse layer-mask variants produced **zero newly correct**; top-3 4× forgot one old full response.
- S108 / S109: two frozen shared GT3+GT9 blocks, 20 global gains, real MPS two steps; 353 correct full answers remain, none gained.
- S110 original S101 genuine answer diagnosis #42879592400: of 103 wrong answers, **69 start wrong at token 0; 18 diverge at token 1; 16 have incorrect final EOS**. Most errors are first-token. 156/228 counterfactual owner pairs both correct; no leading-whitespace false success.

## S111 hypothesis

Global gains apply identically to all prompts, causing gradient interference across questions. Rather than another model of separate per-layer weights, use the **already paid S95 prompt-only signed-hash vector** (512D) to condition the intensity/sign of two existing frozen GT3/GT9 donor residuals at the ten otherwise unconnected transformer boundaries.

Each 512D S95 numeric feature vector is already computed independently of the correct answer. A **fixed, non-trainable** 16×32 group-sum + L2 normalization compresses it to sixteen features, and one bias is prepended. Train only a 2×10×17 coefficient tensor. The entire model adds **340 trainable FP32 parameters in RAM**, but exactly **680 raw FP16 weight bytes on disk**, plus safetensors header/config bytes. All original S101 source and S95 routing weights stay frozen.

This is a generic deterministic transformation of prompt tokens, **not** a lookup table keyed by question IDs and **not** an output selector that accesses a target answer. Frozen S95 chooses S93/S87/S91; S111 is active **only** for S93, with S87/S91 unchanged.

## Failure gates

Stage A reconstructs original physical S101 model (S93 compressed CB61, S94 physical three-profile bank, S95 router, S98 copy, S101 KV cached CPU). On the exact 456 pre-existing TRAIN prompts it checks (a) 456/456 exact full/EOS greedy token equality to unhooked S101 when conditional coefficients are all-zero; (b) 353 full + 387 first correct; (c) real serialized FP16 sidecar + required runtime config yield all-in bpp ≤0.700000; (d) same-host CPU speed ratio ≥0.90 vs unhooked S101.

Synthetic unit tests also verify independent prompt-dependent effects, both donor gradient paths, no extra donor weights in serialized sidecar, S95 alternate profile isolation, and prompt scope teardown.

Even Stage A success does not show task improvement. Stage B S112 may train the new coefficients ONLY after real Stage A passes, with the same immutable 353/387/456 full-answer and physical limits. Previously inspected QA never used, not even for tuning hyperparameters. Any future independent performance evaluation must use a new, once-only, predeclared holdout.
