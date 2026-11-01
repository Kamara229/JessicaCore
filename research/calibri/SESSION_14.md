# CALIBRI — session 14: quality gate after negative trained-model result

v13 CI on a real pretrained Pythia-14m MLP projection (128 × 512) completed:
- Original PPL 247.544 (WikiText-2 **test** 1,016 scored tokens, context 128).
- Standalone signed INT4 physical 33,042 B: test PPL 251.600.
- Standalone signed INT8 physical 65,810 B: test PPL 247.480.
- Train-activation selected **rank8 → INT4 residual + 1% FP16 sparse**
  physical 47,212 B: test PPL 269.261.
- Calibration module-relative squared error: INT4 0.0186151; stacked
  0.0133717. **Improved local calibration error but worsened test PPL.**

This is an important negative result. Minimizing local module squared error
is *not* a valid standalone proxy for preserving end-to-end language quality.
The stacking method was also larger than INT4, and its execution speed was
not measured. It must not be described as a successful compressed model.

v14 retains the original untouched v13 code and adds a **separate development
quality gate**:
1. Real training activations construct all 10 physical variants.
2. Independent WikiText-2 **validation** split evaluates whole-model PPL
   for every candidate (after substituting the same single projection).
3. Within a fixed <=1.5× INT4-byte budget, approve only candidates whose
   development PPL is <=1.01× FP32-original development PPL.
4. From approved variants pick the **smallest physical byte count** (PPL
   tie break). If none pass, report NO_APPROVED_VARIANT and use original
   FP32/FP16-style weights instead of silently violating quality constraints.
5. Evaluate on independent untouched WikiText-2 **test** only the predeclared
   controls and selection. Report development activation MSE separately.
6. No choice or hyperparameter adjustments based on test PPL.

Key scientific limitation: this is one 65,536-parameter projection in a
14M-parameter dense model, not a MoE, not QMoE reproduction, and there is no
real model bundle, latency benchmark, combined multiple-module fidelity,
shared dictionary or trained lowrank optimization. A pass under 1% PPL is
not a guarantee of user-facing general-purpose quality.

v14 goal: falsifiable control for whether a **global quality constraint**
avoids the v13 failure. It is not yet a demonstration of CALIBRI's advantage
over GPTQ/AWQ/SLIM/TileQ/RS-MoE.

No main merge, APK, Render, Supabase, or user model changes.
