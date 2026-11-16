# S5 — controlled FP32 layer-restoration sensitivity (DIAGNOSTIC, no packaged FP32)

Real previous results: S2 original FP32 vs original 0.66784-bpp CB59 hidden cosine 0.68→0.04 across 30 layers and neither teacher input embedding nor teacher output head nor both alone restore coherent generation. S3 48 paid full-211+GT rank24/32 training gave **0/8** first-token agreement and **0/8** full answer agreement despite valid 0.69656 bpp. S4 stronger same-rate source fitting reduced original FP32 weight SSE 1.67% yet still **0/4** first-token agreement and worse hidden cosine.

**Goal:** rigorously determine if loss is concentrated in earliest, latest, attention or MLP blocks before investing further in new mixed-rate codec. S5 is deliberately **UNPAID FP32 hybrid diagnostic only**, and no hybrid can be called a valid 0.7-bpp release.

1. Begin original pinned 134,515,008 trained FP32 SmolLM2-135M-Instruct and original 211-matrix CB59 source SHA `50eac24ba7d7a452c495f8adced2cdef3f9cc44e024217fc21d8f2cebb87413c`.
2. Use only three previously exposed S0/S1 TRAIN-debug questions: 2+2, opposite hot, and sunrise. No new EVAL/QA.
3. Patch the CB59 student's **input embedding and output LM head** to their **same original FP32 teacher values** in all configurations (already proven insufficient in S2), then temporarily restore:
    - first 5/10/15/20/25/30 whole decoder layers from trained FP32;
    - last 5/10/15/20/25/30 whole decoder layers;
    - attention sublayers in **all 30** (MLP still CB59);
    - MLP sublayers in **all 30** (attention still CB59).
   Include a zero-restored-layer embedding+head comparator and all-30 exact-model positive control. Restore all original student modules in `finally` after each intervention. These hybrids are **not exported weights**.
4. Score first-next-token FP32 exact match, FP32 next-token rank in the hybrid logits, and 8-token full output/EOS on the original FP32, direct CB59, selected control hybrids. Log each original prompted raw answer and every intervention result; NEVER turn teacher FP32 into an inference dependency.
5. Log how much FP32 storage would be needed if hybrid matrices were packed naively; the physical 0.7 bpp budget is NOT met by these experiments. From the causal impact infer where next rate-aware quantizer should allocate its scarce bits; do NOT claim model recovery.
6. Research only, no NRA/S3 language adapter, no main/LIVE/Jessica tests/Render/Supabase/APK changes, no changes to CALIBRI2 43M research.
