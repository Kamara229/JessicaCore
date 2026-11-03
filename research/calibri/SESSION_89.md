# CALIBRI v89 — genuine independent untouched original TEST for frozen physical 4D neuro-weights

**Preregister before reading any new official TEST data.** Research-only stacked on #218, never deployed to working Jessica/Render/Supabase/APK/main.

## Motivation from real v88, not hypothetical
Original pretrained SmolLM2-135M-Instruct CB88 physically full source **11,413,683 bytes / 0.67880503 bpw**, SHA **d0f547e651c9a3b411fa7b19c0181839ab813c69e9092c77dca2a47d8e5d85ec**. Genuine original TRAIN-VALID v88 Actions #41586136382 SUCCESS: WikiText KL baseline CB59 11.37736, CB85 11.33837, CB88 **8.66044**; AG News 12.91852, 12.86311, **10.80833**. But 2-prompt real greedy distinct ratio CB88 **0.125** (CB59 0.125, CB85 0.1875). This was a **positive TRAIN-VALID KL gate and negative generation-diversity result**, NOT recovered general QA or unseen TEST.

## Frozen independent scientifically honest comparisons
Original pretrained FP32 teacher, untouched native-C original CB59, prior frozen input-adaptive CB85, newly frozen 4D tensor+native CB88. All full-model, same official unseen originals, no source fit/update from TEST. Literal checksums on all three source archives must match previous REAL run, before any evaluation.

New official SciQ **test[560:608]** 48 examples, BoolQ **validation[3210:3258]** 48 examples; independently heldout whole-answer scoring, exact paired McNemar/5k paired bootstrap. WikiText official **test tokens[185000:186024]**, 16×64 contexts = 1008 original next-token positions. AG News official **test rows[6900:7020]**, consume first 1024 original model tokens, text only with no labels. All disjoint from v86 and previous uses. Four never-used greedy prompts (24 tokens each) with real KV cache, record diversity, top-token concentration and CPU decode tokens/s, plus language CPU prefill.

## Predeclared research-quality gates (every gate required)
- Full source physical <=0.7 bits / original pretrained unique parameter (includes original native packed source, all tensor factors, bias, CRC/SHA).
- CB88 original heldout SciQ and BoolQ accuracy **no lower than BOTH CB59 and CB85** independently.
- On EACH corpus CB88 KL strictly improves versus BOTH frozen compressed baselines, NLL no worse than BOTH; report 5000 paired-window bootstrap confidence intervals. Small sample sizes mean no claims of significance if bootstrap crosses zero.
- Actual four-prompt greedy distinct-token ratio >=0.25 (prevent obvious collapse; no semantic quality guarantee).
- Exploratory single-CPU real KV-cache decode tokens/s CB88 >=90% original CB59; do not claim robust CPU speed parity from a single run.

Source/model CPU runtime distinguishes original native 5-bit C from new additional Torch tensor contraction, not fused. A green GitHub workflow **is only execution success**. If any scientific quality gate fails, report negative and do not promote/deploy the model, irrespective of KL or compression improvements. No training on TEST; follow-up next only with NEW TRAIN + VALID/TEST, not this already-exposed test.

## Facades
\`frozen_4d_independent_v89.py\` thin CLI → \`v89\` public facade → \`v89.pipeline\` orchestrator → \`v89.provenance\` frozen source/test ranges and \`v89.stats\` paired pure statistical checks. Four pure regressions and isolated GitHub Actions CPU workflow. No changes to live Jessica. 
