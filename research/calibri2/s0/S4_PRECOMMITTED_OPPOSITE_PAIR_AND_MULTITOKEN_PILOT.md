# CALIBRI 2.0 S4 — precommitted TRAIN-only pilot (22M, no architecture growth)

Date: 2026-11-09. Research-only proposal; **no training authorized by this document**. Branch: `research/calibri2-s0-small-first-paid-architecture`; draft PR #376. Preserve 135M v001, main, LIVE Jessica, Render, Supabase and APK.

## Verified starting evidence

S3 run 42530436760, TRAIN artifact 13638046399; independent byte/token audit in `S3_INDEPENDENT_ARTIFACT_TRAIN_AUDIT_2026-11-09.md`. Physical packed CB59 1,932,195 bytes / 22,368,320 parameters = 0.6910469807298895 bpp, SHA256 `b7f6450e29acfa53699314392853efae1029bfc158d30d3d62a06767905d8720`. At step 1920: 125/176 exact full + EOS, 60/88 opposite pairs both correct; step 2560: 122/176, 58/88. Step 2560 multi-token phrases 1/32, sentences 1/16. No proven generalization, FP32 parity, or comparable native CPU >=90%.

## Hypothesis and controls

The S3 curriculum learns short associations but fails to retain both contradictory answers and multi-token sequences. A **single** bounded 22M pilot should test whether paired replay plus whole-sequence EOS supervision can improve these TRAIN behaviors without silently sacrificing other TRAIN domains. This is not a license to optimize a public held-out score.

- Keep exact 320x6, 22,368,320 parameter architecture, tokenizer, packed CODEBOOK+5bit physical format and frozen starting checkpoint. Never import pretrained FP32 model weights. Use only already documented TRAIN pairs; **do not read or train on DEV/VALID/QA/TEST**, and do not select hyperparameters, checkpoint, or stopping time from them.
- Fix the sample schedule *before* running: construct deterministic mini-epochs over all 88 TRAIN pairs, with left and right examples adjacent but their order swapped by a seeded permutation each mini-epoch. Include each pair side equally; retain all existing domains, not just phrases/sentences. Avoid synthetic examples that reproduce any held-out prompt.
- Teacher-forced loss must include **every answer token and EOS**; log per-token position losses and exact generated sequences, not only first-token accuracy. Report premature EOS, omitted EOS, wrong first token, and repetition separately. For contrasting pairs, track `both/left_only/right_only/neither`; do not average away failures.
- Set the training budget and fixed checkpoints in the workflow before dispatch. Suggested conservative cap: one 640-step pilot with snapshots at steps 0, 160, 320, 480, 640; no adaptive extension, no automatic retry unless an infrastructure/code failure. This is a proposed design, not an observed result. Use at most two new CI attempts in one check.
- Run the existing offline `audit_train_answers.py` on the actual saved TRAIN JSON and archive its full row output. Audit all 176 rows at every snapshot, ensure both sides of every pair, strict full-token equality including EOS, distinct first-token counts, repetition flags and category breakdowns. Preserve the original S3 artifact and compare like-for-like.
- Independently hash the **extracted** packed binary and record exact file bytes and `8*bytes/22368320`; fail the pilot if physical bpp > 0.700 or if the artifact is not CODEBOOK+5bit. Native C inference must execute the exact saved packed checkpoint; separately measure comparable CPU throughput against the same-baseline implementation with controlled hardware, thread count, prompt length, warmup and token count. Report ratio; do not claim >=90% until measured.
- Report final fixed-step TRAIN behavior and all snapshots; no best-checkpoint cherry-picking. A favorable aggregate TRAIN count is insufficient: require evidence of sustained multi-token full+EOS generation and no systematic opposite-pair collapse. If this fails, remain at 22M and revise mechanism rather than increase model size.

## Scope / decision gate

Implement only under `research/calibri2/s0/` and its dedicated workflow on the research branch. No main merges, no Jessica test execution, no changes to production systems. Do not expand to 134,515,008 or 360M until the current 22M generates useful correct answers robustly and the independent physical/speed gates are met. All figures above are TRAIN diagnostics, not proof of held-out quality.
