# CALIBRI S93 — repair-first protected full/EOS training (research only)

**Hypothesis:** S92 failed because its starting S91 physical candidate ALREADY forgot 25 S87 full answers; the 32-step optimizer allocated half of its primary gradients to still-wrong novel pairs, leaving insufficient power to recover those 25 while protecting 257 S91-correct outputs.

## Frozen evidence
- S87: 196/456 strict full/EOS TRAIN.
- S91: 257/456, 86 gains over S87, 25 losses, one first-token loss; all-inclusive 0.684717 bpp, CPU speed ratio 1.00858.
- Exactly 282 individual TRAIN identities have strictly correct full/EOS from **either** frozen S87 **or** S91. This is a protection *union*, not a single-model score.
- S92 MPS GPU succeeds; its physical mini-guard with 105–107 rows still showed 20–22 union full regressions, no novel successes beyond union, and one first-token loss. Independent full 456 CPU S92 check was underway when S93 was authored. Do not misrepresent the preliminary mini-guard as final 456 result.

## Intervention isolated to S93
1. Start from real S91 physical compressed source and neural Green Tunnel (NOT S92 model) to avoid inheriting unknown later effects.
2. Train first **72/96** real MPS joint steps on the 25 identified S91-forgotten S87 answers, with first-token repair at every fourth step; only last 24 steps mix two fresh TRAIN contrast pairs with two repair rows.
3. Rotate **six S91 previously FULL-correct protections + one first-only protection per step**, with a first-order halfspace projection. Preserve both FP16 core codebooks and the trainable internal rank-8 Green Tunnel; do not train or replace the frozen controllers, output head or zero-gate path.
4. Serialize FP16 physical codebooks + GT at 24/48/72/96. Mini-guard all selected 257 S91 full successes, 25 forgotten S87 successes, first-token regressions and 12 still-wrong paired TRAIN targets, measuring exact EOS.
5. An independent CPU guard on *all* 456 frozen TRAIN identities requires **all 282 protected full outputs AND all protected first-token outputs** plus at least one new full/EOS answer, bpp<=0.7 and CPU>=90% S82. Reject if any criterion fails; never deploy, merge, or claim independent generalization.
6. Never read existing sealed QA/TEST, never interact with LIVE Jessica/Render/Supabase/APK. Jessica relay messages and HTTP429 are not quality tests.

If strict union remains infeasible, research evidence will motivate architecture-level recovery rather than stronger TRAIN memorization.
