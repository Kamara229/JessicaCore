# CALIBRI 2.0 S34 — frozen experiment acceptance contract

Scope: isolated research branch only. No Jessica chat tests and no production/LIVE/Render/Supabase/APK edits.

## Verified input
S33 GitHub Actions #42484161593: 128 proposed steps, 30 accepted, 98 reverted; strict TRAIN new 0/16→4/16, old 6/8→6/8. S33's latest accepted count was 30 at steps 96 and 128. No inference about why further candidates failed without inspecting per-candidate loss and exact old-ID guard results.

## Precommitted S34 procedure
1. Restore exactly the S26 36M CB59 source and verify SHA plus paid/dense logits. This bpp applies ONLY to the original compact source.
2. Reproduce S33 by rerunning all 128 steps from the original source with identical optimizer and LR decisions; HARD STOP unless accepted=30, new=4/16, old=6/8. Preserve replay diagnostics including late rejected candidates.
3. Attempt exactly 80 more dense FP32 TRAIN updates with the committed color/noun/EOS loss and four old TRAIN anchors. Fixed LR sequence: 2e-5, 8e-6, 3e-6. Accept only primary TRAIN improvement and exact preservation of the six initially correct old full-answer+EOS identities; otherwise revert both weights and AdamW moments.
4. Preserve every attempted candidate and actual full greedy TRAIN answer. Save final FP32 weights only if the entire job succeeds. This FP32 artifact is not a paid CB59 output and does not establish <=0.7 bpp.
5. Never read S29 sealed unseen QA or previously inspected S26 QA for training, checkpoint choice, or early stopping. Do not convert the TRAIN-only result to a generalization claim.

## Interpretation of outcomes
- If S33 replay fails, report the negative discrepancy and do not run S34 training.
- If S34 accepts zero updates or new strict TRAIN answers do not improve, report the negative result unchanged. A better forced-gold noun rank alone is not a correct free-generation answer.
- If S34 improves TRAIN quality, the next independent step is a **physically paid compact-weight/Green Tunnel experiment** with native CPU performance and heldout assessment, not a parity declaration from FP32 data.

No changes to the original research checkpoints or to any production services.
