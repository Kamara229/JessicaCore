# CALIBRI S125 — transfer audit: S87-J7 → S124 (not a weight merge)

**Actual GitHub evidence reviewed:** original S87-J7 PR #504, real Apple MPS + serialized old196 CPU job #42831915482 (SUCCESS); S102 hybrid of S101 with J7 PR #506, real MPS + frozen S101 456 CPU job #42857448935 (SUCCESS as test, scientifically REJECTED); S124 true differentiable prefill-only GPU training PR #528.

## J7 empirically established

Original S87-J7 was a 36M S82-baseline 2-block GT3/9 **joint codebook + donor training** experiment, not current S101's 340 feature-gated shared adaptation. It ran **64 real Apple MPS gradient updates**, in eight fixed **8-step atomic transactions**, with **196/456 old S82 exact full+EOS TRAIN outputs** protected. It checked the actual physically serialized candidate source/GT and re-inferred old protected TRAIN on CPU before each commit. Checkpoints at 8,16,24,40,56,64 were COMMITTED after ZERO observed protected losses; 32 and 48 each LOST one previously correct old TRAIN answer, were ROLLED BACK, and the original parameter tensor values PLUS complete Adam state were rechecked, learning rate halved (3e-5→1.5e-5→7.5e-6). **6 committed, 2 rejected/rolled back.** J7 experiment end physically **0.685140382975 bpp**, **0 new correct full+EOS** in its eight small preflight examples. No independent fresh QA was executed. **Do not misinterpret a technically successful GitHub workflow as new skill acquisition.**

The original J7 source report explicitly states `native_ALL_456_TRAIN_old_skills_guard_completed=False`, `full_CPU_vs_GPU_post_training_parity_completed=False`, and `independent_QA_completed=False`. Thus J7's 196 old TRAIN safeguard is evidence that **rollback works**, not proof of actual 456 original full+EOS retention, nor of better open-ended speech.

## S102 ALREADY attempted the original J7 method on S101

S102 PR #506 ported *algorithm only* to the newer **S101 full 353/456, first 387/456**, original frozen S94/95/98 and CB61 36M profile bank. On real Apple MPS it jointly updated **two already paid S93 core codebooks and original GT3/9**, with one 8-step transaction. The actual physical S101 456 KV-cache CPU verifier at job #42857448935 gave **353→348 full+EOS**, **5 forgotten, ZERO gained**, lost positions **2 second token / 3 EOS**, `rollback_verified=true`, `committed=false`, candidate REJECTED. Bpp 0.6995738373982864, CPU ~0.96809x original. Therefore J7's methodology has already saved S101 from accepting a damaging joint update, but **does not itself yield a better model**.

S112 and later S119/S120/S121 further showed that small protected sets and smooth gradient constraints alone do not guard all 353 originally-correct S101 complete+EOS. Independent one-shot S118 was negative for S115 (30/128 vs S101 31/128) and is now **INSPECTED/EMBARGOED**. S122 created new 128 **TRAIN ONLY** independent-color paired owner questions using old S82 TRAIN vocabulary. S123 original physical S101 vs frozen S115 scored 14/128 full, 39/128 first, zero bidirectional full pairs for BOTH, so this is a legitimate additional development problem, NOT independent QA.

## Exact transplant for S125

Port the **atomic optimizer+parameter checkpoint protocol**, NOT J7's S82 source weights, old 196 count, original old data schedule, or any QA cohort:

1. Start from original frozen 36M S101 physical compressed CB61 + already paid GT3/9 donors, original frozen S94 bank, S95 prompt-only route, S98 neural copy. Train ONLY the existing **340 FP32 S111 conditional coefficients** at S95-route S93 during **prompt prefill ONLY**; frozen other profiles unaffected. Follow S124 three actual S101 differentiable cached steps (prefill ON, continuation and EOS OFF), entirely TRAIN-only S122 newly authored 128 and authorized old 456.
2. Before EACH bounded optimizer block, clone **all trainable tensor values and complete Adam optimizer state**. S87-J7 used 8-step transactions, but S125 SHOULD cap at **1–2 steps per transaction** until physically repeatable safe updates are established; do not allow long unchecked drift.
3. After each block physically FP16 serialize+reopen S101 **340 coefficient sidecar**, include actual prefill-only runtime JSON bytes in physical bpp; verify immutable source hashes and no extra donor matrices.
4. Regenerate the TRUE original S101 S95/S98 greedy KV cache **all original 456 TRAIN complete responses including EOS**, ensure exact preservation of all **353 original correct complete answers AND all 387 original correct FIRST tokens**, not a single aggregated correct count. Also measure 128 NEW S122 TRAIN full, first and all 64 paired both-correct assignments, but do not use S118.
5. For acceptance of a **research transaction only**, require no lost original identities, all-in bpp <=0.700000, paired same-host original S101 CPU ratio >=0.90, original frozen profiles and S95/S98 unchanged. Any loss, first error, physical budget, speed or decoding failure => **atomic rollback of ALL 340 trainable values + complete Adam state**, verify identical reloaded original 456 serialized CPU behavior, reduce LR 2× and continue from the restored safe checkpoint.
6. Keep experiment outcome classifications separate: `transaction_committed` = safe old skills; `new_TRAIN_gain` = improved learning signal; `model_accepted` = FALSE until new never-opened, preregistered **one-shot independent QA**, native CPU/MPS parity and free-form response tests. If safe but no new TRAIN improvement, no model quality claim.
7. S125 synthetic unit code proves the *transaction primitive only*; **it is not a full S125 real-MPS trained model**. The original S124 MPS run should finish and be analyzed separately, without modifying its in-progress workflow. Only then integrate and run the new transactional S125 GPU experiment.

## Exact source references

- J7: `research/calibri2/s87/pilot_transactional_retention_j7.py`, PR #504, GitHub Action #42831915482.
- Already-attempted S101 J7 hybrid: PR #506, GitHub Action #42857448935.
- New model frozen baseline: S101 353 full/387 first.
- S124 cross-color prefill MPS: PR #528, Action #42902875753.
- This research-only S125 atomic rollback primitive: `research/calibri2/s125/j7_atomic_physical_retention.py`; tests are SYNTHETIC, not physical model acceptance.

No LIVE Jessica, Render, Supabase, APK, normal Jessica chat tests, production main changes or codebook weight merges.
