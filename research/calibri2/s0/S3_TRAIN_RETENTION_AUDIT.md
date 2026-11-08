# CALIBRI 2.0 — S3 TRAIN-only retention audit (2026-11-08)

Source: GitHub Actions run 42496717434, job 128780525811, training log `CALIBRI2_S3_FIXED_TRAIN_MILESTONE`; source `research/calibri2/s3/diverse_task_replay.py`. This document does not use QA/DEV/VALID/TEST to choose training settings.

## Fixed-step TRAIN results (full answer including EOS)

| step | total /176 | old yes/no numeric /12 | old arithmetic /12 | new comparison /32 | new two-word /32 | new sentence /16 |
|---:|---:|---:|---:|---:|---:|---:|
|0|64|12|11|5|0|0|
|640|113|11|8|22|0|0|
|1280|120|12|7|26|3|0|
|1920|127|12|10|28|3|2|
|2560|124|11|10|26|3|2|

The final TRAIN score declined 127 to 124 between steps 1920 and 2560, including old numeric yes/no 12 to 11 and new comparison 28 to 26. Thus 124/176 is **not stable retention**, and selecting step 1920 post hoc would be improper. New two-word 3/32 and sentence 2/16 show *some exact TRAIN multi-token outputs*, but not robust language generation. Full per-example decoded texts, repetition behavior, and EOS failures must be inspected from the saved artifact before attributing causes.

The physically reloaded checkpoint reproduced the in-memory TRAIN probe: 124/176. Exported packed CB59 is 1,932,195 bytes, 0.6910469807298895 bpp, SHA-256 `e4d9a4ef7705c660b803fd823263d0eceb804b4732afdfac30a38884fc96d159` as reported by the training process. Independent local byte hashing and native CPU >=90% remain unverified.

## Next bounded TRAIN-only protocol (not launched)

Keep architecture 320x6, 22,368,320 parameters and physical CODEBOOK+5bit <=0.700 bpp. Start from a predeclared packed checkpoint, record its hash and prohibit choosing it from QA. Predeclare a fixed training-only curriculum containing previously mastered opposite yes/no, color-binding, comparison, two-word copying, and short sentences; ensure each opposite side is rehearsed and track each side independently. Retain the already observed step-1920-to-2560 forgetting as a failure mode to investigate, **not** a reason to pick a best TRAIN checkpoint. Record exact decoded responses, token IDs, EOS, repetitions, first-token diversity and old/new retention at fixed milestones. Prefer a short cost-capped diagnostic run over another long 2560-step sweep. No hyperparameter search, no public split access during training, and no scale-up until stable multi-token answers are demonstrated. Native C execution and a same-hardware comparable CPU reference must be measured before claiming >=90% speed. No FP32 parity claim.

Do not modify main, LIVE Jessica, Render, Supabase, APK, or legacy 135M v001. No 360M.
