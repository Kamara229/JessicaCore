# NRA2 S3BI — train-only role-head repair after held-out S3BH negative result

## Motivation and scientific separation

- S3BH (CI #43640363396) used its **one-time independent S3AQ EVAL128**; it is now **permanently consumed** as a blind set. Observed source/destination pair identification S3BA 114/128 vs S3BE 110/128, reachability 108/128 vs 102/128. Training improvement from 1848/2048 to 2043/2048 in S3BE did NOT generalize. This evaluation is diagnostic background ONLY, **not** data fed to new code.
- S3BI uses the original **pre-existing S3BA FP16** baseline `4fedf391ee02b1d1c45c82c3690d212ba9a43e9d0da32446f5d32ac6c379bb09` from CI #43590032315, not S3BE. We revert to the prior conservative research checkpoint, not selecting among new weights using consumed EVAL.
- Frozen original Calibri 43M native CPU, Green Tunnel, S3O graph/normalizer, NRA2, and original 616 J7 generations stay unmodified; gate=0, no acceptance, no LIVE, Render, Supabase, APK, QA/TEST or main changes.

## Preregistered single training action, no independent test

Exactly 2048 existing genuine original CPU **TRAIN-only** token sequences, in fixed order: S3M128 + S3X128 + S3AQ TRAIN256 + S3AU512 + S3AY512 + S3BC512; each verified against original J7 and physical SHA. Existing DEV128 and previously-consumed EVAL128 may be named as **forbidden SHA IDs** for overlap checks but must NEVER be passed as examples, tokenizer inputs, model features, targets, or used to choose hyperparameters/checkpoints. No further model inference on either set.

- Init S3BA, torch seed 2026111511, CPU x86_64, AdamW step count **120**, learning rate **2e-5**, weight decay **0.05**, gradient clip 1.0, batch=all TRAIN.
- Cross-entropy roles on the six TRAIN groups with respective weights **[2,2,2,1,1,0.65]** (less chance of memorizing S3BC style).
- Original TRAIN quartet/octets role swap/same-role equivariance regularization weight **0.80**.
- **S3BA teacher role KL** on the exact same permitted TRAIN inputs (temperature 2.0), weight **0.60**, preserving original source-first and destination-first semantics.
- Semantic source-vs-destination distinctness regularization on TRAIN role distributions, weight **0.10**.
- Require original frozen graph unchanged bitwise, original 616 J7 verified by six genuine token captures; physical FP16 SHA256 and reload; **no regression in both-role TRAIN accuracy** versus S3BA on each of the five historical TRAIN groups, or reject this research candidate before any future evaluation.
- This stage reports only **TRAIN** metrics. No S3AQ DEV/EVAL scoring, no role-head promotion, no claims of real free-form EOS quality, all-in <=0.7bpp or paired CPU speed.

## After S3BI (a different branch/stage)

If TRAIN-only guards pass, separately preregister NEW truly independent held-out graph topologies and novel question syntax, disjoint from all old S3AQ and historical datasets. Lock S3BI FP16 SHA and blind protocol before authentic original CPU captures. Only then one-time compare with fixed S3BA baseline; never reuse S3BH EVAL for model selection.

## R1 amendment after TRAIN-only guard rejection (not from heldout)

The initial S3BI research CI **#43641294513** successfully completed all 20 preliminary integrity tests and original frozen-CPU TRAIN2048 training but **rejected the full optimizer head** because at least one of five historical TRAIN role-group counts regressed. No S3BI model was promoted, published as successful or evaluated on blind questions. The S3BJ independent blind128 remains sealed (CI #43641688882 success; SHA `e48faef1a8b164cad9641f1b5946932f15c27baeeab47d829fb5ec754dd46265`).

R1 preserves the *identical* original training objective, seed, 120-step optimizer and all six frozen CPU TRAIN2048 sources. After fitting and persisting the raw TRAIN candidate, it tests only a **predeclared TRAIN-only trust-region family**: physical query tensors = S3BA query tensors + alpha * (raw S3BI query tensors - S3BA query tensors), alpha in [1, 0.75, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]. Choose the FIRST (largest alpha) with (1) zero negative net TRAIN both-role accuracy against S3BA in ALL five older TRAIN groups, (2) strictly positive TRAIN S3BC group gain, (3) at least one physical FP16 query tensor changed; otherwise fail closed. This is **training-set constrained rollback**, NOT validation on either old or new EVAL. Publish raw head separately and a newly SHA-locked safe FP16 research head only when the guard passes, gate 0 and candidate_accepted false regardless.

Previous S3AQ DEV and consumed EVAL, as well as freshly precommitted S3BJ EVAL are **not read/scored/used** in R1; the held-out evaluation will need its own later stage and no retuning following that one-time test.
