# CALIBRI v80 — explain why v79 answer-supervised low-rank source was rejected

**Research-only, 2026-11-03.** This is the next engineering gate after a **completed and verified** genuine pretrained 135M v79 experiment, not a new assumption of recovered quality.

## Verified prior experimental outcome

CI run **41550969128** succeeded. The physical original CB59 trained checkpoint was **11,229,303 B / 0.667839413 bpw**. The paid rank4 plus output bias correction was fully stored with the original source as one **11,725,487 B / 0.697348923 bpw** checked archive. No new original dense pretrained 2D weights loaded into the student runtime; the original K32V8 5bit operators executed via C, rank4 via CPU Torch. TRAIN FIT included 24 SciQ + 48 BoolQ original source questions. TRAIN VALID included independent 12 SciQ + 16 BoolQ original TRAIN questions, with WikiText/AG News TRAIN VALID language KL safeguards.

Two repeated v79 answer-conditioned optimizer runs **rejected all new factor checkpoints**, froze step0 parent source (same v77 source SHA `4d5c1258924b77a458b1d08d962914f6c8845632f8604a0a04e929cd43ef759c`) and did NOT change the paid model. On independent new v79 TEST, original teacher supported SciQ scored 23/27 versus CB59 7/27 and paid source 9/27; BoolQ teacher 21/32, CB59 13/32, paid source 12/32. Real greedy output distinct generated tokens 0.7222 for original teacher and 0.0694 for paid source. **No semantic competence recovered**. Improved language KL/NLL relative to original CB59 was inherited from v76, NOT a new v79 gain.

## v80 experiment: predeclared TRAIN-VALID failure-mode diagnostic only

The v79 CI retained pre-TEST numerical artifacts with all four predeclared TRAIN checkpoints (`0,5,10,18`), original-source checksum, per-task accuracy, conditional correct-answer CE, per-corpus KL, preregistered pass/fail guards and both same-host repeats. **Use only these original TRAIN and TRAIN VALID files from original run 41550969128. NEVER use its independent v79 TEST audit for source selection, optimizer changes or model promotion.** The CI downloads exact artifact `calibri-v79-source-and-tested-QA-numerical-audit` (file includes both categories; script is explicitly restricted to v79_train_first.json / v79_train_second.json / v79_FROZEN_before_TEST.json).

For each original source snapshot: (a) verify actual complete model size and bpw, label/question counts, pre-TEST freeze, reproducible source SHA, selected step and original selection decision; (b) re-derive gains in SciQ/BoolQ TRAIN VALID accuracy and answer-choice CE vs parent; (c) re-derive signed WikiText/AGNews TRAIN VALID teacher-KL regression vs parent; (d) independently re-evaluate the original **both nonregressing accuracy; ≥0.05 absolute accuracy gain in at least one; no >0.25 KL nats regression per corpus** promotion requirements; (e) identify each candidate's *actual* failed guard, compare independent same-CPU-run diagnostics and any CE–accuracy mismatch. This cannot create a better candidate; it is a root-cause investigation of a negative result.

CI must produce reproducible machine-readable `v80_original_TRAIN_VALID_failure_audit.json`, a concise numeric record of failures, while avoiding any pretrained checkpoint, original QA text, student logits, label processing, or TEST data. The **scientific result can be null**: the earlier v79 rejection must remain honored. Do not imply training v80 has started or new QA TEST data was used. No changes to Jessica, production, main, Render, Supabase or APK.

## Next falsifiable learning hypothesis after analyzing the causes

The parent CB76 output correction improves heldout KL in two text corpora but retains extraordinarily low answer accuracy and repeated greedy tokens. If v79 new supervised checkpoints have lower answer-choice cross entropy but unchanged or worse TRAIN VALID accuracy, source-fitting sample size/overfitting and output-head-only representational capacity are plausible causes, not proven causal attribution. If new snapshots violate both language KL and QA gates, the weighted objective is misaligned and needs a new dedicated **TRAIN FIT/VALID** design—do not tune on the existing independent v79 TEST rows. Explore sparse internal tensor corrections, teacher-versus-packed activation transport and constrained loss for greedy-collapse, **count every stored byte and real CPU op** under ≤0.700 bpw. Improve teacher-anchored supported SciQ and BoolQ and generation using *new*, independent QA and corpus splits before claiming success.

## Actual CI

Pending first v80 diagnostic CI. No new learned model or numerical checkpoint is claimed until verification.
