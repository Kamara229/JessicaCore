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

**Verified CI 41554768160 SUCCESS:** fully parsed original saved v79 TRAIN-only checkpoint audits, reconstructed source-budget and every original predeclared QA/KL gate. No TEST data was opened, no weights trained, no candidate promoted. See below.

## VERIFIED source-only failure audit — GitHub Actions run 41554768160

Successful Github Actions run **41554768160**, original code branch and pinned original numerical artifacts from completed v79 run **41550969128**. Exactly **11,725,487 bytes / 0.697348923 bpw** remains the original full source budget, and SAME-RUNNER SHA agreement of v79 full source was verified. The independent audit recomputed original TRAIN VALID metrics and original pass/fail requirements; **zero new candidates are eligible**.

These are **signed differences versus the prior TRAIN-fitted v76 parent**, not TEST results. QA accuracy positive means improvement; KL positive means **quality REGRESSION** relative to pretrained FP32 teacher.

| Original v79 candidate snapshot | SciQ TRAIN VALID gain (12) | BoolQ TRAIN VALID gain (16) | WikiText TRAIN VALID KL Δ (nats) | AG News TRAIN VALID KL Δ (nats) | Verified failing gates |
| --- | ---: | ---: | ---: | ---: | --- |
| Step 0 (unchanged paid v76 source) | 0 | 0 | 0 | 0 | Parent/no ≥5pp gain |
| Step 5 | **−2/12** (−16.7pp) | **+5/16** (+31.25pp) | **+0.25268** | **+0.32537** | SciQ regressed AND both language KL constraints failed |
| Step 10 | **−1/12** (−8.33pp) | **+5/16** (+31.25pp) | **+0.22079** (passes) | **+0.41873** | SciQ regressed AND AG News KL failed |
| Step 18 | **+1/12** (+8.33pp) | **+4/16** (+25pp) | **+0.26440** | **+0.50674** | Both language KL constraints failed |

**Key real finding:** QA supervision does affect functional answer predictions within the frozen TRAIN VALID set. Step18 even improves raw accuracy on BOTH SciQ and BoolQ TRAIN VALID, but fails the predeclared absolute +0.25-nat KL nonregression guard on BOTH corpora—by ~0.0144 nats on WikiText and ~0.2567 nats on AG News. Early steps 5 and 10 increase BoolQ scores while reducing SciQ accuracy and also harming language KL. Therefore **declared selection of unchanged v76 parent step0 was correct; v79 QA source was not improved.** Do not retrofit thresholds to promote step18 now that heldout v79 TEST has already been measured. The observed TRAIN-VALID raw QA gains are based on **tiny 12- and 16-question sets** and cannot support correctness on new unseen QA.

**Diagnostic interpretation:** Fixed 0.10-weight teacher-language cross-entropy anchor in v79 allowed language KL degradation while answering small supervised sets; the failure is a real quality–accuracy tradeoff, not solely an optimizer no-op. This does not prove an internal representational bound; alternate objective constraints, greater QA source diversity and internal-layer correction may improve things. These findings are from exact saved v79 TRAIN VALID metrics and have not retrained or altered any source. The v79 independent TEST result stays scientifically NEGATIVE and cannot be reclassified as success.

### Concrete next hypothesis (v81, NEW-data protocol required)

Pre-register a new experiment that actively **enforces both-domain teacher-KL ceilings during training**, rather than only a small constant penalty: compare projected updates / dual-constraint penalty versus original QA objective, use a wider, disjoint SciQ/BoolQ TRAIN FIT and TRAIN VALID split and fresh never-touched TEST. Keep physical CB59+paid source ≤0.700bpw and original direct native kernel; consider correction across earlier attention/MLP tensors if output-head-only further fails. Evaluate both teacher-sane QA and new genuine greedy generation diversity on independent tests. Do not inspect, rescore, or tune further against old v79 heldout QA as a new optimization target; no model deployment to Jessica.

**Verified Actions:** https://github.com/Kamara229/JessicaCore/actions/runs/41554768160
**PR:** https://github.com/Kamara229/JessicaCore/pull/210
