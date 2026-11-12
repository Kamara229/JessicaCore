# CALIBRI S179 — second-token / EOS diagnostic plan (research only)

## Frozen point of comparison
- Parent: immutable S176 decoding policy and all accepted safety guards, evaluated with the S178 paired REAL CPU J7 harness.
- S178 CI **43317481920** PASS, archived artifact `calibri-s178-paired-5x10-counterfactual-real-J7`. S176 original TRAIN: **428/456** full-EOS correct; new TRAIN **128/128**; protected **64/64**; novel **64/64**. S178 adaptive 5→10 candidate: **zero** changed final full-EOS outputs, **zero** gains, **zero** losses. 36 selector changes were *not* answer changes. Measured all-in bpp ~0.68029; CPU ratio ~0.97108.
- Therefore **28 old TRAIN examples are wrong under S176**. Their answer labels are *evaluation-only*, NEVER inference inputs.

## Single S179 research question
Which failure mechanism prevents the frozen S176 model and S178 adaptive 5→10 voting from producing a correct complete second token plus EOS on the 28 incorrect original TRAIN examples, while retaining every previous correct answer?

## Diagnostic protocol (no training or promotion)
1. Freeze exact S176 weights, tokenizer, decoding, numeric profile, S177 5→10 selector, S174/S176 guards, seeds, prompts, and J7 cohort digests. Reuse S178 CPU harness; no writes to protected runtime components.
2. **Inference first, labels later.** For all original **456** prompts (not merely the 28 errors), collect anonymized SHA256 prompt IDs, pre-/post-guard first-token and second-token top-k logits and ranks, EOS probability/rank, beam/vote disagreement (5 vs 10), guard decision and veto reason, fallback path, and exact full decode. Collect only model-derived candidate evidence. Do not access expected ids or per-example correctness to select/rank candidates, route prompts, or adjust thresholds.
3. Save immutable complete output and candidate traces. Only after both frozen and shadow outputs are finished, join known TRAIN labels offline and identify the **28** failing cases. Group failures by: wrong token 1, wrong token 2 conditional on correct token 1, missing/early EOS, routing/guard veto, selector null/proposal cancellation, and unchanged votes. One case may have multiple reasons; record explicit counts and representative SHA256 IDs.
4. Distinguish **candidate generation** from **guard filtering**: compare raw S160 head, 5-cycle and 10-cycle consensus, actual S176 guard chain, and final full-EOS. Confirm exactly why 36 raw selector changes produced 0 final changes. Do not infer that votes statistically improve accuracy without final output gain.
5. Propose **one** minimal, deterministic S179 *shadow-only* second-token/EOS candidate policy using logits from the same frozen model (e.g., disagreement-aware rerank or calibrated EOS stopping). No gold-derived rule, memorized prompt exceptions, source lookups keyed by TRAIN identity, or tuning on the 28 errors.
6. Candidate must be evaluated paired across **all 456 old + 128 new + protected64 + novel64**, including full EOS. Report gained/lost full answers *separately*, first-token regressions, net gain, per-case vetoes, true all-in physical bpp, and native CPU runtime. Add held-out prompts not used for policy selection to guard against overfitting. Never accept gains that forget any previously correct answer.
7. **Acceptance**: strictly positive new full-EOS correct answers; **0** lost previously correct full-EOS and **0** first-token regressions in all protected cohorts; full EOS; bpp **<=0.7**, native CPU speed **>=90%** S176 reference; no changes to LIVE, Jessica, Render, Supabase, or APK. If any gate fails, reject candidate and preserve raw diagnostic artifact without promotion.

## Immediate next action
Implement **read-only trace capture** only as a separate S179 research script and unit tests for label isolation and guard trace completeness. Do **not** launch a test in normal Jessica chat; do not start CI until the trace-only script and gate tests are reviewed.
