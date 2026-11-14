# NRA2 S3AP — scientific postmortem of S3AO one-time independent EVAL128

**S3AO GitHub Actions #43579515862: completed successfully; exact 616 S194 J7 original token digests preserved by S3AN #43572358979.** Native original 43M CPU, four physically frozen role heads, same original S3O graph and original 2241p NRA, output gate 0. Artifact `nra2-stage3ao-once-only-eval128-s3o-s3z-s3ag-s3am`, SHA-256 of archive `d5845c240a20daaf814f35867dd586418e064f31b39f6af4369737b36e43b29e`.

## Precommitted independent EVAL128 outcome
The new S3AK EVAL128 split is **permanently scored, NOT future blind**; it cannot be used as training, tuning, checkpoint selection or re-scored as fresh evidence.

| Predeclared metric | S3O | S3Z | S3AG | S3AM |
|---|---:|---:|---:|---:|
| Reachability correct, all 128 | 65 | **90** | 69 | 68 |
| Both semantic roles correct, all 128 | 64 | **96** | 66 | 66 |
| Source-first reachability, 64 | **64** | 59 | 37 | 36 |
| Destination-first reachability, 64 | 1 | 31 | **32** | **32** |
| Source-first both roles, 64 | **64** | **64** | 34 | 34 |
| Destination-first both roles, 64 | 0 | **32** | **32** | **32** |
| Graph exact, all arms | 128 | 128 | 128 | 128 |
| False positives (all) | 31 | 19 | 30 | 30 |
| False negatives (all) | 32 | 19 | 29 | 30 |

**Paired reachability** on the same 128: S3O vs S3AM = S3AM-only 31 / S3O-only 28 / both 37 / neither 32. S3Z vs S3AM = S3AM-only 2 / S3Z-only 24 / both 66 / neither 36. S3AG vs S3AM = S3AM-only 1 / S3AG-only 2 / both 67 / neither 58.

Important: the earlier **S3AI** scored a *different* EVAL128 and S3O=96, S3Z=91, S3AG=92. Comparing scores across these different distributions does not establish one universal ranking. On S3AO independent EVAL, **S3AM 68/128 did not improve S3O 65/128 meaningfully in a paired sense and regressed sharply versus S3Z 90/128**. The S3AM TRAIN512 512/512 is not evidence of reliable generalization.

## Actual failure boundary
Frozen S3O graph is exact 128/128, so graph extraction on this specific split is not the bottleneck. The *query-role / source-target binding head* is the likely practical limitation: independent word-order transfer and semantic role recognition fail unpredictably. S3AG and S3AM each collapse source-first role accuracy from S3Z's 64/64 to 34/64, despite training-only near-perfect classification. Do not claim a single root cause from aggregates: optimization drift, distribution shift, non-generalized role language, and head initialization are hypotheses, not established facts. A hard 0.5 edge threshold was predeclared; no retuning permitted from EVAL128.

## Safe next experiment, on entirely NEW data only
1. **New precommit S3AQ**: generate disjoint topology + novel phrasing TRAIN/DEV/FUTURE-EVAL partitions, explicitly excluding all S3AK (now scored) and S3AE (scored) prompts and topologies. Train and tune using TRAIN/DEV only. Future EVAL never prefilled until a physical candidate SHA is locked.
2. Preserve **S3Z as historical reference, not promote it**. Compare a controlled initialization from S3Z against from S3AG on TRAIN/DEV *only*, and a variant with high-weight source-first preservation/paired source-destination reversal consistency. Do not select from scored EVAL128.
3. Add per-step role gradient, positive/negative and mention-order balance, train/DEV source-first and destination-first metrics; halt/rollback on DEV regression. Use original frozen S3O graph, 43M original CPU, 2241p NRA, output gate zero.
4. On never-scored future EVAL, compare precommitted models once only with paired source-first/destination-first roles and reachability. Reject if either group regresses materially or candidate is not better than reference.
5. Before any acceptance, independently prove *real* decoder + EOS improvement, packed **all-in bpp ≤0.7**, paired CPU speed **≥90%**, and zero loss on all original **616 J7** tokens.

This S3AP postmortem is only a read-only research record; no new inference, weight change, retune, training, sealed QA/TEST, main/LIVE, Jessica ordinary-chat test, Render, Supabase or APK.
