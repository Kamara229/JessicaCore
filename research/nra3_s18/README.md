# NRA3 S18a — REAL original 43M+GT native C, 4 lexical-family + 6 construction TRAIN-internal CV

## Motivation

S17c TRUE independent original43M+GT native C physical S16b frozen EVAL24 GitHub Actions **#43902361580 SUCCESS** (computational integrity) **FAILED independent linguistic generalization**: trained rank8 directional valence on old72 TRAIN had hard-link **F1 86.60%** (181/216 gold correctly linked), but new 24 EVAL gave **TP=9/74, FP=30, FN=65; Precision 23.08%, Recall 12.16%, F1 15.93%** across 856 all-direction candidates. The source model was physically frozen before those sentences existed; old S17c 24 are now **CONSUMED**, cannot be used for new TRAIN or for iterative testing/tuning. Original upstream Green Tunnel S95 conditions on **full prompt**, so end-to-end streaming causality is not established despite strict pair scorer wordpiece positions.

**Scientific conclusion:** S16b's good TRAIN metrics mostly measure fitting 72 highly patterned samples and do NOT demonstrate abstraction; no production promotion, candidate always `output_gate=0`.

## Concrete S18a research-only cross-validation

Branch from the original **pre-S17 S16b research branch** to exclude S17c EVAL source physically. The complete S9 old72 TRAIN SHA `b09d60a77a17d30c2dbf846e40ab97860def8635e2bf31a5b0ab8981832b744d` contains 12 independent lexical families × six construction templates; S16 gold SHA `9843c35ee9aa75b8f5c9c963e441589dd2f0292e41ec7933971d5aa3dcd365d6` has 216 true directed predicate→argument roles.

PRECOMMIT two complementary **OLD TRAIN internal**, **NOT new untouched independent EVAL**, axes:

1. **Four complete lexical-family fold tests**: hold out three entire lexical families per fold (source families `i%4`), train from scratch on remaining nine full families (54 sentences) and evaluate the 18 **original TRAIN-internal** unseen-to-this-fold sentences. Across all four folds, every original family held out exactly once.
2. **Six complete syntax-construction fold tests**: hold out each original construction in turn, train from scratch on remaining five (60 sentences) and evaluate 12 **original TRAIN-internal** unseen-to-this-fold examples. Every original template held out exactly once; note folds that exclude all recipient-subject cases also lack any source-role examples in TRAIN (valid visible weakness, no selective skips).
3. In each fold, use ORIGINAL S10 real Calibri43M post-GT CPU hidden from these 72 original TRAIN, ORIGINAL all 43 C-packed CB59 matrices and frozen learned S11 39param router. Enumerate **ALL** 1,920 ordered wordpair candidates over all rows, even nonverbs, negatives and punctuation. Gold labels enter only the fold's supervised TRAIN target, NOT frozen model scoring and NOT split selection (split only uses immutable original family/template metadata).
4. Train the identical old S16b **2,309 physically paid FP16 params (4,665 bytes each)** from scratch for **exactly 150 predeclared steps**, identical optimizer and fixed loss weights for all 10 folds. **NO checkpoint selection, hyperparameter adjustment or early stopping based on heldout fold.** Physically reload round-to-FP16 weights before every reported score; archive all 10 SHA-verified full physical binaries and both train-/heldout-F1 metrics. The same candidate is NEVER promoted.
5. Report micro exact-directed-role P/R/F1 across the internal group-heldout pairs for each axis (216 gold once per axis), plus full per-role, per-surface direction and per-fold hard confusion. Important: performance here is still on previous TRAIN identities, albeit held out from each fold's weight updates. It does **not** reverse or supersede the already measured S17c independent failure.

**Decision gate:** If family-template heldouts collapse vs train, S19 should add an explicitly constrained Russian morphology/syntax module and diversify NEW TRAIN grammar/lexical/verb constructions using TRAIN only, before any future NEW blind independent EVAL. Do not infer an answer or negation, pronoun coreference, or active EOS generation quality from directed link F1.

No S17 source/gold or prior S12c/S15c consumed EVAL/old EVAL128/reserved S3CI is loaded, and no LIVE Jessica, ordinary chat tests, Render, Supabase, APK, main or original weights are changed. Standalone raw NRA+S11+S16b bpp=0.8625 > 0.7; all-in <=0.7 bpp and CPU>=90% are NOT certified.
