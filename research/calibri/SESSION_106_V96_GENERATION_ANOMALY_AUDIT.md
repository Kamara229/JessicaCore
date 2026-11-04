# CALIBRI v106 — cheap generation anomaly signals; original v96 frozen source

Goal: strengthen v96's empirically weak token-id repetition alarm WITHOUT
treating morphology as answer quality. Use only original v96 completed
Actions #41739479888, original JSON SHA256
`917c786af2bb54c8ae1f9b88029230c54e4761373fe791f99263939da4ac2ac5`,
and the already recorded 12 genuine 32-token continuations from
original FP32, native CB59 and same-byte v92. No model is rerun.
Full result retains source prompt hashes, first visible answer and
numeric features for auditing.

The v96 boolean `repeat_over_75pct` checked whether a *single
token ID* occupied over 75% of 32 tokens. This alarm missed repeated
words split across multiple code IDs and word fragments repeated in
different combinations. Native output remains meaningless when many
distinct token IDs are emitted.

v106 two *exploratory* source-only indicators:
- Fraction of 5-character alphanumeric windows whose 5-gram occurs at
  least twice in the same generated text; threshold >0.50.
- zlib-compressed UTF-8 size / original UTF-8 size; threshold <0.60.
Flag unusual repetition only if BOTH and >=32 alphanumeric characters.
Record fake-role `system/user/assistant` lines as a separate anomaly.
Do not infer answer truth from any lexical threshold.

IMPORTANT: thresholds are chosen AFTER seeing v96's known outputs.
Any apparent sensitivity/specificity on that same sample is an
**in-sample, posthoc** demonstration, NOT independent validation,
unseen generalization, or a model quality PASS. Detector requires
out-of-sample TRAIN/VALID sources with independent human review before
autonomous use by Jessica. No dynamic score tuning or new model
selection here.

Prior v105 verified the original strict-prefix answer grader
misclassified factually correct original FP32 answers: manual first
response review FP32 4/8, CB59 0/8, v92 0/8. The v103 independently
labeled chat-template BoolQ trial showed native CB59 and physically
paid v102 predict YES for all 24 questions, not meaningful accuracy.
v104's newer eight-bank source retained nonsensical generation.
No changes to archived source weights, model training, frozen v89
official TEST, main/Render/Supabase/APK/live Jessica. Facades v106:
signals, source-only audit, launcher, toy tests and isolated CI.
