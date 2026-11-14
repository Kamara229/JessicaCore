# S3AS: preregistered S3Z-initialized query-role retraining protocol

Status: research-only, pre-training and pre-DEV-scoring protocol. This file commits the exact experiment design before any optimizer step. It does **not** train or score a model.

## Frozen ancestry

- S3AQ split preregistration run **43579926284**, commitment `c55799793263d5b4175fdfd62c293bcc2cb1794c73fa6ba856cf8b0a961f7839`.
- S3AR native original 43M CPU capture run **43580373587**, success, physical all-token FP16 SHA `5c44f4d52b5f2f618b93885f288975e269a4d44d21506cd7906368fd20467f2b`. Captured **TRAIN256 + DEV128 = 384** true CPU readouts; original **616 J7 preserved**; EVAL128 never prefilled, trained, or scored.
- Parent **S3Z**, locked FP16 SHA `60a7121b8270f647909776b58dc21b07c6cb885fd6ebeb6cb3158b23221f6e9f`, from original S3Z run **43532157094**.
- Frozen S3O original graph SHA `40a8903d5419cd8641b5d01e75e429e10b449304b3ba140d2245bf7f8a9d00b6`.
- Old TRAIN-only replay: S3M original TRAIN128, S3X reversed TRAIN128, each with real original frozen CPU features. No previously scored EVAL enters any loss or selection.
- **Absolutely untouched**: S3AQ independent FUTURE EVAL128, original S194 sealed 616 J7, original Calibri 43M weights, Green Tunnel, standalone NRA2 2241p, LIVE/Render/Supabase/APK/main/QA/TEST. Output gate stays exactly zero.

## One fixed candidate; no post-hoc hyperparameter search

1. Load original S3Z as the initialization, freeze graph branch and normalization parameters; optimize **only** the learned semantic-role query head.
2. Use only 512 training rows: S3M TRAIN128 + S3X TRAIN128 + S3AQ TRAIN256. A separate S3AQ DEV128 is not visible to the optimizer, gradient, teacher, or selection before physical candidate lock.
3. Fixed seed **2026111494**, AdamW lr **0.0002**, weight_decay **0.02**, gradient clipping norm **1.0**, **160** optimizer steps, native CPU only.
4. Loss: **2×(old S3M role CE) + 2×(reversed S3X role CE) + 1×(new S3AQ role CE) + 0.15×(TRAIN256 quartet equivalence-and-directional-reversal role symmetry)**. Quartet loss follows the existing S3AM math, not scored EVAL.
5. Save one physically reloadable FP16 role-head candidate, calculate SHA256, verify bitwise unchanged graph, S3Z original physical SHA, same S3O graph and exact output gate zero. **No candidate accepted.**
6. **Only after FP16 is physically locked**, open DEV128 once for **paired** S3Z-vs-S3AS role accuracy, including source-first/destination-first **source**, **destination**, **both**. Record positive/negative differences and any source-first regression, without retraining or selecting another candidate.
7. Never open FUTURE EVAL128 in S3AS. Future separate candidate-acceptance decision requires independent full EOS/CPU speed/all-in physical bpp/no-forgetting checks and an independent evaluation protocol.

## Interpretation

S3AO once-only EVAL128 had S3Z both roles **96/128** versus S3AM **66/128**. This is prior scored exploratory evidence, never reused as a blind tuning set. TRAIN fitting alone is insufficient; the key objective is **generalized correct semantic-role binding in unseen source/destination phrasing** without losing previously acquired source-first competence.
