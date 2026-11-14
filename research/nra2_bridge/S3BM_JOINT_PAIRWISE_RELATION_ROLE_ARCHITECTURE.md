# S3BM: joint relational semantic role architecture (research, zero gate)

## Problem and non-leakage
Two independent sealed heldouts S3BH (old EVAL128, CI #43640363396) and S3BL (NEW independent EVAL128, CI #43644153544) are **consumed**. Reusing either one's prompts, syntaxes, labels or CPU tensors in training, hyperparameter selection, checkpoints or extra scoring is forbidden. S3BL found 63/128 S3BI roles and 64/128 S3BA roles under newly phrased questions despite physically identical graph results 128/128. Thus more training on the old independent S3O mention head cannot establish generalization.

## Proposed actual architecture
File `stage3bm_joint_semantic_pair_role_head.py` implements `JointSemanticPairRoleHead`, **not** a synonym lookup or a changed architecture on the original Calibri or graph. Frozen S3BA's `LearnedQueryRoleHead` initializes an exact old-logit baseline. A new trainable shared `anchor_query` sees the left, entity and right-context span for **each** of exactly two literals, then cross-attends through the **whole query clause** using `context_key` and `context_value`, producing mention-conditioned cues outside the old nine-token windows. The new `side` projection combines local anchor/context, whole-clause attention and relative mention position.

An ordered pair comparator computes `f(mention0-minus-mention1, global_clause, symmetric_product, delta_position) - f(mention1-minus-mention0, global_clause, symmetric_product, -delta_position)`. This **antisymmetry is structural**: permuting candidate index order only (with literal span offsets) cannot change predicted source/target distributions. The resulting correction increases one candidate's source logit and decreases its destination logit, with opposite update to candidate 1. Both-role predictions must therefore remain coupled. This is a learned semantic relationship; no source/target gold, graph labels or output answer can reach forward.

Initialize final pair-score weights and bias to **zero**. Before any training, the new head exactly reproduces frozen S3BA, so no accidental generalization claim. The old base and S3O graph have all gradients disabled and remain bitwise identical. Trainable pair parameter count is separately guarded, at no point affecting LIVE, frozen 43M core, Green Tunnel, NRA, S194 616 J7, Render, Supabase, APK, QA/TEST or main.

## Criteria and next research phases
1. S3BM standalone CPU unit tests: baseline equality, candidate-permutation equivariance *with nonzero residual*, gradients restricted to new pair layers, graph isolation and invalid offset fail closed. **No EVAL data or CPU adapter inference**.
2. S3BN training: use precisely the six existing original frozen 43M CPU **TRAIN-only** corpora (2048 examples), initialize from frozen S3BA physical SHA `4fedf391ee02b1d1c45c82c3690d212ba9a43e9d0da32446f5d32ac6c379bb09`. Freeze original graph+base, train ONLY added pair parameters, enforce older TRAIN replay guard, physical FP16 SHA and 616 J7. No use of spent S3AQ or S3BJ EVALs.
3. Pre-register a **third** new truly unseen graph+grammar split before any CPU EVAL captures or model scores, physically freeze the new S3BN candidate and baseline first; run only one blind comparison. Do not assume the new architecture generalizes based on TRAIN accuracy.
4. Full two-word EOS, physical all-in bpp ≤0.7 and paired CPU speed ≥90% CB59 are distinct future gates, never inferred from standalone research head scores.

All output gate=0 and candidate_accepted=false until external approval.
