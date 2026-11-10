# CALIBRI S118 — first independent never-opened QA for frozen S115 prefill-only

RESEARCH ONLY. This branch evaluates a FROZEN physical candidate, not model training or tuning. The previous inspected QA/TEST from all S and V versions remains embargoed. Do not merge or promote to main/LIVE Jessica/Render/Supabase/APK.

## Why S118 is now warranted

S115 physically demonstrated, on the original 456 authorized TRAIN prompts, the FIRST no-forgetting gain in recent S103-S115 research: 354/456 complete+EOS correct (1 new, 0 previously correct lost) and 388/456 first (1 new, 0 previously correct lost), all-inclusive physical 0.6998282595499457 bpp, paired S101 speed ratio 0.9907299146 on the same CPU host. The candidate consists of unmodified original S101 compressed CB61, S94 bank, S95 router, S98 copier and frozen original GT3/GT9, plus S112's previously trained 340 FP16 conditional GT coefficients ON only during the **first prompt-prefill KV decoder call**. It is TRAIN-only evidence; no valid statement of generalization exists yet.

S116 independently found that all seven S112 old-full forgotten cases involved *second token (1)* or *final EOS (6)*, and S98 copier state remained stable. S115 prefill-only removes continuation interventions and avoided all seven old errors.

## Pre-registered fresh, never-before-opened QA

S118 runs exactly ONE model-comparison evaluation on a freshly authored deterministic cohort (seed 1180118) with **64 novel counterfactual owner-binding pairs =128 questions**, no training or inference based on their answers. It is intentionally different from original S82 TRAIN, and no old inspected QA source is imported. It uses:

- Entirely NEW question frames and NEW owner names, distinct from S82 TRAIN_NAMES/TRAIN_FRAMES; excludes all 24 original S82 TRAIN nouns, choosing tokenizer-compatible heldout nouns plus a fixed extra noun vocabulary that was not used to train S82.
- First 32 pairs have same color in both owner assignments (tests second-token noun copy). Remaining 32 have different colors (tests first-token owner binding). Four colors are the known task-domain concepts red/green/blue/yellow, but the owner/noun association and wording are novel. Each pair supplies two questions swapping both owners' values, with different ground-truth answers. Exact gold = two tokens plus EOS; all labels are generated from the template BEFORE any model scoring.
- If any new question SHA256 collides with any of the immutable historical 456 TRAIN prompts or a new question, or fewer than 12 disjoint tokenizer-compatible nouns exist, **FAIL**, do not use old QA as fallback.
- The recipe is frozen in the branch before the workflow is first launched. DO NOT change frames, names, seed, nouns, or evaluation parameters after seeing its results; do not use this QA to pick alpha, train, choose layers, or gate.
- Compare original S101 and SHA-verified exact S115 prefill_only physical sidecar/config on all 128 via SAME frozen S95 route, S98 copy and original cached CPU autoregressive EOS generation. Aggregate exact full+EOS, first, gains, losses, 64 both-sides-correct pairs, performance by same-/different-color category and by frozen S95 route. Print aggregate stats ONLY; never raw QA text, token labels, per-question oracle tables, or candidate answers.
- Independently account physical all-inclusive bpp, compare against original S115 saved SHA/config; paired 48 question CPU ratio. No model acceptance on these data, even if the candidate improves. This QA becomes INSPECTED immediately after the first run, with NO future fitting/retuning against it.

## Limitations

This holdout deliberately tests transfer of a two-word color/noun binding skill, not broad scientific knowledge or coherent free-form multi-sentence prose. Even perfect scores cannot prove parity to 135M FP32, general SciQ/BoolQ or unrestricted natural conversation. A separate truly fresh predeclared comprehensive holdout and canonical S82 CPU/native-C/MPS checks are required before any future production decision.

No model weights change. S115 original research artifact is untouched and remains unaccepted.
