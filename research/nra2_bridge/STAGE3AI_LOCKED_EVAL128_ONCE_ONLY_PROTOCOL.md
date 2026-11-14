# NRA2 S3AI — precommitted independent EVAL128, one-time comparison only

**Status: protocol only. No candidate evaluation has been executed in S3AI.**

This is strictly research-only work. Jessica normal chat remains a relay and does not run tests. No LIVE, Render, Supabase, APK, main, original weights, accepted weights, sealed QA/TEST, or Green Tunnel may be changed.

## Immutable inputs fixed *before* looking at EVAL128 scores

- New independent EVAL128: S3AE precommit SHA256 `0ccbb2b19b3feb93b79f5dcbc892f2e414a564b6e4d83e52997098346e4f480f`; 32 novel graph topologies × 4 held-out grammar templates, 128 prompts; 64 reachable / 64 unreachable; 64 source-first / 64 destination-first.
- S3AH true original frozen 43M CPU EVAL128 capture: successful GitHub run `43539973968`; native token320 FP16 SHA256 `ef25951db280c9968932c200ba7b4b5c58625bdd69ac200e4e08e38861cb0f7`; graph-label FP16 SHA256 `cbfa5c27f114b0047d2f1c54917414bee7b7aeb1d55e3172608b1ae7caac494e`; hidden320 SHA256 `bb76e56825397ce4fd9a863348fb267b6fc1dc42f6b0ce36ba46f5493303c4a4`. Candidate EVAL128 predictions never scored in S3AH.
- Frozen S3O original graph + query role weights: `40a8903d5419cd8641b5d01e75e429e10b449304b3ba140d2245bf7f8a9d00b6`.
- Frozen S3Z role-only trained candidate: `60a7121b8270f647909776b58dc21b07c6cb885fd6ebeb6cb3158b23221f6e9f`.
- Frozen S3AG role-only TRAIN512 candidate: `2c23de87efc2369ed6fc0880604420070c116a4ee70b66a4eb0a946f1afeb118`; training run `43539383181`. Its observed 512/512 TRAIN is not a generalization claim.
- Independent original NRA 2241p weights remain frozen; same original S3O graph + normalizer for all three arms. S194 original 616 generated CPU/J7 tokens must remain digest-identical; preserved by S3AH.

## Single predeclared experiment

On the **same** 128 pinned S3AH native CPU token features, perform **one** evaluation of all three physically frozen arms **S3O vs S3Z vs S3AG**. Use exact original S160 tokenizer and S3O query mask; build graph hard edges using `sigmoid(logit) > 0.5`; compare source/target role predictions by `argmax`, and use **soft** role probabilities with frozen NRA to compute hard reachability using `NRA(logits) > 0`. No graph, thresholds, masks, labels, prompts, weights or score conditions may be adjusted after observing EVAL128 results. Confirm all three arms produce bitwise identical graph logits and unchanged graph weights/normalizer.

**Report, without threshold-based cherry-picking:** exact graph correctness; correct source, target and **both roles**; reachability hard accuracy, false positives and false negatives for each arm; paired S3O↔S3Z and S3Z↔S3AG differences; per-query-style (4 × 32), per graph family, source-first (64), destination-first (64), positive and negative (64 each); per-example hashes and correct/incorrect indicators. Exact role/head and reachability measures are research proxy metrics, **not full+EOS generation improvement**.

## Gate and safeguards

One attempt, immutable output artifact, zero optimizer steps, no tuning or repeated blind passes. Never use EVAL128 for subsequent training or model selection; if code/pinning fails before predictions, correct the infrastructure fault **without examining any EVAL metrics** and record why. If scores are observed, the split becomes permanently non-blind and must not be reused as fresh EVAL in later experiments.

Original output gate remains exactly 0.0; candidate_accepted remains false. Full decoder J7 gain, paired CPU speed and all-in <=0.7 bpp must be independently measured before any candidate acceptance. No deployment or merge into LIVE.
