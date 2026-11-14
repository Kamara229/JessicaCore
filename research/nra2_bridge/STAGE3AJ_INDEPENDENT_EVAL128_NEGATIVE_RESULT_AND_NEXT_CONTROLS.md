# NRA2 S3AJ — frozen postmortem after S3AI once-only EVAL128

Research-only, no LIVE, Render, Supabase, main, QA/TEST, accepted weights or Jessica tests. No code, weights or threshold are changed by this document.

## Locked evidence (do not repeat the blind evaluation)
S3AI frozen three-way result: GitHub Actions **#43541542901**, successful, artifact **14092047898** (`nra2-stage3ai-once-only-eval128-s3o-s3z-s3ag-frozen-cpu`). All original **616 J7** token digests preserved and original `output_gate=0`. Native real CPU 43M S3AH EVAL128 features and prereg S3AE EVAL128 data; only original S3O graph weights, same normalizer, frozen NRA. Graph exact **128/128** across three arms.

| Precommitted metric | S3O | S3Z | S3AG |
|---|---:|---:|---:|
| EVAL128 reachable yes/no correctly | **96/128** | 91/128 | 92/128 |
| Source and destination roles both correct | **98/128** | 92/128 | 96/128 |
| Source-first reachability | 64/64 | 64/64 | 62/64 |
| Destination-first reachability | **32/64** | 27/64 | 30/64 |
| Destination-first both roles correct | **34/64** | 28/64 | 32/64 |
| Hard false positives (all) | 16 | 19 | 17 |
| Hard false negatives (all) | 16 | 18 | 19 |

These are measured once-only bridge reachability/role proxies, **not** full-decoder + EOS gains.

Against S3O baseline, S3AG loses 4 reachability and 2 both-role correct on independent EVAL128; against S3Z, S3AG gains only +1 reachability. S3AG TRAIN512 role performance cannot be generalized to held-out new grammars. No candidate is accepted and no deployment is authorized. A drop in EVAL128 performance warrants no changes to frozen weights or acceptance thresholds.

## Postmortem boundaries

The S3AE EVAL128 set has now **been scored** and is permanently not blind. Do not train, select, retune, choose checkpoints, fit thresholds, or re-evaluate proposed improved models on these 128. Analyzing these aggregate results as a scientific postmortem is permitted, but it is **not fresh evidence of improvement**. Avoid overclaiming a particular root cause: distribution shift, overfitting of query-role semantics, and grammatical reversal are hypotheses pending new precommitted tests.

## Next independent controlled experiment proposal (not executed here)

1. On a **new research TRAIN-only** split, develop explicit graph-role symmetry (source↔destination reversal) consistency regularization for role head while original 43M decoder, S3O graph branch, S3O normalizer and independent NRA stay frozen. No S3AE EVAL128 information may affect optimizer updates or tuning.
2. Before training the new candidate, independently pre-register another **never-scored future EVAL** split with new graph topologies and at least four wholly unused grammar styles, balanced source-first/destination-first and yes/no. Separate TRAIN vs EVAL SHA commitments and generated seeds; do not touch QA/TEST. Prefer equal difficulty and predeclared paired baseline comparisons across S3O, S3Z, S3AG and next model.
3. Pin new candidate FP16 **before** any future EVAL CPU prefill. Run genuine 43M CPU source capture once and lock SHA. Predeclared threshold **0.5**, frozen NRA, zero output gate. Strictly enforce original **616 J7** digest, and report all both-role, edge, reachability, subgroup, false-positive/negative, and paired metrics without cherry-picking.
4. Before accepting any candidate, independently show real decoder + EOS improvements, physical all-in bpp ≤0.7 and paired CPU performance with no original-616 degradation. If any fails, reject without changing LIVE.

No Julia/Python training code or CI launched in S3AJ. This document preserves the negative result and sets safeguards for the next research hypothesis, rather than turning evaluation data into training data.
