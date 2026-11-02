# CALIBRI v53 — preregistered three-family leave-one-family-out TRAIN-only gate

**One research stage after v52 PR #182**, all 3/3 GitHub Actions jobs PASS (run 41487925814). The real pretrained Pythia14m full-state physical INT6 CB46 checkpoint is 10,803,661 bytes, identical on both CPU hosts. The three SOURCE-ONLY physically CRC/SHA-bound FP16 residual-row recovery sidecars (6%, 12%, 24% caps per original 2D tensor) are frozen and independently bit-for-bit identical to v52: 610,751; 610,715; 610,875 bytes respectively. All corrected full-model versions are <6.50 physical bits per parameter including 15,480B reserved future POST bytes. Full-model warm CPU prefill128+KVdecode32 speed penalty <=20% in v52. Model is *NOT 6-bit at runtime memory*; complete INT6 source expands into dense FP32 hot cache at cold startup and physical patches apply once.

**Known failure mode:** six-window v52 selector gives four AGNews training windows vs just two WikiText windows, so the apparent 4/6 majority can be dominated by one corpus family. Simply adding more correlated snippets must NOT be called increased independent evidence. This v53 changes ONLY the TRAIN-only selection rule and new TRAIN corpus family, keeps same frozen source-only candidate physical bytes, rate cap and speed test.

## New genuine third TRAIN corpus family, predeclared before test

Use EXACTLY TWO new disjoint 2048-token training windows for each of THREE dataset families:
- WikiText-2 TRAIN token positions **[8192:10240]** and **[10240:12288]**, disjoint from all earlier WikiText PRE TRAIN-A/B/C/D and validation.
- AGNews TRAIN record ranges **[1024:1152]** and **[1152:1280]**, disjoint from all prior AGNews TRAIN[:1024].
- SciQ TRAIN record ranges **[0:128]** and **[128:256]**, FIRST SciQ in this CALIBRI experiment family; strings `Support + Question + answer options`, correct answer not marked, with Pythia token cap 2048. Both entirely disjoint from SciQ TEST. Pin dataset revision.

**Three-family macro gate** rather than the v52 4/6 per-window majority. Compute `gain_family = mean_windows(NLL_INT6_PRE − NLL_candidate)` separately for each of the three equally-weighted families. Only accept a candidate if:
1. At least **two of three family means** are strictly positive.
2. Equal-family macro NLL improvement ≥ **0.0005** nat/token.
3. Worst family mean NLL improvement ≥ **−0.0080**.
4. Worst pair of remaining family means when **each one family is omitted** ≥ **+0.0001**.

Among eligible, choose maximum worst leave-one-family-out mean, tie-break to lower cap % (6% before 12% before 24%); if none eligible choose plain INT6 PRE with zero sidecar. Fixed candidate order exactly `base_int6_pre, sse_cap_06pct, sse_cap_12pct, sse_cap_24pct`. This is a transparent HEURISTIC for robustness, **NOT** a calibrated statistical confidence interval. Neither the threshold nor correction candidates may be adjusted after evaluating any test. Both CPUs MUST make the identical one-shot decision; the independent comparator MUST recompute the complete family and LOFO arithmetic and hashes.

## Fresh first-use heldout QA and news families

Read ONLY after all physical source/candidates stored and frozen training gate decision made:
- **SciQ TEST[:128]**: separate from SciQ TRAIN windows, rendered as support/question/answer candidates but **no correct-answer label**, 2048 Pythia tokens. This is same *family* as third TRAIN input but separated records.
- **SuperGLUE BoolQ validation[:128]**: a genuinely new family not in any previous CALIBRI TRAIN/TARGET, rendered passage/question, 2048 Pythia tokens, answer labels omitted.
- **AGNews TEST[:128]**: separate heldout split from all AGNews TRAIN snippets, 2048 Pythia tokens, class labels omitted.

Evaluate true full-model causal language PPL for original pretrained FP32 teacher, plain PRE INT6 and ALL three capped physical FP16 correction candidates (even rejected). Do not call this classification or answer accuracy, or extrapolate 2048-token PPL to production Jessica. Observe and preserve every regression. Data source revisions and all token digests are stored; the independent third verifier checks all.

**Warm inferencing:** original vs TRAIN-selected truly physical complete INT6 model hot loaded from CB46+CR47, batch 1, fixed 128-token genuine TRAIN-A prompt + 32 KV-cache autoregressive steps, 2 warmups and 7 repeats each host (Ubuntu22.04/24.04, AVX2, 2 threads). User hard <=1.20 runtime latency ratio; quality research aspiration provisional <=5% teacher PPL not user-fixed. Real active FP32 RAM not reduced by at-rest INT6. All storage bytes/CRC/SHA and 15,480B correction reserve counted; no hidden teacher in fresh runtime.

Research-only draft PR: **do not modify production Jessica, ordinary chat, main branch, Render, Supabase, APK, settings, secrets or users' data**. End stage with actual 3-job verified outcomes and explicit positive/negative selection.
