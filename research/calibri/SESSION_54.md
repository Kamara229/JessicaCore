# CALIBRI v54 — four-family, eight-fold jackknife stability test (predeclared)

**Next single stage, research-only**, continuing measured v53 PR #183, run 41489153443 (3/3 GitHub Actions PASS). All pretrained Pythia14m full-source INT6 CB46 storage is 10,803,661 actual bytes. PRE WikiText-2 TRAIN-A [0:2048] remains fixed; deterministic signed INT6 2D code and original 1D state preserved. The three original-weight FP16 residual-row corrections are FROZEN physical CR47 sidecars and must be bit-identical to v51-v53: 6%-per-tensor 610,751 B (2056 rows), 12% 610,715 B (2050), 24% 610,875 B (2162); parent CB46 hash-bound with CRC/SHA. Actual complete model + sidecar + reserved 15,480 future affine bytes <=6.50 bits per ALL original pretrained parameters. **Each runtime model reconstructs FP32 active RAM once on cold start: no claim of native packed INT6 GEMM or RAM compression**. The user's hard runtime speed limit is <=+20% on real warm prefill+decode.

## Why change the selection gate, not the stored correction?

v53 scored two disjoint 2048-token TRAIN windows in each of three corpus families, which correctly prevents one source family from dominating many correlated windows. Yet candidate selection could still hinge on *which* of the two excerpts happened to be drawn from a family. v54 adds a new genuinely fourth TRAIN family, plus **leave-one-training-window-out stability**. Freeze the gate math, cutoffs, sample indexes, and candidate files before examining any new heldout data. All baseline/patch/candidate source bytes and SHA must remain identical to v53.

**Eight fixed training windows, four source families, all unseen in prior experiments:**
- WikiText-2 TRAIN tokenizer positions `[12288:14336]`, `[14336:16384]` (two 2048-token windows; PRE TRAIN-A is [0:2048]).
- AGNews TRAIN **record** indices `[1280:1408]`, `[1408:1536]`, first 2048 model tokens of each rendered concatenation.
- SciQ TRAIN record indices `[256:384]`, `[384:512]`, disjoint from v53 SciQ TRAIN[:256], rendered Support + Question + four unordered answers, no answer labels.
- SuperGLUE BoolQ TRAIN record indices `[0:128]`, `[128:256]`, never used for INT6 correction selection in previous CALIBRI sessions; rendered Passage+Question, omit boolean answer label. Distinct from prior BoolQ VALIDATION tested in v53.

## Frozen four-family and jackknife decision (TRAIN only)

For each physical INT6 variant `base_int6_pre`, `sse_cap_06pct`, `sse_cap_12pct`, `sse_cap_24pct`, evaluate full real pretrained Pythia loss on all eight source-only TRAIN excerpts. The NLL gain for a window is `NLL_PRE - NLL_candidate`. Compute two-window mean for each family, **macro mean across FOUR equally weighted families**, and 4 leave-one-family-out *three-family* mean gains. Eligibility requires all of:
- at least **3 of 4** family means >0;
- macro family NLL improvement **>=0.0005** nat/token;
- worst individual family mean **>=-0.0080** nat/token;
- worst leave-one-family-out three-family mean **>=0.0005** nat/token.

Then perform EIGHT paired leave-one-window-out jackknife challenges: omit each TRAIN window in turn and recompute all 4 criteria using just one window for the impacted family (two for other three). Must pass **>=6 of 8** folds for eligibility. This is a **heuristic sensitivity analysis**, not a calibrated confidence interval and no guarantee of generalization. Among eligible choose candidate by (highest number of successful folds, highest worst family-left-out NLL gain, highest macro NLL gain, smallest correction concentration 6% before 12% before 24%). If none eligible choose raw 6.1438056bpw PRE baseline. All tie breaks fixed. Neither the training gate nor physical candidate selection may be modified using post-gate test evidence.

## Blind new external heldout (first read AFTER gate selection)

1. HellaSwag `Rowan/hellaswag` VALIDATION[:128] — a new heldout data family, rendered `ctx` and `endings`, first 2048 Pythia tokens; do not use gold correct-ending labels.
2. `allenai/ai2_arc` ARC-Easy VALIDATION[:128] — disjoint task configuration from previous ARC-Challenge, rendered question + options without gold answer, 2048 tokens.
3. Yelp Review Full TEST[640:704] — new disjoint records, 2048 pretrained model tokens, no sentiment labels.

PPL for original pretrained FP32 teacher, plain PRE INT6 and **ALL** three physical correction variants, not merely chosen. Keep every negative PPL result. These are fixed short PPL excerpts, **NOT actual QA task accuracy, not a representative full evaluation**. Do not promote/retrain based on revealed heldout. Source revisions and SHA256 for all eight TRAIN plus three heldout excerpts persisted for audit.

**Speed**: true original Pythia CPU batch1 128-token prefill +32 KV-cache decode, 2 warmups, 7 actual timed repetitions, two pinned AVX2 2-thread Ubuntu22.04 and 24.04 CPU hosts. Third independent GitHub checker MUST rederive every family gain, four LOFO means, eight jackknife outcomes, training-only chosen variant, all files/bytes/SHA, and external PPL/speed comparison. Physical file integrity determines CI success; a negative speed/quality outcome is scientific evidence and reported transparently. Absolute speedups under CPU jitter are not claims of faster packed compute.

No main, Render, Supabase, live Jessica chat, APK, settings, secrets, or production modifications. Research draft PR only.
