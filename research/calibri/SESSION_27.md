# CALIBRI session 27 — isolate BF16 arithmetic from cross-host variation

## Motivation and previous verified negative evidence

GitHub Actions 41350732476 (session 26, PR #154) loaded the SAME pinned real
`google/switch-base-8` model and validation-token hashes on two independent
Ubuntu 22.04 and Ubuntu 24.04 x86_64 runners. Both jobs passed within-process
three-repeat invariants, weight-loading checks and exact restoration; the
**between-host exact-gate FAILED**. Numerous original/unmodified teacher
logits SHA256 mismatched and next-router exact masks sometimes differed.
Therefore a small apparent v24 KL gain is not yet established as reproducible.
It cannot be attributed to compression without first separating CPU execution
variance from reconstruction effects. The ordinary Jessica chat's 90s timeout
is another failure and does not count as a research experiment.

## Predeclared controlled experiment

Same immutable public BF16 pretrained checkpoint SHA
`92fe2d22b024d9937146fe097ba3d3a7ba146e1b`, corpus revisions and
token hashes as sessions 25/26. Two independent fresh GitHub hosted runners
(ubuntu-22.04, ubuntu-24.04), pinned PyTorch 2.5.1 CPU and Transformers
4.46.3. No model training, QMoE 0.8 bpw replication or quantized speed claim.

For every validation corpus (WikiText-2 and TinyStories), with explicit
jitter=0, model.eval(), two PyTorch threads, deterministic algorithm option:

- **Native BF16** checkpoint, MKLDNN enabled, to reproduce v26 conditions.
- **FP32 arithmetic reference** using *the same originally loaded BF16
  parameter values converted to FP32*, MKLDNN disabled. Not a separate
  independently sourced FP32 teacher; differences within a host include the
  deliberately changed compute dtype **and** MKLDNN, and this confounding is
  openly declared.

Read real entire model logits only in runner memory; upload no weights and
no full-vocabulary logit tensor. Record full-logit hashes, full-vocab top-1
predictions, full-logit per-token L2 norms, 24 fixed evenly spaced vocabulary
probes per token, and **exact 8-bit router masks per token**, plus CPU
capability/model and elapsed-time diagnostic (not native/packed benchmark).
Keep validation windows and full-pretrained source hashes identical between
hosts. Cross-compare numerical probes: max absolute, relative and RMSE; top-1
agreement; exact per-token next-router agreement; full-logit SHA equality.

## Evaluation gates and limits

CI **fails** on missing checkpoints, token source differences, incomplete
loading, nonfinite/shape-incompatible logit probes or missing evidence. CI
**reports, rather than hides**, numeric differences as research findings:
BF16/FP32 exact cross-host equality is recorded separately in JSON flags.
No claims that non-bitwise equality necessarily means inaccurate output; a
small numeric drift may be benign, while routed-expert changes can magnify
it. No full-vocabulary max-abs logit difference is available (only predeclared
24-vocab probes); do not mislabel it as such. All model weights remain
outside uploaded artifacts; only limited diagnostic metrics included.

## Decision

If FP32+MKLDNN-off is stable but BF16+MKLDNN-on differs, investigate the
hardware-dependent BF16/oneDNN execution path with a follow-up fixed-dtype
MKLDNN on/off factorial test or pinned identical container. If both differ,
prioritize source reconstruction/host nondeterminism and static CPU features.
If cross-host hashes match, proceed with rate-distortion and exact-size
controls for real weights and then separate native packed ARM throughput
verification. No merger into main or Render/Supabase/APK mutation here.


## Actual GitHub Actions result — 41351926001 (SUCCESS, two CPU runners)

The pinned real pretrained Switch experiment executed on **both**
Ubuntu 22.04 and Ubuntu 24.04. The source/checkpoint/token/parameter
invariants matched. **Cross-host exact full-logit SHA256 matched for
WikiText2 and TinyStories, BOTH for BF16+MKLDNN and for FP32+MKLDNN-off.**
The 24-probe per-token cross-host maximum absolute difference was **0**
and exact next-router mask agreement was **100%** on both corpora in both
modes. This contrasts with v26's negative cross-run finding and shows
that cross-host drift is **not universal** for these pinned weights.
This is a reproducibility control, not validation of v24 precision claims.

Within one Ubuntu 24.04 process, switching from BF16+MKLDNN to FP32
arithmetic+MKLDNN-off changed logits substantially: sampled maximum
absolute delta **7.563 (WikiText)** and **7.208 (TinyStories)**;
sampled logit RMSE **0.688 / 0.692**; full-vocabulary token top-1
agreement **92.19% / 95.31%**; next-router exact-mask agreement
**100% / 98.44%** respectively. These are **two deliberately changed
execution factors**, and must NOT be described as BF16 quantization error
alone, since MKLDNN also changed. They are not PPL or speed results.

The v27 run enabled `torch.use_deterministic_algorithms(True)` and
forced router jitter to zero **before** evaluation; v25/v26 used the
default deterministic flag and only zeroed jitter after an initial
original forward. Next hypothesis: which control, if any, explains the
v26 drift? A single-factor **deterministic-algorithms on/off** comparison
at unchanged BF16+MKLDNN and fixed zero jitter is the next justified
experiment, with cross-run exact hashes and sampled numeric/router
deltas. Do not infer causality from the v27/v26 juxtaposition.
