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
