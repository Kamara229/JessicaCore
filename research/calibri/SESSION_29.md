# CALIBRI session 29 — oneDNN AVX2 ceiling as a controlled factor

## Verified rationale

Session 27 (GitHub Actions 41351926001) replayed real pinned
`google/switch-base-8` BF16 and a separate FP32 arithmetic control on
WikiText2/TinyStories; source hashes and full logits were bitwise identical
across two particular Linux hosts. Comparing BF16+MKLDNN with
FP32+MKLDNN-disabled produced larger within-host differences, but
precision and kernel backend changed simultaneously, so attribution is not valid.

Session 28's first raw host pair from run 41352373405 happened to use
**different CPU families** (AMD EPYC 7763 AVX2 vs Intel Xeon Platinum
8573C with AVX512), and observed substantial cross-host logit changes.
Within each host the deterministic PyTorch flag ON/OFF/ON did not change
outputs. That run's cross-host *comparator* initially failed for a
dependency import error, NOT for a numeric mismatch; the JSON artifacts
remained available. After repairing its dependency, a subsequent v28
run 41352717114 completed successfully on **two AMD EPYC 7763** hosts:
all forward-logit and exact router SHA256 matched with deterministic
ON, OFF and ON. This apparent contrast is consistent with, but does
NOT prove, CPU-dependent arithmetic. Hosted runner hardware is allocated
opportunistically; the OS matrix does not guarantee AMD vs Intel.

## v29 preregistered experiment

Real checkpoint revision `92fe2d22b024d9937146fe097ba3d3a7ba146e1b`,
no training and no changes to model parameters, native BF16+MKLDNN,
same seed 27, PyTorch deterministic algorithms ON, same 2 threads,
zero router jitter BEFORE first forward, same tokenizer and original
WikiText2/TinyStories validation samples. Two GitHub-hosted Linux
environments each run two independent **fresh Python processes on the
same allocated CPU host**:

1. Default oneDNN CPU ISA dispatcher, with ISA environment variables
   explicitly absent.
2. Same real BF16 checkpoint and data, with
   `DNNL_MAX_CPU_ISA=AVX2` and `ONEDNN_MAX_CPU_ISA=AVX2` set
   BEFORE the separate Python process imports Torch.

The official oneDNN dispatcher contract supports a requested maximal
ISA via `DNNL_MAX_CPU_ISA=AVX2`. It applies to supported oneDNN kernels
but **does not guarantee** all PyTorch compute kernels will use AVX2 or
prove the library's effective executed ISA. Log requested caps and
physical hardware CPU brand and flags; use oneDNN runtime verbose logs
to aid diagnosis. A requested ISA cap is not an Intel CPU emulator.

For each host/setting/corpus record source weight and token SHA256, fixed
24-vocabulary probes per token (sampled max-abs/RMSE), full-vocab logit
SHA256, full-vocab top-1 IDs, and **exact per-token eight-expert router
dispatch masks**. Verify original source checkpoints and tokens were
identical across all four sessions. Compare default/capped **within the
same CPU host** first; then compare across OS hosts under both requested
ISA settings. Explicitly flag if hosted CPUs are NOT heterogeneous and
leave the AMD-vs-Intel hypothesis inconclusive in that case.

## Decision constraints

- If caps alter real Intel outputs but not AMD, that supports a CPU/ISA
  kernel-path hypothesis; additional effective-kernel tracing needed.
- If AMD and Intel outputs match when both capped but not when uncapped,
  that supports (but does not definitively prove) differing ISA dispatch
  as an explanation of the earlier drift.
- If capped runs still differ, scrutinize other Torch ops, model CPU
  flags, oneDNN implementation versions and kernel dispatch traces.
- If both hosted runners are same CPU model, only **within-host** effects
  are tested; no claimed cross-hardware generalization.
- Any initial CI dependency/tooling error is NOT equivalent to a
  failed scientific result; preserve negative data and repair tooling.

This does NOT evaluate WikiText perplexity, rate-distortion, physical
0.8 bpw compression, QMoE reproduction or native packed throughput.
No main, Render, Supabase, APK, or ordinary Jessica-chat test changes.


## Observed measured result — GitHub Actions 41353364980 (SUCCESS)

Both real-pretrained BF16 runner jobs and their independent numerical
comparison completed, with all pinned checkpoint/weight/token/source
integrity checks passing. This time hosts were **heterogeneous**:
Ubuntu 22.04 ran on AMD EPYC 9V45 (AVX2, AVX512 BF16 available);
Ubuntu 24.04 ran on Intel Xeon Platinum 8573C (AVX2 and AVX512F,
AVX512 BF16 not advertised). The two subprocesses **within each
individual host** were paired on the same actual CPU.

The exact sample results:
- **AMD** default vs requested AVX2 cap: WikiText 24-probe max absolute
  logit difference **4.5**, full-vocab top1 agreement **95.3125%**,
  exact next-router mask agreement **100%**. TinyStories max absolute
  **3.5**, top1 **96.875%**, router **98.4375%**.
- **Intel** default vs requested AVX2 cap: WikiText max absolute
  **7.5**, top1 **92.1875%**, router **100%**. TinyStories max
  absolute **3.5**, top1 **96.875%**, router **100%**.
- **Across AMD vs Intel, default dispatch**: WikiText max absolute
  **4.0**, top1 **93.75%**, router **100%**; TinyStories max absolute
  **3.0**, top1 **96.875%**, router **98.4375%**. Full-logit hashes
  differed on both corpora.
- **Across AMD vs Intel, both requesting AVX2 cap**: WikiText max
  absolute **3.5** and probe RMSE **0.12649**; full-logit SHA256 still
  DIFFERED, but full-vocab top1 and next-router tokens agreed 100%.
  TinyStories logit SHA256, all probes and next-router SHA256 matched
  EXACTLY, max-absolute probe difference **0**, top1/router 100%.

The oneDNN verbose log reported baseline ISA as AVX512 with BF16 on AMD
and AVX512 on Intel, and actual accelerated/fallback matmul algorithms
differed. With the AVX2 ceiling requested, no oneDNN primitive verbose
lines were observed in those subprocess outputs. This may mean BF16
matmuls fell back outside oneDNN; the experiment **cannot prove** which
effective per-op kernels computed the capped logits. It does demonstrate
an actual numerical effect of the oneDNN ISA environment cap request
within the same hardware, and **partial** convergence across different
CPU families, NOT universal equivalence.

**Scientific verdict:** the AVX2 ceiling did not make all cross-CPU
logits bitwise reproducible. One corpus matched, the other did not.
Do not claim the ISA alone caused all previous drift or that a
calibrated BF16 model is now portable. Follow-up should capture
per-op dispatch with independently verified actual ISA, and use an
explicit FP32 CPU reference as a control. No PPL, compressed bytes,
native packed throughput, or 0.8-bpw quality claim in this session.
Only the isolated research branch/PR #158 changed.
