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
