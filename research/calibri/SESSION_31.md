# CALIBRI session 31 — PyTorch ATen dispatch selection under fixed oneDNN cap

## Actual evidence motivating v31

v30 run **41354377388** passed all GitHub Actions jobs: two independently
assigned hosted CPUs loaded the SAME real pretrained
`google/switch-base-8` BF16 checkpoint and validation token SHA256.
One host, Intel Xeon Platinum 8370C, executed **1706 oneDNN
`gemm:jit:bf16`** matmul primitives in the default ISA mode; another,
AMD EPYC 9V74, executed **1706 `brg_matmul:avx512_core_bf16`**.
After **requesting** DNNL_MAX_CPU_ISA=AVX2, zero oneDNN primitives were
logged on both hosts, yet PyTorch ATen CPU profiling still reported
`aten::mm`, `aten::bmm` and `aten::linear` on the actual pretrained
forward. Cross-host logits on WikiText still differed in sampled values
by up to 3.5, despite the same requested oneDNN maximum ISA. That is
NOT a proof that ATen fallback executed AVX2 machine code: the oneDNN
cap does not cover other dispatchers, and a profiler operator name
cannot disclose the assembly microkernel.

## One isolated next experiment

For each of two independent hosted Linux CPU runners, execute two
**fresh Python processes on the SAME host**, loading the same pinned
public pretrained Switch BF16 weights + WikiText2/TinyStories corpus
tokens. Set and hold CONSTANT throughout:

- DNNL_MAX_CPU_ISA=AVX2 and ONEDNN_MAX_CPU_ISA=AVX2;
- model weights and source manifest, BF16 dtype and MKLDNN enabled;
- same seed, two threads, torch deterministic=True, zero router jitter
  *before* the first forward, same validation token windows.

Change only `ATEN_CPU_CAPABILITY` set BEFORE importing PyTorch:
1. `ATEN_CPU_CAPABILITY=default`: PyTorch CPU DispatchStub DEFAULT.
2. `ATEN_CPU_CAPABILITY=avx2`: PyTorch CPU DispatchStub AVX2.

Require the observable PyTorch internal
`torch._C._get_cpu_capability()` to **actually return** the corresponding
`DEFAULT` / `AVX2` selected dispatcher value, not merely the requested
environment string. This is stronger dispatcher evidence than the
earlier oneDNN cap, but **still not proof** of all per-op instructions
or of optimized ATen/BLAS kernel assembly.

Reuse the real full model forward instrumentation from v30, retaining:
full logit SHA256, 24 frozen vocab samples per token, top1 IDs, exact
next-router one-hot masks, PyTorch ATen profiler counts and separately
oneDNN primitive implementation log. Check source weights and token
SHA256 across the four records. Compare each CPU's DEFAULT vs AVX2
outputs, then cross-host outputs for the two selected modes.

## Interpretation gates

- If both modes on a host are exactly equal, there is no detected
  model output effect for this PyTorch dispatcher setting under the
  fixed oneDNN cap and finite evaluation windows. Not a universal proof.
- If they differ, identify affected logits/router dispatch per corpus;
  examine operator counts and follow up with kernel-level instrumentation.
- If cross-host selected AVX2 hashes still differ, the selected ATen
  dispatcher setting is not sufficient for full reproducibility.
- If hosted CPU models coincide, cross-host comparisons cannot establish
  heterogeneous CPU behavior.

Do NOT claim direct instruction tracing from
`torch._C._get_cpu_capability`: it describes DispatchStub's selected
CPU capability, not each actual emitted or executed machine
instruction. Numerical max absolute errors apply only to fixed 24
vocabulary probes, not all logits. There is NO new physical compression,
WikiText PPL, 0.8 bpw result, native packed speed/ARM speed,
model training, or user-side ordinary Jessica chat test in this task.
No main, Render, Supabase, APK, production data or credentials touched.
