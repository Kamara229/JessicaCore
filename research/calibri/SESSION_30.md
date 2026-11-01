# CALIBRI session 30 — see actual oneDNN kernels and ATen operators

## Grounded problem

Previous real pretrained Switch-base-8 experiments were isolated from
production. v29 GitHub Actions 41353364980 used same checkpoint, validation
token hashes and native BF16+MKLDNN execution on physically heterogeneous
CPUs (AMD EPYC 9V45 and Intel Xeon Platinum 8573C). Both hosted runners
executed default dispatch and DNNL_MAX_CPU_ISA=AVX2 in fresh subprocesses.
The cross-host full logits did not match exactly on WikiText even after
AVX2 cap (fixed probe max |difference| = 3.5); TinyStories matched exactly
after cap. Within the **same** physical CPU, requesting the cap also
changed numerical output (AMD WikiText max sampled |delta| 4.5; Intel 7.5).
This strongly merits an actual backend execution trace, not speculation.

## Single next experiment

Both native PyTorch and oneDNN have CPU kernel implementation choices.
The requested maximal ISA environment variable is not a guarantee that
every PyTorch CPU operation uses that ISA. This v30 experiment preserves
v29's pinned public pretrained BF16 weights, token windows, MKLDNN flag,
jitter=0 and deterministic flag and measures, separately:

1. **oneDNN primitive runtime dispatch** via ONEDNN_VERBOSE=1 and
   DNNL_VERBOSE=1. Record actual reported CPU ISA line, operation,
   implementation string and number of executed primitives for each
   fresh default/AVX2 subprocess. Strings such as
   `brg_matmul:avx512_core_bf16` are evidence of a specifically
   executed oneDNN primitive.
2. **PyTorch ATen CPU operations** via
   `torch.profiler.profile(ProfilerActivity.CPU)`, around the **same
   original full model forward**, not independent synthetic matmul.
   Record exact operator name and call count for `aten::mm`,
   `aten::addmm`, `aten::bmm`, etc. ATen operator labels identify
   the **operator**, not its compiled CPU microkernel or actual ISA.
3. **Numerical controls** reuse v29 checksum/token/parameter validation,
   24 frozen vocabulary probes per token, full-output-logits SHA256,
   full-vocab top1 IDs and exact per-token router masks. This verifies
   the profiler did not silently change the original experiment's input
   or measured output contract.

Two independent GitHub-hosted Ubuntu CPU hosts (22.04/24.04) each
perform both modes. Hardware is opportunistic; comparison is explicitly
labeled inconclusive about AMD-vs-Intel if allocated to same CPU family.
No model weight checkpoints or full raw output logits uploaded.

## Scientific interpretation safeguards

A oneDNN verbose trace with `primitive,exec` certifies only that *that*
oneDNN primitive and implementation executed. A lack of oneDNN verbose
lines in capped BF16 runs may mean PyTorch used a fallback, but DOES NOT
establish which ATen CPU ISA was used. Distinguishing real ATen assembly
would need kernel-level instrumentation/stack profiling or controlled CPU
feature masking in a future step. Do not infer that PyTorch profiler
operator names alone reveal AVX instructions. The proposed study measures
dispatch mechanisms, not a novel compression algorithm. No WikiText PPL,
physical 0.8 bpw coding, native packed inference speed or downstream
task quality is claimed. No changes to main, Render, Supabase, APK or the
user's ordinary Jessica chat.
