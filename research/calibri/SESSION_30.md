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


## Measured hardware and dispatch from first run 41354038653

Both independently hosted real-pretrained, source-matched BF16 runner jobs
completed successfully. The v29-style numerical check passed. Initial
v30 final gate marked one missing DEFAULT oneDNN primitive as an error;
that was **a mistaken assertion**, not a failed real-model run:
a CPU may fall back from oneDNN BF16 without any primitive logged.
This report gate was corrected in code (79ace349); pending confirmation
by fresh CI. All original collected evidence was still uploaded.

Hardware was heterogeneous **within AMD**, not AMD versus Intel:
- Ubuntu 22.04: AMD EPYC **7763** (AVX2; no AVX512 BF16 reported).
  **Zero** oneDNN primitive executions logged both DEFAULT and requested
  AVX2 cap. PyTorch CPU profiler still captured **730 aten::mm**,
  **144 aten::bmm**, **874 aten::matmul**, and **730 aten::linear**
  per WikiText window set, in both settings. Whole-output logits and
  router masks were bitwise identical within this host across the cap.
- Ubuntu 24.04: AMD EPYC **9V45** (AVX512 BF16 available).
  DEFAULT oneDNN log reported runtime ISA 'Intel AVX-512 with Intel
  DL Boost and bfloat16 support'; **1706 executed matmul primitives**
  with `brg_matmul:avx512_core_bf16` implementation. Requested
  AVX2-cap process reported **zero** oneDNN primitives; ATen profile
  still reported **728 aten::mm**, **144 aten::bmm** and **872
  aten::matmul** for WikiText (versus **730/144/874** default).
  The change in executed ATen call counts can arise from routing-path
  changes, not necessarily disappearance of ATen operators.

The measured SAME-HOST BF16 numerical effect of requesting the cap:
- AMD 7763: exact full logits and exact router masks unchanged on
  WikiText and TinyStories; maximum sampled logit difference = 0.
- AMD 9V45: full logits changed; WikiText 24-probe maximum absolute
  logit difference **4.5** and router agreement **100%**; TinyStories
  maximum **3.5** and exact router agreement **98.4375%**.

The cross-host outcome also remained unequal:
- DEFAULT: WikiText sampled max absolute **9.5**, router agreement
  **98.4375%**; TinyStories max **3.0**, router **98.4375%**.
- Both requesting AVX2: WikiText max **10.0**, router **98.4375%**;
  TinyStories max **1.5**, router **100%**. Neither corpus had
  byte-identical full-logit SHA256.

**Interpretation**: runtime evidence now confirms a v29 source of
observed arithmetic-path variation: one host executed oneDNN
`avx512_core_bf16` matmul primitives by default, and the capped
processes did not report those primitives. The numerical effects and
operator counts are observable. We have **not** established which
actual optimized ATen/BLAS/vector instructions ran when oneDNN emitted
no primitive logs. The requested AVX2 ceiling does not make all native
PyTorch CPU operations AVX2, and is not a portable numerical
reproducibility guarantee. This is not an improved CALIBRI compression
method, PPL or packed speed benchmark.
