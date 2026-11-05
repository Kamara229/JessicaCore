# CALIBRI v184 — completed physically paid shared rank-2 QAT (5 Nov 2026)

**Engineering success: a complete new, native CPU executable model at
0.694314 total bits per original parameter. Scientific real-answer quality:
FAIL.** This model cannot be promoted to CALIBRI/Jessica.

The test is [GitHub Actions run 42073916008](https://github.com/Kamara229/JessicaCore/actions/runs/42073916008),
conclusion **SUCCESS**; numerical checks, 16 actual whole-model QAT
steps, source packing, SHA verification, 20 first-token probes, 24
real prompt-answer generation pairs per arm and 8 CPU timing pairs
all completed. [Raw artifact 13437224865](https://github.com/Kamara229/JessicaCore/actions/runs/42073916008/artifacts/13437224865)
contains COMPLETE physical pretrained source, frozen protocol and
all actual generated texts/token IDs, grader outputs and paired
native timings. Research-only draft PR [#322](https://github.com/Kamara229/JessicaCore/pull/322).

## Precise physical source

- Immutable original CB59 (all 211 matrices K32/V8 packed, FP32
  1D tensors, original tied head, full metadata+CRC/SHA):
  11,229,303 bytes, SHA256
  `50eac24ba7d7a452c495f8adced2cdef3f9cc44e024217fc21d8f2cebb87413c`.
- A source: each matrix has signed INT8 rank-2 output-row gains:
  **409,344 bytes** across 211 unique trained original 2D matrices.
- A gains: physically paid individual FP32 scales for all 211 matrices:
  **844 bytes**.
- Shared B source: TWO rank-2 matrices (input widths 576 and 1536),
  cross-layer FP16 and physically shared: **8,448 bytes**.
- New outer metadata/header/CRC/SHA: **26,513 bytes**. No
  unaccounted dense, private answer-model correction, or adapter.
- **Full exact source file: 11,674,452 bytes = 0.6943137229713431
  bits / 134,515,008 unique original pretrained parameters.**
  Required upper cap: 11,770,063 bytes (0.700 bpp),
  remaining margin **95,611 bytes**.
- Sealed full source SHA256:
  `7321a8a857896ae961c801af5fc9d7c4e29f0d66fcca8306f747eda7855b0056`.
- **405,578 nonzero signed INT8 A entries** of 409,344:
  the source actually changed during training. The
  input embedding/output head share the same source correction.

All 211 corrected 2D matrices execute directly from packed CB59
codebook source and new rank-2 factors through the new native C
GEMV/embedding routines, not a hidden full FP32 pretrained model.
The exact TRAIN surrogate vs physically saved CPU native source
logits differ only **7.0e-7 relative**, with the same top token.

## Pre-registered actual quality, NOT source RMSE or training loss

| Compared actual model | Original FP32 | Original CB59 | v184 shared-rank2 |
| --- | ---: | ---: | ---: |
| Complete source bpp | 32 | .667839 | **.694314** |
| Original FP32 first-token matches on 20 historic DEV | 20/20 | 0/20 | **0/20** |
| Unique first-token IDs on these 20 | 10 | 3 | **1** |
| Machine-confirmed correct responses, 24 historic DEV | 6/24 | 0/24 | **0/24** |
| Reached 32-token cap without EOS | 4/24 | 24/24 | **24/24** |
| Mean distinct fraction of emitted tokens | .89476 | .05078 | **.05859** |
| Median candidate CPU speed versus native CB59 (8 paired fixed 32-token runs) | -- | 100% | **96.63%** |

FP32 original 6/24 includes only 15 parseable machine grades;
9 are conservatively UNGRADED (not necessarily wrong). Same 24
prompts and fixed cap for all arms; these are historical development
prompts, NOT an independent blind TEST. No claim of 0/96 or of a
universal performance score.

Actual answer examples, pulled from the same archived raw answers:
- Public DEV `Calculate 7 plus 3.`, original FP32: `7 + 3 = 10`
  with EOS.
- Native CB59: repeated unrelated `speci speci speci...`.
- Native CB184: `ate1111111unsqueeze11111111unsqueezeateate...`,
  then repeated irrelevant fragments with no EOS.

**The new rank-2 sidecar preserved physical bytes and speed but did
not restore coherent language, let alone original FP32 ability.**
Do not claim original QA parity merely because training gradients,
source SHA or native GEMV were correct.

The supervised TRAIN optimization did execute 16 steps with
straight-through INT8/FP16 rounding on frozen public training
data and original model teacher targets. Its displayed TRAIN
cross-entropies e.g. step 4: 12.329, step 8: 15.154,
step 12: 18.265, step 16: 7.857 are for DIFFERENT samples and
cannot be interpreted as a monotonic training curve or validated
accuracy. TRAIN loss was never an acceptance gate.

## Scientific next step

The failure is not kernel correctness, raw byte cost or CPU speed.
The initial K32/V8 5-bit assignment irreversibly destroys too much
coordinated pretrained behavior, and a **rank-2 across-layer shared
residual with 16 fixed training steps cannot repair it**.

The next representation experiment must change the **discrete
source itself**: for example jointly train/reassign the actual
5-bit code indices with learnable dictionaries in a tightly
budgeted differentiable or alternating optimization, or train a
new native low-bit model architecture directly. Unlike v176,
which kept code indices fixed while learning codebooks/scales,
this would alter the compressed codes, without increasing their
five-bit physical cost. Such an approach remains a **hypothesis**
until complete binary file, real QA and native CPU speed are
measured; no success is assumed and source code must not leak
the original FP32 model.

No NLL/KL/PPL acceptance. No private v89 TEST used. Keep
research branch/draft PR unmerged. No LIVE Jessica, main,
Render, Supabase, APK or Green Tunnel edits. Jessica is
relay only, never asked to run tests.
