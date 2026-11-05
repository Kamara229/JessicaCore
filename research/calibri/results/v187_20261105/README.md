# CALIBRI v187 — completed genuine four-axis source pilot (2026-11-05)

**Engineering pilot EXECUTED; end-user model quality NOT TESTED.**
A novel 4D `[rowpair,columnblock,2,4]` source representation
is physically valid and natively executable on CPU, but the
measured stock direct C implementation is substantially slower
than original CB59 for the **eight chosen pretrained matrix
operators**, and source reconstruction error improves only
marginally overall.

[GitHub Actions #42087645964](https://github.com/Kamara229/JessicaCore/actions/runs/42087645964):
final SUCCESS on 3 targeted tests plus 8 actual original
pretrained SmolLM2-135M matrices, AFTER initial
[#42087239118](https://github.com/Kamara229/JessicaCore/actions/runs/42087239118)
FAIL due missing empty-source guard in the unit tests (fixed,
no real-model results observed beforehand).
[Complete real-source raw outputs artifact #13444451062](https://github.com/Kamara229/JessicaCore/actions/runs/42087645964/artifacts/13444451062)
contains frozen protocol, all eight operator details, native
numerical error and all paired timings. Research draft PR #325.

## Actual measured hypothesis

Reshape original pretrained FP32 2D matrix into four axes,
`[output row pairs, input column blocks, 2, 4]`; one
K32/5-bit code now represents BOTH adjacent original
output rows' weights for the same four input channels.
This is a **different real weight representation** from
baseline CB59's one K32/5bit code for eight contiguous
input channels of a SINGLE output row.

Every tested tensor has literally the same number of
5-bit packed code bytes, FP16 per-original-output-row
scales, and FP16 32x8 codebook bytes as its original
CB59 1x8 compressed counterpart. No residual, hidden
dense trained source or unpaid factor. Direct native C
GEMV and demand embedding agree with explicitly decoded
2x4 representations in numerical unit tests.

The *actual fully priced 134,515,008 original-weight model
archive* was NOT built or measured; no 4D whole-model
generation was attempted; no private independent TEST
was consulted. This is a PRE-REGISTERED real original
weight source/operator pilot, NOT a finished 0.700bpp
model nor a usable QA result.

## All eight preselected original matrices

Ratio <1 indicates smaller relative FP32 reconstructed-weight
error with 4D. Native speed ratio <1 indicates 4D codebook
kernel slower than the original direct K32/V8 CB59 source,
at matched single-operator input/geometry.

| Original trained matrix | Relative 4D weight error / original CB59 | Measured 4D native GEMV speed / CB59 |
|---|---:|---:|
| Shared input/output embedding | 0.980109 | 84.30% |
| Block 0 attention output | 1.017933 | 72.34% |
| Block 0 MLP gate | 0.999298 | 76.03% |
| Block 0 MLP down | 1.000151 | 71.26% |
| Block 15 attention output | 0.995823 | 71.35% |
| Block 15 MLP down | 0.998822 | 72.58% |
| Block 29 attention output | 0.996546 | 71.36% |
| Block 29 MLP down | 0.999403 | 71.17% |

- **6/8** matrices have modestly smaller source weight error;
  **2/8** become worse.
- Median per-matrix 4D/original source relative weight L2
  ratio: **0.9990602**; ~0.094% smaller median error,
  TOO SMALL to substantiate recovered model function.
- Median of eight **SINGLE-OPERATOR** native GEMV speed
  ratios: **71.85%**, below the prospective 90% native
  whole-model CPU goal, but this is NOT itself measured
  whole-model generation throughput.
- Early attention O source error even rises by **1.79%**,
  despite matched physical code bytes.

**Scientific decision:** Do not promote 2x4 retiling or
claim that it solved compressed LLM quality. It is a real
multidimensional representation with exact source/CPU
implementation, but this unchanged codebook / naive
2x4 LUT design has negligible error benefit and
material operator-speed cost. An optimized multi-axis
family would need stronger ORIGINAL-FP32 behavior
benefit and throughput, not merely source RMSE, before
a full physical runnable model is warranted.

The concurrent v186 actual 5-bit behavioral source
training showed only 3/12 held-out opposite-answer
pairs and 2/24 machine-confirmed historical DEV QA,
with arithmetic still repeated `555...`; it remains
unusable. The next research decision should prioritize
the original FP32 model's demonstrated *real answer
capability* on a larger, independently varied public
question set before spending further effort mimicking
very weak teacher outputs under a tiny <0.7bpp budget.
A larger original model may ultimately be necessary
to test meaningful correctness preservation; that
is a hypothesis, not a measured CALIBRI success.

No NLL/KL/PPL acceptance, no unseen v89 private TEST,
no main, LIVE Jessica, Render, Supabase, APK or Green
Tunnel changes. All experiments ChatGPT in GitHub
research branches only; Jessica ran NO tests.
