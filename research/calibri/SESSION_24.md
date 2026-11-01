# CALIBRI session 24 — routing-aware mixed precision for a REAL MoE

## Scientific motivation

Session 23 first proved actual pretrained \`google/switch-base-8\` expert
dispatch counts matched the Switch router exactly, not synthetic expert
activations. On decoder block 1, all 8 expert wi+wo pairs:
- FP16-equivalent physical reference 75,497,472 bytes.
- Uniform INT5 **23,654,688 B**: teacher-logit KL 0.002040 per
  validation sequence; teacher top-1 agreement 96.09%.
- Uniform INT6 **28,373,280 B**: teacher-logit KL 0.0007563 per
  sequence, teacher top-1 agreement 98.44%.
- True routed expert paired-output relative errors differ by >60×:
  expert 1 (INT5) train MSE 0.00011852 vs expert 4 0.00755495.
- A geometric TT2+packed INT5 pair was **slightly larger** than plain
  INT5 and did not materially improve expert output error for two
  frequent experts. DO NOT claim it helps this true pretrained MoE.

The next hypothesis is a joint **allocation**, not inventing a
slightly different codec. Optimize global physical bytes under
coverage and rare-expert error guarantees, keeping separate train and
validation routing. Expert-sensitive mixed precision is EXISTING
PRIOR ART; a useful CALIBRI result would need to beat strong
comparable baselines in end-to-end quality, full bytes and runtime.

## Pre-declared allocation rules (zero selection from evaluation)

Load immutable v23 TRAIN-only routed-matrix MSEs and physical byte
counts. Source GitHub Actions:
https://github.com/Kamara229/JessicaCore/actions/runs/41339418910
Metric JSON SHA256:
\`ecbc5daeceeb71d1fe3ab1d78dbce6807acce61f16253d28e3db4d04786e6f27\`.
All 8 expert wi+wo pairs are independent binary options INT5 / INT6.

Exhaustively check **2^8=256 allocations**. Minimize the
TRAIN-only routed-frequency-weighted local relative expert-output
MSE surrogate. This quantity is NOT exact global task error:
different expert output powers and nonlinear propagations are not
captured by relative MSE and frequency weighting alone.

Budget ceilings:
- 90% of physical all-INT6 size (25,535,952 B).
- 95% of physical all-INT6 size (26,954,616 B).

Rare-expert fail-closed gate: per expert *TRAIN* local relative
pair-output MSE ≤ 0.005. The 90% budget is deliberately infeasible
under this strict cap; optimizer must say NO_FEASIBLE instead of
lowering quality. The 95% budget is feasible.

Portfolio controls: all INT5, all INT6, 90% train-error-optimal,
90% popularity-only, 90% fixed expert order, and 95% guarded optimal.
The three 90% portfolios have exactly the same physical bytes.
Test independent validation routed local MSE (previously recorded
in v23 but NOT used to select allocation).

Now actually load the pinned pretrained BF16 Switch checkpoint and
temporarily substitute one decoder MoE layer's **all eight** wi+wo
pairs per portfolio; measure on WikiText2 validation and TinyStories
validation:
- Whole-model teacher next-token logit KL.
- Teacher top-1 token agreement.
- Actual next-layer router capacity-mask agreement.

These are pseudo teacher-forced text contexts, not Switch's original
span corruption task; no external downstream task quality claims.
The model executes **restored BF16 physical quantized weights** for
these tests, not a packed MoE kernel. No QMoE checkpoint or
claimed 0.8bpw quality. One layer only.

No production, main, Render, Supabase, APK or phone modifications.

## Measured actual pretrained MoE results (Actions 41340130094 — SUCCESS)

All six portfolios computed from the immutable v23 *TRAIN-only* routed
errors; exhaustive 2^8 search checked 256 allocations. Model revision
`92fe2d22b024d9937146fe097ba3d3a7ba146e1b`. Exactly one decoder
MoE layer's eight wi+wo pairs changed temporarily; real later decoder
router agreement recorded. Each corpus had 256 source tokens (4×64
encoder, 4×32 pseudo-teacher-forced decoder contexts).

**Budget constraints**:
- 90% of all-INT6 storage (25,535,952 bytes): 93 feasible portfolios
  *without* rare-expert cap. Under TRAIN per-expert MSE cap 0.005:
  **ZERO** feasible allocations, correctly returned NO_FEASIBLE.
- 95% of all-INT6 (26,954,616 bytes): 5 portfolios satisfy the same
  rare-expert cap; chosen one uses 26,603,808 bytes.

| Variant (paired expert wi+wo) | Physical bytes | WikiText teacher KL/seq | TinyStories teacher KL/seq |
| --- | ---: | ---: | ---: |
| uniform INT5 | 23,654,688 | 0.0027174 | 0.0027009 |
| uniform INT6 | 28,373,280 | **0.0007451** | **0.0009117** |
| 90% TRAIN-routed best allocation | 25,424,160 | 0.0026211 | 0.0021567 |
| 90% popularity-only (same bytes) | 25,424,160 | 0.0028678 | **0.0013855** |
| 90% fixed expert order (same bytes) | 25,424,160 | 0.0027576 | 0.0014353 |
| 95% rare-guarded allocation | 26,603,808 | 0.0017113 | 0.0016123 |

90% routed-error best assignment was expert precision
**[6,5,5,6,6,5,5,5]**. 95% guarded assignment was
**[6,5,5,6,6,6,6,5]**. Both are selected *without access* to validation
teacher logits.

**Critical scientific result**: the portfolio with lowest routed
TRAIN expert-local MSE does **not** reliably minimize global teacher
KL on a shifted corpus. At the EXACT SAME BYTES, 90% popularity-only
was substantially better on TinyStories (0.0013855 vs routed optimizer
0.0021567), despite being worse on WikiText. Simple routing-aware
weighting is insufficient. The 95% guarded choice is smaller than
INT6 but has >2x its WikiText teacher KL; global task impact matters.

**Measurement correction (2026-11-01):** the original v24 run
computed elementwise equality of all eight bits in the next-router
one-hot mask. This artificially inflates what could be mistaken for
token-level routing agreement. The previously quoted 99.6–100%
figures were *bitwise mask similarity*, **NOT exact expert agreement
per token**. v24 code has now been corrected to report both exact
per-token whole-mask equality and separate diagnostic bitwise
equality. The corrected CI run supersedes router-agreement values
from run 41340130094. Top-1 token prediction agreement (96–100%)
is a different metric and was calculated correctly.

This confirms genuine complexity in joint MoE allocation (cross-expert
nonlinear effects, distribution shift), but the technique itself is
expert-aware mixed precision, well-established prior art. **No
demonstrated quality/size/latency Pareto breakthrough**.

CAUTION: uniform INT5 KL on v24 WikiText (0.0027174) differs from v23's
0.0020403 despite pinned model; those are *separate runs* and should
not be numerically compared without investigating CPU/BF16 operation
variance and exact sample/logit hashing. All within-v24 table values
were measured in a single run under matched controls.

## Next experimental decision

1. Do NOT use routed-output MSE alone as a model-level quality proxy.
2. Add TRAIN-only whole-model teacher-logit and next-router penalties to
   select or reject precision assignments; keep rare-token hard gate.
3. Compare against a truly untouched, larger, task-appropriate Switch
   evaluation corpus; explicitly separate source dataset and decoder
   span-corruption objective from artificial teacher-forced context.
4. Geometry: earlier v23 TT+INT5 was larger and not more accurate than
   plain INT5 for two real experts. Stop assuming shared geometric
   weight fields are useful until a functional-neighborhood graph
   measured on common routed activations supports it.
5. Android packed inference and QMoE checkpoint reproduction remain
   separate unverified gates.

Results: https://github.com/Kamara229/JessicaCore/actions/runs/41340130094
No production changes.
