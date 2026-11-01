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
