# CALIBRI v38 — bounded global precision allocation and new movie-review corpus

## Measured prior evidence, not hypotheses

In PR #165 / v36, ALL TWELVE real pretrained Pythia-14m MLP projections
(786,432 coefficients) were physically compressed into a 343,840-byte
CB36 bundle (3.49772 bpw including CRC, FP16 row scales, headers, outer
descriptors/SHA). At identical bits and bytes, training-output covariance
scale calibration reduced new AG News PPL from 1528.88 to 650.91, but
uncompressed teacher PPL was 235.47 and above-rate INT4 PPL 470.50.
The compressed full-model result was severely degraded.

PR #166 / v37, GitHub Actions 41359403451, used real source weights and
TRAIN-only layer NLL & row-output-SSE to allocate the exact SAME 343,840
physical bytes globally across modules. It passed cross-CPU
source/physical SHA and exact PPL replay, but at 343,840 bytes global
TRAIN-calibrated PPL was WORSE than local v36 calibration:
- WikiText DEV 529.87 global versus 443.63 local.
- Disjoint AG News test[3000:3200] 741.29 global versus 611.28 local.

Failure mechanism was diagnosed: per-byte greedy scores gave ZERO
INT4 rows to all six large (128×512) down-projections while favoring
cheap 16-byte upgrades on the six narrow-row (512×128) up-projections.
This is not evidence that all cross-layer allocation must fail; it
is a negative result for the tested unconstrained heuristic.

## Frozen v38 experimental hypothesis BEFORE any new-corpus inference

Preserve the original pretrained model, unmodified local v36 control,
its rowwise mixed INT3/INT4 codec, v35 scale calibration grid, and
exact **343,840 physical bytes** for all 12 matrices. Preserve
zero tuning on any development or evaluation PPL.

New variant: same training-only 3→4-bit row marginal-output-SSE / byte
ratio multiplied by the original model's single-matrix pure-INT3
TRAIN NLL sensitivity clipped between 0.25 and 4.0, but with
*per-matrix safeguards derived ONLY from the original local control*:
- Each module must receive **at least ceil(75% × local INT4 row count)**
  INT4 rows, selected by actual local TRAIN-output-SSE gain.
- Each module may receive **no more than floor(125% × local INT4 row
  count)** INT4 rows.
- Spend remaining global bits greedily on TRAIN-only marginal-score
  gain per byte, without exceeding those module floors/ceilings.
- Reject an incompletely filled physical budget rather than
  quietly compare unequal-size files. Baseline and candidate
  both have EXACT **343,840 real physical bytes**, counting per-row
  FP16 scales/CRC and the complete CB36 container.
- Perform fixed-scale-grid, TRAIN-only output-SSE calibration on
  each per-module row-bit assignment without changing bits or bytes.

Do not infer per-kernel native speed, global optimality, or mathematical
novelty from the heuristic. Its purpose is to test whether protection
of six starved down-projections repairs the *specific diagnosed v37
failure* without simply giving the new method more physical bytes.

## Model, source, and independent evaluation

Exactly the same real `EleutherAI/pythia-14m` checkpoint,
WikiText-2 TRAIN (256 tokens) and development validation
(1024 tokens), 6×2 source MLP modules; use only WikiText TRAIN
for layer sensitivities, captured module activations and row scales.

**First use in CALIBRI: `nyu-mll/glue`, `sst2` config,
`validation` split.** Fixed first **2048 tokenized words/subword
tokens** of the ordered movie-review sentences, from the public
validation split. This is genuinely a *different subject-domain
corpus* from WikiText and AG News, unlike the v37 AG News slice.
Pin dataset revision and token SHA; no parameter selection after
seeing SST-2 PPL. A single fixed small slice is exploratory, not
a broad confirmatory multi-dataset claim.

Compare original teacher, v36 local uncalibrated and calibrated
bundles, the new **floor-protected** global uncalibrated and calibrated
bundles, and a clearly HIGHER-RATE physical INT4 control
(401,224 bytes). Measure full-model PPL using densely decoded
real compressed MLPs while other source weights remain unchanged.

Two independent GitHub Actions hosted Linux CPUs: PyTorch
2.5.1+CPU, Transformers 4.46.3, actual ATen CPU capability AVX2
selected, oneDNN maximum AVX2 requested, deterministic algorithms
enabled, two threads. Verify identical source matrix SHA, model
revision, WikiText TRAIN/DEV & NEW SST-2 token SHA, real CB36 physical
file bytes, per-module INT4 floor/ceiling counts, full bundle
SHA256 and each run's PPL. Preserve negative comparisons and any
failure to reproduce across CPUs as separate evidence.

## Decision gate and limitations

If bounded global calibration does not BEAT local calibration at
identical bytes on new SST-2, document the negative result. If it
does, that only supports an exploratory rate–quality improvement
for one tiny pretrained Pythia model and a small first-use sample;
it does NOT prove 0.8-bit MoE compression, new multidimensional
neural architecture, model-wide physical file savings, model-wide
native speed or general applicability.

All real CB36 bundles contain ONLY 12 MLP matrices; they are
NOT a deployable full-model checkpoint. PPL uses restored dense
weights rather than packed streaming. Do not alter main,
production Jessica, Render, Supabase, APK, or user data. The
ordinary Jessica chat is NOT used for this experiment.


## CI implementation correction before seeing new-corpus PPL

Initial GitHub Actions run 41360088261 loaded the source and SST-2 but
terminated **before evaluating any variant's PPL**: bounded greedy
allocation had exactly 32 physical bytes unfilled (343,808 B vs the
frozen 343,840 B control), because short 16-byte upgrades were at
per-module ceilings and long upgrades cost 64 B.

To preserve the *precommitted* hard equal-byte and per-module 75–125%
precision constraints, the encoder now performs a deterministic
TRAIN-only byte-fill postpass if 16/32/48 bytes remain: downgrade
(64-remaining)/16 lowest-TRAIN-gain non-floor short rows, upgrade
one highest-TRAIN-gain available 64-byte long row. It keeps all
12 module floors and ceilings intact and uses NO development,
SST-2 validation, or PPL to choose the swap. The physical
rate–quality experiment therefore remains at 343,840 B, not a
loosely reported lower-budget result. Record the swap in experiment
metadata and treat this as a packer-fairness implementation fix,
not a confirmed improvement in model accuracy.


## ACTUAL results — GitHub Actions 41360326071: all three jobs PASS

Following a corrected TRAIN-only exact-byte fill (documented above),
both independent real-pretrained CPU jobs and the final source/quality
comparator succeeded. Both hosts yielded identical model, corpus
and input token SHA256, identical physical CB36 SHA256/byte sizes
for ALL tested variants, identical module row allocations and
**exactly matching PPL** on both corpora. These are measured,
not inferred. The first run failed before PPL due to 32 unused
bytes, and its failure remains documented.

### Physical 12-MLP result

All four mixed INT3/4 bundles (local uncalibrated/calibrated and
bounded global uncalibrated/calibrated) occupy **343,840 actual
bytes**, including individual signed bitstreams, per-row
FP16 scales/CRC, and the outer CB36 SHA256 container:
**3.49772135 bits per pretrained Pythia-14m MLP coefficient**,
786,432 coefficients across 12 matrices. The normal packed INT4
higher-rate control occupies **401,224 bytes** (4.08146 bpw).
The original 12 MLP coefficient FP16 reference would have
1,572,864 bytes *in theory* but is not a full checkpoint.

### Full-model validation PPL; ALL 12 MLPs replaced together

| Actual tested full-model variant | WikiText2 DEV PPL | NEW GLUE SST-2 validation PPL |
|---|---:|---:|
| Original pretrained Pythia teacher | **173.38149** | **224.16761** |
| Locally allocated 3.49772bpw, uncalibrated | **1597.66884** | **1501.97911** |
| Locally allocated, TRAIN-calibrated | **443.63432** | **570.07926** |
| Floor-protected GLOBAL allocated 3.49772bpw, uncalibrated | **1666.13674** | **1394.43692** |
| Floor-protected GLOBAL, TRAIN-calibrated | **501.85659** | **680.31779** |
| Simple rowwise INT4 HIGHER-RATE anchor | **372.07758** | **482.23761** |

**NEGATIVE result:** floor-protected global calibration is still
**WORSE** than local calibration at EXACT SAME physical 343,840B:
WikiText DEV PPL **+58.22227** (443.63 → 501.86);
first-use GLUE SST-2 PPL **+110.23854** (570.08 → 680.32).
The uncalibrated global variant happened to improve SST-2
relative to uncalibrated local (1501.98 → 1394.44), but subsequent
TRAIN calibration reversed that ranking, so the predeclared
primary CALIBRATED matched-byte comparison failed. All four
3.49772bpw variants remain drastically degraded versus source.

### Precision-floor diagnosis

The frozen 75%-125% bounds did prevent any of the six down-
projection matrices from receiving ZERO 4-bit rows (v37 failure).
Five down-projections received **43** INT4 rows, the final one
**69**. Five short-row up-projections received **197** 4-bit
rows and one **195**. This still clusters at the floors and
ceilings: a gain-per-byte greedy objective continues to
overallocate cheap up-projection upgrades even when every
down-projection has a safety floor.

**Interpretation:** limiting per-matrix starvation alone does not
produce useful all-layer model quality. The next defensible
experiment should evaluate a TRAIN-only *joint model-level
loss* sensitivity or layerwise reconstruction interaction,
rather than repeatedly treating local activation-output SSE
as end-to-end quality. Any next hyperparameters must be
preregistered before testing a NEW sealed corpus/slice;
do not tune the present SST-2 validation set after seeing
these results. A 2048-token movie-review sample is not a
broad language benchmark. The CB36 bundles are still only
MLP matrices, not a deployable full-model checkpoint, and
all PPL was measured after dense decoding, NOT from native
packed streaming. No QMoE Switch 0.8bpw reproduction or
new multidimensional neural weight architecture demonstrated.

No main, Render, Supabase, APK, production Jessica, or ordinary
Jessica chat changes.
