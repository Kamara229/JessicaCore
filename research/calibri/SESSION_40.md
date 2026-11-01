# CALIBRI session 40 — 4× TRAIN sample, same real physical 12 MLP bytes, first-use IMDb test

## Verified earlier experimental evidence

v39 / PR #168, Actions 41361149476: independent CPU jobs and
cross-host physical source comparison passed on REAL pretrained
Pythia-14m. All 12 source MLP matrices were individually physically
signed INT3/INT4 encoded with FP16 row scales and CRC, then bundled
into a true on-disk CB36 format with outer SHA256 at **343,840 bytes,
3.497721bpw** (786,432 original MLP parameters). Original uncompressed
model had WikiText2 development PPL 173.38149 and first SciQ validation
PPL 59.74490. At same actual bytes, LOCAL TRAIN-calibrated
PPL was 443.63432 / 150.60845, and two-round joint TRAIN-NLL selected
PPL was 443.87060 / 150.89525: **worse** despite lowering actual
full-model TRAIN NLL from 6.35182285 to 6.34403276.

Previously seen corpora cannot be marketed as untouched: WikiText2
validation, TinyStories, AG News, SST2 and SciQ all appeared in
CALIBRI exploratory investigations. No 0.8bpw QMoE replication,
genuinely multidimensional new learned weights, or full-model physical
checkpoint speed was demonstrated.

## Frozen question: does 4× more TRAIN-only data improve joint row swaps?

Use exactly the v39 signed 3/4-bit codec, rank/order, predeclared
2-round × up-to-20 TRAIN-only joint NLL proposals, same 75% minimum /
125% maximum per-module 4-bit row count, identical existing FP16
scale calibration search and CRC-protected CB36 packaging.

Change ONE independent variable:
- v39-length TRAIN calibration/control arm: first **256** token
  IDs of the Pythia-tokenized WikiText2 TRAIN stream.
- broader TRAIN calibration/selection arm: first **1,024** token IDs
  of the SAME source WikiText2 TRAIN stream; context 128, exactly
  4× as many real genuine layer-input activation rows.

Capture real pretrained activations for all 1,024 TRAIN tokens once
then take the deterministic first 256 rows for the short arm. For each
arm independently, apply the same local MLP bit allocation from
TRAIN activation second moments, train-only FP16 row-scale
calibration, and the original v39 joint all-12 TRAIN-NLL-based
zero-byte row precision swap search. Report **both baseline and
joint** TRAIN NLL for each arm, plus original source matrix and
token SHAs, per-module precision bits, all actual physical file sizes.

Hold strict source bytes: each LOCAL 256, JOINT 256, LOCAL 1024,
JOINT 1024 MLP-only physical bundle **must equal EXACTLY
343,840 bytes** including every bitstream, module scale/CRC, outer
manifest/SHA. Persist and read every binary via disk, verify SHA,
all twelve modules and source matrix shapes. Physically packed
4-bit INT4 at **401,224 bytes** is a clearly HIGHER-RATE quality
anchor, not the equal-budget control.

The 256-token arm allows a DIRECT 256 vs 1024 comparison **on
the same never-before-evaluated corpus**, unlike comparisons
across different research sessions that would confound evaluation
corpus and code versions.

## True post-selection evaluation, locked until after binary serialization

Use WikiText2 **validation** first 1,024 tokens as known development
control. Introduce genuinely first-use `stanfordnlp/imdb` public
dataset, `test[:64]` first 64 reviews; normalize literal HTML
breaks to spaces and take the first **2,048 Pythia tokens**.
Record IMDb dataset revision and token SHA256.

**CRITICAL:** do not load either development text or the IMDb
TEST text until after both 256 and 1024 TRAIN allocation,
calibration, actual full-model joint TRAIN-NLL search and physical
bundles are completely frozen on disk. Do not choose the better
TRAIN-data length, bit allocation, or scale-grid hyperparameters
after seeing test PPL. The new IMDb scores are one-shot
exploratory generalization evidence, not a statistically broad
representative movie review sentiment benchmark.

Evaluate real full pretrained Pythia full-model PPL using all 12
MLP modules restored as **dense decoded weights** from the
physical CB36 streams (other original attention, embeddings etc
unchanged). Compare: teacher, 256 LOCAL un/cal, 256 JOINT cal,
1024 LOCAL un/cal, 1024 JOINT cal, and >budget INT4.
Both equal-rate calibration controls must be measured, not inferred.

Repeat on independently allocated GitHub Ubuntu 22.04/24.04 CPUs,
Python 3.11, CPU PyTorch 2.5.1, Transformers 4.46.3, deterministic
algorithms enabled, two threads, selected ATen DispatchStub AVX2
and oneDNN max requested AVX2. Demand identical source model
revision, per-MPL source SHA, train/dev/IMDb token SHA256,
actual CT33/CB36 physical SHA and bytes across hosts.
Numeric TRAIN NLL deviations may be reported separately, never
silently conflated with physical SHA correctness.

## Decision gate

Higher TRAIN-NLL overfitting is a negative result if it increases
first-use IMDb PPL relative to identical-byte LOCAL control.
Larger TRAIN is useful only if it provides a reproducible
improvement on held-out IMDb, with the same 343,840 physical bytes
and other inputs frozen. Even relative success does not prove
new multi-dimensional neural weight mathematics, a useful full
language model, native packed inference speed, RAM reduction,
novel patent, Switch/QMoE 0.8bpw, or an entire deployable checkpoint.

All modifications remain in isolated `research/` branch and
draft PR, no merge to main, changes to Render, Supabase, APK,
production Jessica, user data or use of regular Jessica chat.


## ACTUAL measured outcomes — GitHub Actions run 41361751878 (all 3 jobs PASS)

Two independent hosted CPUs (Ubuntu 22.04 and Ubuntu 24.04) loaded
**real pretrained Pythia-14M** and all 12 source MLP modules.
The independent checker confirmed **exact source model revision,
original matrix SHA, WikiText TRAIN/DEV and first-use IMDb test
token SHA, all seven physical binary SHA and actual disk lengths**,
and **identical PPL** across both CPU hosts. CI PASS means experiment
integrity, not necessarily better rate–distortion.

### Fixed physical storage (strict controls)

Each 256-TRAIN or 1024-TRAIN local un/calibrated or joint calibrated
model contains precisely **343,840 actual bytes** of CB36-packed
12 MLP matrices, **3.49772 bits per source MLP coefficient** with
786,432 original coefficient positions. This includes packed symbols,
per-row FP16 scales, CT33 headers/CRCs, CB36 descriptors and outer
SHA256; both experimental arms were actually persisted to disk,
read back, verified and restored for PPL. The higher-rate INT4
reference is **401,224 bytes** and NOT equal-byte. The original
FP16 MLP projection sizes would total 1,572,864 bytes in theory,
but are not a physical source checkpoint and the 343,840-byte CB36
bundle does NOT store the rest of the model.

### Actual TRAIN joint all-12 loss optimization

| TRAIN calibration/selection data | Local TRAIN NLL before joint search | After 2-round 20-probe/round TRAIN-only joint search | Gain |
|---|---:|---:|---:|
| 256 WikiText TRAIN tokens | 6.3518228531 | **6.3440327644** | 0.0077900887 |
| 1024 WikiText TRAIN tokens | 6.3331585526 | **6.3310850859** | 0.0020734668 |

These NLL values are on DIFFERENT TRAIN token counts and must not
be compared as if calculated on the same heldout text. Both are
train-set optimization metrics, not generalization.

### Full-model PPL (all 12 physical MLPs decoded and substituted)

| Source/configuration | WikiText2 DEV PPL | FIRST-USE IMDb TEST PPL |
|---|---:|---:|
| Original pretrained source, uncompressed | **173.38149** | **154.91574** |
| 256 TRAIN, local uncalibrated, 343,840B | **1597.66884** | **996.42548** |
| 256 TRAIN, LOCAL calibrated, 343,840B | **443.63432** | **359.56549** |
| 256 TRAIN, JOINT NLL-selected calibrated, 343,840B | **443.87060** | **360.18371** |
| 1024 TRAIN, local uncalibrated, 343,840B | **1599.17854** | **1013.36266** |
| 1024 TRAIN, LOCAL calibrated, 343,840B | **424.62494** | **357.03614** |
| 1024 TRAIN, JOINT NLL-selected calibrated, 343,840B | **423.21892** | **357.19382** |
| Physical INT4 HIGHER-RATE control, 401,224B | **372.07758** | **328.03266** |

The IMDb test was frozen until AFTER training-only selection and
both arms' physical files were written. These are real measurements
on a fixed first 2,048-token slice from IMDb test[:64], NOT the
full IMDb benchmark.

**Two distinct findings must not be confused**:

1. **Positive TRAIN-breadth effect:** at equal 343,840 bytes,
   using 1024 rather than 256 WikiText TRAIN tokens improved
   the calibrated JOINT candidate's IMDb test PPL from
   **360.18371 to 357.19382** (decrease 2.98989);
   the calibrated LOCAL comparator itself improved from
   **359.56549 to 357.03614** (decrease 2.52936).
   On WikiText DEV the 1024 JOINT improvement vs 256 JOINT
   is **443.87060 → 423.21892**.
   These are small exploratory absolute improvements, not
   proof that the optimizer is globally beneficial.

2. **Negative primary JOINT-vs-LOCAL matched-rate gate:** with
   **1024 TRAIN tokens**, actual joint NLL search improved
   WikiText DEV relative to LOCAL by **1.40602 PPL**
   (423.21892 vs 424.62494), BUT it was **0.15768 PPL
   WORSE** than local on NEW IMDb TEST (357.19382 vs
   357.03614). Likewise at 256 TRAIN, JOINT was +0.61822
   PPL worse on new IMDb than its local calibrated control.
   Therefore broader calibration did **NOT** establish
   out-of-domain benefit of the joint bit selector over
   simple LOCAL rate-matched calibration.

All compressed variants remain markedly worse than original
pretrained model (IMDb 154.92), and worse than stronger
above-budget INT4 (IMDb 328.03). The 4× TRAIN sample did not
solve severe loss of original model quality from compressing all 12
MLP modules to ~3.498bpw.

### Scientific next gate

Do NOT reselect v40 algorithm parameters by its now-exposed
IMDb test PPL. The next isolated hypothesis should use
**independent TRAIN-only selection and TRAIN-only validation
for accepting individual joint bit swaps**, rather than
optimizing the same training segment and expecting unseen
improvement. Pre-register two non-overlapping WikiText TRAIN
token slices (e.g., 2048 for calibration, next 2048 for
train-validation), score proposals on the latter *without
reading the eventual new external evaluation corpus*. Keep
bit cap, physical CRC, per-module precision limits and
strong local calibrated controls identical. Require an
independent previously unexamined heldout corpus for a
final generalization check, and record negative findings.

No model-wide physical compressed checkpoint, native packed
inference speed, genuinely new multidimensional neural
weight architecture or QMoE SwitchTransformer 0.8bpw
reproduction was demonstrated. No main/Render/Supabase/APK
or production Jessica changes, and no ordinary Jessica
chat text tests were used.
