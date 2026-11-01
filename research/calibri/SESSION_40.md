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
