# CALIBRI session 37 — TRAIN-only inter-layer bit allocation after actual all-12 failure

## Completed prerequisite (do not misstate as unfinished)

CALIBRI v36 / PR #165, GitHub Actions **41358761722** completed all
three GitHub jobs, measuring all twelve REAL pretrained Pythia-14m
MLP weight matrices and a newly introduced AG News test corpus.
Two independent hosted CPUs produced matching binary SHA/source token
SHA and identical PPL.

Physical 12-MLP CB36 bundle (786,432 source weights):
- uncalibrated rowwise mixed INT3/INT4 **343,840 bytes (3.49772 bpw)**;
- TRAIN-calibrated at IDENTICAL bytes **343,840 bytes**;
- standard rowwise INT4 HIGHER-RATE **401,224 bytes (4.08146 bpw)**.
AG News TEST first 1024 tokens, full-model PPL:
  original 235.46882; uncalibrated 1528.88080;
  calibrated 650.91275; INT4 higher-rate 470.49882.
WikiText development PPL corresponding:
  original 173.38149; uncalibrated 1597.66884;
  calibrated 443.63432; above-budget INT4 372.07758.

**Scientific verdict**: calibration significantly improves the
specific uncalibrated same-byte control but ALL-12 3.49772bpw
compressed model suffers a huge quality collapse. This is a
**negative practical compression outcome** despite reproducible
physical bytes. No 0.8bpw QMoE, production model or native speed.

## Frozen v37 hypothesis — does global TRAIN sensitivity help at equal bytes?

Rather than impose 3.5bpw on EVERY source matrix separately,
hold the **EXACT observed v36 physical TOTAL BUNDLE SIZE
343,840 bytes** fixed and allow bits to move between matrix/layers.
The model and model task do NOT change; this is a quantization
allocation experiment and NOT fundamentally new tensors.

Baseline: replicate v36's TRAIN activation second-moment row
bit selection independently for each of the twelve source MLP
modules at per-matrix 3.5bpw caps, then apply v35 fixed
TRAIN-output covariance scale calibration. It MUST still occupy
343,840 physical bytes across all twelve CT33 streams and
the self-describing CB36 global header/descriptor/sha.

New candidate allocation, both calibrated and uncalibrated:
- begin with all rows INT3, physically serialize and count every
  CRC, bit symbol, header, FP16 scale, and 12-matrix container byte;
- for EACH individual trained MLP matrix, measure its **training
  NLL penalty** from temporarily replacing only THAT matrix by
  physically equivalent signed INT3 row quantization, restoring
  original weights between measurements;
- on real train-layer input activations, evaluate per-row actual
  `||X @ (W_row - Q3)||² - ||X @ (W_row - Q4)||²` using the
  two fixed existing three-/four-bit scales and codebooks.
- normalize the positive individual-module TRAIN NLL sensitivity
  relative to the median, clipped to a predeclared [.25, 4] range,
  then multiply it by per-row positive TRAIN output SSE improvement
  per **additional actual packed byte** (16 or 64 depending on row width);
- allocate bit upgrades globally by a deterministic descending
  greedy ratio, no looking at validation or AG News data. This is
  a heuristic, NOT an optimal integer knapsack solution;
- apply v35's SAME fixed training scale grid and per-row output SSE
  calibration to the global allocation, without changing any
  per-row allocated bit widths.
- compare five PPL configurations on same original model:
  original uncompressed source, v36 original LOCAL uncalibrated,
  v36 LOCAL calibrated, GLOBAL uncalibrated and GLOBAL calibrated,
  plus HIGHER-RATE standard INT4 quality anchor.

**Exact fixed storage ceiling**: 343,840 physical serialized
bytes for all twelve MLPs and container. Actual bytes and
binary SHA must be reported. Both local and global calibrated
files must have exactly the same bytes as their respective
uncalibrated versions. If global greedy fails to spend every byte,
it is under cap; do not silently call it exact equal-rate.

## Reproducibility / evaluation plan

Training: EXACT 256 WikiText-2 TRAIN tokens, context 128;
all source 12 MLP layers and full-original NLL.
Development diagnostic: 1024 WikiText-2 validation tokens.
NEW SEALED SLICE: 2048 tokens from AG News
`fancyzhx/ag_news`, `test[3000:3200]`, fixed in advance and
separate from v36's first AG News records. Do NOT call AG News
a brand-new dataset: it is an untested **non-overlapping slice
of a dataset used by v36**. AG News must NEVER influence row
allocation, calibration or chosen parameters.

Two independent hosted Linux CPUs, pinned PyTorch2.5.1 CPU,
Transformers4.46.3, selected ATen DispatchStub AVX2,
oneDNN max request AVX2, two CPU threads, deterministic ON.
Record original pretrained checkpoint SHA, dataset revision and
token SHA for TRAIN, dev, and disjoint AG News TEST. Preserve
all real physical CB36 binaries, per-matrix source SHA,
per-row bitwidth counts, train ablation sensitivity and both
corpora true language perplexity. Fail closed if source/physical
corruption, missing layers, budget violation or invalid PPL;
do NOT fail CI merely because quality is worse.

## Decision logic / limitations

If GLOBAL beats LOCAL calibrated model on BOTH corpora at exactly
equal physical bytes, this is meaningful evidence of useful
inter-layer rate allocation for this narrow trained model.
Otherwise retain negative result; random variation or one
positive development corpus is not a universal gain. Even if
relative PPL improves, compare to source and above-budget INT4
and do not call high PPL a usable quality model.

This experiment does NOT compress embeddings, attention or
norm weights; the physically serialized bundle is not a complete
runnable compressed checkpoint. PPL executes full decoded dense
weights, not a memory-streamed native packed engine. CPU
microbenchmark of a Python loop would not be native inference
speed. Greedy bit allocation and quantization have prior art.
No 0.8bit QMoE/ SwitchTransformer reproduction, mobile/ARM
benchmark or new multidimensional neural weights proved.

Research branch and draft PR only: do not change main, Render,
Supabase, APK, production Jessica or user data. NO ordinary Jessica
chat text test requested.
