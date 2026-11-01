# CALIBRI session v41 — disjoint real pretrained TRAIN-A selection, TRAIN-B one-shot acceptance, unseen SQuAD

## Existing observed data, not speculative gains

v40 research PR #169 / Actions run 41361751878 completed on two independent Linux CPUs with consistent SHA and PPL. **ALL 12 actual pretrained Pythia-14m MLP projections** (786,432 coefficients) occupied exactly **343,840 physically persisted bytes, ~3.49772 bits/MLP parameter**, including signed INT3/INT4 bitstreams, per-row FP16 scales, CRC and CB36 outer SHA. These archives contain only MLPs, NOT the complete checkpoint. v40 joint TRAIN-NLL optimized candidate scored IMDb first-use test PPL **357.19382**, worse than equal-byte LOCAL TRAIN-calibrated **357.03614**; source teacher **154.91574**. Raising TRAIN tokens 256→1024 improved both local and joint absolute PPL, but failed matched-rate joint-vs-local generalization. WikiText DEV was less negative for joint, illustrating development/test mismatch.

PR #170 repaired Codex interpretation of completed CALIBRI no-test research messages: CI passed, merged code bead82ce..., Render auto-deploy LIVE, Supabase read-only ACTIVE_HEALTHY, no ordinary Jessica test was run. These backend repairs did not change research weights.

## Registered v41 hypothesis and frozen design BEFORE touching new evaluation text

**Question:** Does accepting the v39 full pretrained-model TRAIN-NLL-guided zero-byte row precision swap ONLY when it also improves on a separate TRAIN-only token window avoid the v39/v40 small, harmful held-out degradation? This is a train-selection hypothesis, NOT a guarantee. A neutral / local-fallback result is not evidence of superiority.

Use unmodified original **EleutherAI/pythia-14m**, all twelve source MLP matrices, same pinned v33 physically signed 3/4-bit CT33 + v35 TRAIN-based FP16 per-row scale calibration, original v39 2-round/at-most-20-candidate-per-round joint full-model TRAIN-NLL search, per-module floors min ceil(75% × LOCAL INT4 rows), caps max floor(125% × LOCAL INT4 rows), and EXACT real mixed-CB36 **343,840 physical bytes**.

**TRAIN A:** original ordered WikiText-2 TRAIN tokens [0:2048), only source used for activation collection, local bit planning, rowwise FP16 scale-grid calibration, MLP marginal SSE proposal proxy, and selecting joint full-model NLL swaps. With all twelve quantized at once, accept only actual model TRAIN-A NLL reductions; search uses no other corpus.

**TRAIN B:** same tokenizer and fixed WikiText-2 TRAIN dataset, disjoint continuation tokens [2048:4096). Do not inspect it during proposal construction, calibration or 2-round iterative NLL selection. After joint TRAINA search is OVER, evaluate exactly TWO models on B ONCE: local matched-byte calibrated MLP bundle and final fully-trained joint-A calibrated bundle. If joint B NLL is lower than local B NLL by strictly **more than 1e-6**, keep joint; otherwise keep original local. No further row-swap trials or hyperparameter decisions may use B. This is an **internal train-selection gate**, NOT a genuinely untouched validation set after using it. Always retain ungated candidate and gated decision as physically packaged variants to expose false-positive selection. If gate rejects, gated=local PPL by construction; this trivial equality MUST NOT be presented as evidence of model improvement.

**New first-use external domain:** `rajpurkar/squad`, fixed validation[:64] original ordered context+question passages, first **2048 pretrained Pythia tokenizer tokens**; source and token SHA recorded. Source text must be read ONLY AFTER full TRAIN-A search, TRAIN-B acceptance, and actual on-disk write/read and SHA validation of all compared physical candidate and baseline CB36 bundles. This is genuinely separate QA-domain text compared to WikiText, AG News, SST2, SciQ and IMDb already used. It is not a full SQuAD accuracy benchmark: PPL only, on small 2048 tokens. Use known WikiText-2 DEV first 1024 tokens as a non-blind historical reference.

**Physical and equal-rate controls:** physically saved local uncalibrated, local TRAIN-A FP16-calibrated, ungated joint-A calibrated, one-shot TRAIN-B-gated calibrated, and higher-rate normal rowwise physical INT4 anchor (about 401,224 bytes). All first four MUST have exactly 343,840 real disk bytes in complete CB36 archive incl. metadata/scales/CRC/outer SHA and all twelve matrices. Absolute source original teacher PPL is also measured, but source weights are not a 343,840 byte control. Model evaluates full pretrained Pythia with the 12 physically decoded dense MLPs substituted, all other layers untouched. No speed of packed execution was measured.

**Cross-host verification:** two separate hosted Ubuntu 22.04/24.04 CPUs; PyTorch CPU 2.5.1, Transformers 4.46.3, selected ATen AVX2 with requested oneDNN AVX2, deterministic algorithms, two threads, pin model/dataset revisions. Independent third CI job verifies exact source model and MLP SHA, disjoint TRAIN A/B token SHA, final gate constraint and chosen file SHA, physical CB36 bytes all 12 modules, numerical outputs and PPL cross CPU. If host-level numerical outputs differ, report separately; unequal physical bytes or sources cause CI FAIL. A lower PPL on the fresh SQuAD requires real measurements, not reasoning from training.

## Outcomes and limitations to preserve

Report TRAIN A NLL gains, independent TRAIN B local/joint NLL, accept/reject, and full-model *new* SQuAD and known WikiText DEV PPL for source, same-rate local/ungated joint/gated joint, and higher-rate INT4. If on SQuAD the ungated joint worsens and B rejects it, then the gate successfully rejected a candidate in this one trial, but it did not itself compress or improve over equal-rate local (gated model is identical to local). If accepted joint worsens new SQuAD then the gate failed. Even if accepted and improves SQuAD, one test is insufficient for statistical claims and does not prove multidimensional new trained neural weights, 0.8-bpw Switch/QMoE, a whole deployable model checkpoint, native packed speed, portability, global rate-distortion optimality or broad generalization.

**ISOLATION:** all code and two-host experiments in `research/calibri-split-train-holdout-v41` and draft research PR, NEVER merge to main, modify working Jessica backend, Render, Supabase, APK, or use ordinary Jessica text test. Honest negative results and CI failures are archived.


## ACTUAL experiment complete — GitHub Actions run 41363524389

Three GitHub Actions jobs **SUCCESS** (Ubuntu 22.04 + Ubuntu 24.04
independent pretrained Pythia-14m runs, and a strict independent
cross-host verifier). The verifier reported `integrity_pass: true`,
`physical_source_binaries_bitwise_equal: true`, `errors: []`,
and `PPL_bitwise_identical_across_hosts: true`. All model revisions,
12 genuine pretrained MLP tensor SHA, WikiText TRAIN A/B and
validation token SHA, previously unused SQuAD dataset/token SHA,
all five physical file SHA and sizes match. CI PASS is provenance
and reproducibility evidence, **not** proof of a universally
good compression method.

### Actual 2048-token TRAIN-A / independent 2048-token TRAIN-B

- TRAIN-A local FP16-row-scale calibrated **full model NLL**
  **6.3493035436**; after v39 frozen two-round 20-probe/round
  joint FULL-model TRAIN-A NLL optimization **6.3424172103**;
  training improvement ~0.00688633.
- The selected candidate was then tested exactly **once** on
  the *nonoverlapping* WikiText TRAIN-B tokens [2048:4096):
  same-rate local calibrated NLL **6.3833079040**;
  selected joint candidate NLL **6.3773140013**.
- Frozen policy (B NLL improvement strictly >0.000001)
  **ACCEPTED** the TRAIN-A jointly selected model. Improvement
  over local on independent B is **0.00599390 NLL**.
  Internal B was not used to rank or retry TRAIN-A candidates.

### Actual, on-disk MLP-only CB36 sizes & hashes

Original trained 12 MLP tensors: **786,432 coefficient positions**.
Every candidate signed INT3/4 binary, whether local uncalibrated,
local TRAIN-A calibrated, ungated joint calibrated, or final
B-gated calibrated, physically occupied **343,840 bytes**
(~**3.497721 bpw** including all FP16 row scales, module CRC,
CB36 manifest/outer SHA). The independent same-rate local
calibrated physical SHA256:
`cd0209af68cf07a8dc81b0c9565e3e9971454e5f80840c7afaa2a4018ba1f64c`.
The accepted joint and final gated physical SHA256 are identical:
`4f56f4ba7af027225ded8ce3fa8ab5fe9c9143a54999c096aa5ffe35caca2878`,
both cross-host identical. HIGHER RATE normal INT4 control actually
occupied **401,224 bytes** and is **not equal-byte**.

**These files ONLY include 12 Pythia MLP matrices, not embeddings,
attention, model config or tokenizer.** PPL was evaluated using
dense decoded weights loaded from the physical files; there was
NO native packed-weights inference or timed serving.

### Actual FULL-model PPL — all 12 MLPs swapped together

| Mode | WikiText DEV PPL | FIRST-USE SQuAD validation PPL |
|---|---:|---:|
| Original uncompressed pretrained source | **173.381488** | **57.953277** |
| LOCAL uncalibrated, 343,840 B | **1620.402307** | **527.014534** |
| LOCAL TRAIN-A-calibrated, 343,840 B | **440.064276** | **146.571447** |
| JOINT TRAIN-A NLL-selected ungated, 343,840 B | **437.005484** | **146.401506** |
| **JOINT selected and accepted on TRAIN-B, 343,840 B** | **437.005484** | **146.401506** |
| HIGHER RATE INT4, 401,224 B | **372.077576** | **123.421469** |

**Small POSITIVE equal-byte exploratory effect:** Joint (vs local
calibrated) PPL changed WikiText DEV **-3.05879** and the
predeclared *first-ever* external SQuAD validation
**-0.16994** (146.57145 → **146.40151**). This was one
2048-token natural language slice of public QA context+question
text, **NOT SQuAD question answering accuracy** and not a
statistically meaningful proven broad improvement. Unlike v39
and v40 the recorded matched-rate new-corpus result was
slightly favorable, but it is tiny relative to the enormous
original-vs-compressed gap: source PPL **57.95328**
versus compressed joint **146.40151** on SQuAD.
The stronger but larger INT4 control **123.42147**
still outperformed all 3.49772-bpw candidates.

No observed success transferring 4-bit row precision
between distinct layers is claimed: the experiment selects
from v39 same/cross-module candidates and the independent
B gate tests only the final chosen joint model. The logged
selection trace can establish particular accepted moves.
Our result does **not** isolate whether TRAIN-A broadening
or TRAIN-B gating caused the observed small improvement
versus v40 because both the calibration corpus length
and external evaluation text differ.

### Scientific decision

This is the FIRST **slightly better heldout matched-rate**
full-model PPL probe for this line of joint Pythia MLP
compression, with source integrity and cross-host agreement,
but *not yet a useful quality model* and not yet novel
multidimensional physical weight layout. The next valid
stage is a genuinely separate precommitted evaluation suite
covering multiple previously unseen text domains/slices,
with v41 quantizer, byte budget, acceptance rule, v39
proposals and hyperparameters FROZEN. Do not tune after
seeing SQuAD numbers; make sure no external test sample
is reused as selection data. To make research more
useful, quantify attainable **rate–distortion curve at
several physical bit budgets** and implement blockwise
decode/compute benchmarking, not just PPL from dense
restored projections; those are separate future tasks.

QMoE Switch 0.8 bits/weight, whole-model deployable compressed
weights, actual native packed-execution speed, innovative
multidimensional tensor graph parameterization and broad
statistical gains have NOT been demonstrated.

No main, Render, Supabase, APK, production Jessica or ordinary
Jessica chat testing changes in this research experiment.
