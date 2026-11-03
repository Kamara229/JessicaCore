# CALIBRI v94 — preserve task correctness and useful generation under stronger compression

## Objective and invariant

Original aim is unchanged: compress original pretrained neural weights as much
as practically possible without losing correct answers, meaningful generation
or CPU processing speed. Alternative representations **need not match** FP32
probability distributions. Compression is useful only if the actual model
functions correctly, consistently and quickly.

Metrics for deciding models: grounded correct-answer accuracy on original
public labeled tasks (BoolQ and SciQ), class-balanced accuracy and exact paired
uncertainty, 16 genuine fixed 32-token greedy prompts, repeated-token
concentration, reviewable decoded text, **whole-model autoregressive CPU speed**
and complete physical bpp including all overhead. Neither KL nor NLL can be
used to accept/reject a candidate. Lexical diversity does not establish
semantic competence.

## Fresh TRAIN-heldout phase (not the frozen independent TEST)

- BoolQ: 128 unused original train examples [7200:7328].
- SciQ: 64 unused original train examples [5200:5264]; four shuffled answer
  choices preserving original labeled correct answer with a deterministic
  per-question SHA-derived permutation. Original public train split only.
- Actual original pretrained FP32 **task-correctness answers only** also scored on exactly the same labeled questions as the quality-preservation reference. This measures preserved functionality, NOT agreement of probability distributions. No candidate can be claimed quality-preserving solely because it improves over degraded CB59.
- Both ranges are disjoint from prior CALIBRI v90/v91/v92 diagnostic TRAIN and
  VALID selection. The already observed v89 official test is not read.
- Genuine original real pretrained CB59 source, and v92 rewritten
  model.layers.13.self_attn.v_proj whole-block candidate: identical total
  source bytes, same native C operation, all other packed blocks unchanged.
- 16 varied greedy prompts across arithmetic, factual, reasoning, instruction
  and creative tasks. Record raw emitted tokens AND human-readable decoded
  outputs, repetition fraction, EOS and matched prompt hashes; do not label
  diversity alone as semantic accuracy.
- Alternate benchmark order per prompt. Measure the **entire native packed
  model** including prompt prefill and all 32 greedy decode steps, not a
  fabricated operator-only speed proxy. Measure aggregate tokens/sec and
  empirical speed ratio against CB59 on the same machine. CPU noise may remain.
- Produce machine-readable prioritized functional regression/repair candidates
  for Quality Doctor: repeated token collapse, class majority collapse,
  QA regressions by domain, source over budget and full-model speed loss.

## Decision logic

Immediate reject only for material domain regression, meaningful diversity
regression, source >0.700 bpp, or full model CPU speed <90% CB59. Small sampled
gains are promising but not scientific proof. Even a dual-domain
non-regressing candidate must be reviewed for factual accuracy, coherence and
instruction following on many completed generations before stage for release.

Restore quality with minimal extra bits by first replacing the **entire
compressed defective operator** (source codebook + indices, same native ABI
and physical bytes), trying multiple operator families, then spending optional
limited residual bits only when their benefit per bit is measured. The original
basis must never lose other layers' data silently. No production/release,
no source mutation, no editing Jessica/Render/Supabase/main/APK.

## Facades

provenance.py: verify exact source/delta checksum, archive invariants and
TRAIN-only rows. qa.py: original public BoolQ + SciQ correct labels with
original single-token constrained completion format.
answers.py: genuinely paired per-domain accuracy and uncertainty.
generation.py: fixed-prompt decoding with real model timing and decoded text.
policy.py: task/generation/bpp/speed decision gate.
repair_plan.py: prioritized next safe research trial.
pipeline.py: tiny orchestration only, restores original block in finally.

Any transition to an automated Jessica quality self-repair Skill must be
read-only until a separately approved sandbox/shadow/canary release workflow
can show benefit and roll back a failed change. Never train on the
already-used frozen official test.
