# CALIBRI v196 — GREEN TUNNEL: continue from actual 135M v186 trained compressed weights

## Research decision

Return to the one positive partial recovery result: **CALIBRI v186**
actually modified the original 135M pretrained packed physical weights
and achieved **2/24 confirmed answers**, **3/12 held-out paired
first-token contrast successes**. Complete model was physically
**11,229,303 bytes = 0.667839 bits/original unique parameter** and
ran at **99.59% of old CB59 native CPU** in matched tests. This did
NOT yet restore reliable language or arithmetic.

A later larger **360M** original FP32 had stronger *original*
arithmetic results, but its actual fully physically compressed
v194 checkpoint at **0.648907 bpp** produced **0/16 meaningful
answers and repetitive gibberish**. v195 independently verified
3 original 360M operator C-native outputs against exact same packed
weights (max relative L2 4.45e-7), then 6 hybrid UNPAID FP32
family substitutions **all** gave **0/8 machine-correct**, so
single-family substitutions did NOT locate a clean rescue.
We must not confuse these failures with a universal impossibility
of learning a better packed representation.

## What Green Tunnel means in the project

**A source adaptation / recovery mechanism** that receives
(1) an already compressed paid source and its exact decoder,
(2) ORIGINAL FP32 model for *public TRAIN-only* teacher guidance,
(3) diverse public text and contrastive answer examples,
(4) an adjustable integer count of repeated recovery cycles,
and (5) a budgeted source-writer to physically commit changes
inside its original file. Its core methodology is potentially
portable to other architectures, but **we have NOT demonstrated
universal cross-model success** and must supply new codec/model
adapters and validation for a new architecture.

This version **does not have a separate neural inference adapter**
or uncharged neural memory. "Tunnel" describes the iterative
transformation of ALREADY PAID compressed weights. All trainable
weights are existing 211 model source FP16 codebooks and 5-bit
indices; original 1D vectors and original per-row scales stay frozen.

## Two full restoration cycles, all source layers, all output words

- Start **EXACTLY from real v186 trained 135M native CB59**:
  11,229,303-byte archive source SHA
  `933daaecf9011f77e3705308faab9d6074b5fb7c4c847d5e1bdb830ad47b965a`.
  Do not reset source to CB59, discard learned codebooks or touch
  LIVE Jessica.
- Repeat 96 registered end-to-end, full-model gradient
  updates per cycle, **2 fixed cycles / 192 actual gradients**,
  updating **ALL 211 paid codebooks** on every update.
- Each language-recovery sample uses original model guidance
  over **all 49,152 vocabulary logits** at selected public TRAIN
  time steps, not a narrow handpicked Yes/No/5/red output
  vocabulary. We do NOT claim we literally fed in every existing
  human word: a tokenizer's vocabulary and a language's words
  are different concepts. This is a full-vocabulary gradient.
- Text source is existing PUBLIC TRAIN WikiText-2/AG News
  samples plus separate original answer-bearing instruction
  samples. 3/4 of optimizer steps use these mixed text/answer
  targets; 1/4 use balanced opposing pairs to preserve
  context-sensitive Yes/No, arithmetic and color bindings.
  Those are original v186 public TRAIN pairs.
- All 211 matrices update paid FP16 centroids from the
  whole-model teacher-guided training gradient each step;
  one registered genuine source matrix at a time may
  additionally replace up to 512 of its existing 5-bit
  group indices. Proposed index replacements are accepted
  only if they improve the ORIGINAL TRAIN objective
  after centroid update. No extra weights, sidecar,
  optimizer state, teacher tensors, or dataset are
  carried into physical source / native inference.
- Do NOT select best recovery cycle by public QA or
  validation. Exactly two cycles predefined, one final
  source sealed BEFORE opening held-out VALID or DEV.
  No training loss, teacher divergence, KL, NLL, PPL or
  first-token success is accepted as proof of useful
  language restoration.

## Quality, physically checked, not inferred

The complete source writer retains exact source file
size 11,229,303, alias handling, CRC and SHA. Native C uses
only paid 5bit trained source. Measure:
1. Native pack vs differentiable optimized source logits
   on TRAIN input (numerical integrity only).
2. 12 held-out v186 opposing VALID pairs from preexisting
   frozen public data: check complete source first-token
   pairs both correct and actual response behavior.
3. Existing 24 original public historical QA prompts, all
   raw 32-token strings/grades across original FP32,
   original CB59, trained v186, and NEW v196,
   including EOS, loops, contradictions and ungraded.
4. The previously sealed v193 16 public DEV questions
   (4 each of 4 categories), original 135M FP32
   actual answers previously frozen, unchanged exact
   v192 grader. Preserve incorrect and ungraded.
5. Eight genuinely matched native CPU timing trials
   against the original CB59 (same work, same CPU,
   same 32-token cap including prefill). Ratio >=90%
   is a separate requested speed gate, not claimed
   until actual source-native execution.

If native outputs degenerate as in v194, this experiment
is **FAIL** regardless of training loss or changed codes.
A larger model can hypothesize more cycles, but **a
cycle count alone cannot guarantee reconstruction**.

### Model-family independence scope

`policy.py` exposes model-size-driven cycle budgets
as **untested planning**, independent of SmolLM2 internals.
The concrete `run.py` adapter targets 135M's
original 211 native packed matrices. Future 360M
and heterogeneous MoE must implement matching shape,
codec and memory governance. The policy does NOT
magically run on arbitrary neural networks today.

All GitHub research in a draft isolated branch,
no changes to main, LIVE Jessica, Render, Supabase,
APK, Green Tunnel production (none exists) or
sealed v89 independent TEST. Jessica is disconnected
and never runs QA.
