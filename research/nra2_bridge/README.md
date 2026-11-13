# NRA 2.0 staged integration into CALIBRI 2.0 — Stage 2

**Scope:** research-only, passive first integration, no output influence.

## Physical interface

Trained standalone relational NRA2 FP16 weights are **physically loaded** from
the immutable independently successful CPU run #43446925758. The trained NRA
has 2,241 parameters and consumes graph adjacency + one-hot endpoints.

CALIBRI's physical compressed 43M Transformer emits 320-dimensional contextual
hidden states. These are NOT graph adjacency, nor object IDs. We deliberately
do not invent a conversion or apply the standalone NRA directly to arbitrary
token states. That would give meaningless answers.

The first stage-2 connection therefore attaches the trained standalone area
to **one late read port**: after decoder layer 12 (zero-indexed layer 11).
The passive hook sees original [B,T,320], stores only shape and count and
returns None, preserving the very same original forward output object.
There are no hooks on layers 1–11. The output gate is exactly zero; changing
it without a trained semantic adapter is a hard error.

## Real native CPU guard

The stage2 workflow inherits **all original S195 physical 43M CPU provenance**
and downloads the pinned S194 584+32 actual CPU J7 baseline. It runs the
unchanged S194 -> S190 -> S187 -> S168 genuine frozen CPU pipeline on
456 old + 128 new protected + 32 novel TRAIN prompts.

Requirements: all 584 protected full answers and all 32 generated token-id
sequences must match the S194 pinned digests **bit for bit**.
The experiment is rejected on any token divergence, unavailable checkpoint,
incorrect SHA, wrong 320-wide model ABI, missing last layer or nonzero gate.
Read-port firing on actual native CPU model is required.

No QA/TEST, no external Jessica chat testing, no LIVE or production writes.

## Not accomplished here

* Calibri hidden-state -> relational graph encoder has NOT been learned.
* NRA2 inference does NOT influence any generated token.
* It does NOT improve the new32 (baseline 0/32 full +EOS).
* Candidate CPU slowdown and full source-inclusive bpp have NOT been
  independently measured. This is not an accepted compressed model.
* J7 parity checks only passive attachment, NOT the safety of enabled NRA2.

## Next strictly ordered stages

1. Independently train a feature-to-relation bridge on **TRAIN-only Calibri
   hidden states** while freezing Calibri + learned standalone NRA2. Ensure
   graph labels cannot enter inference, and use truly untouched evaluation
   task families (including distractors/reversals).
2. Test real independent NRA2 predictions with no injection, compare vs
   frozen baseline, record failures. Do not equate training with competence.
3. Train one zero-initialized gated readout from NRA2 representation to
   last Calibri hidden state. Compare final-layer-only vs alternative taps
   in separately isolated runs, not simultaneously. Record CPU/bpp costs.
4. Enable only after genuine fresh physical 616 J7, >=1 correct new complete
   token sequence with EOS and no forgetting; candidate same-CPU speed >=90%
   baseline; conservative all-inclusive compressed size <=0.7 bpp. No
   promotion/main/LIVE without explicit owner authorization.

All prior research S195–S205 unchanged.
