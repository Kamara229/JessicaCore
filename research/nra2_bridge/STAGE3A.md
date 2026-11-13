# CALIBRI 2.0 NRA2 — Stage 3A: real frozen hidden-state training inputs

## Why this stage is necessary

The independently trained NRA2 (2,241 parameters) accepts graph adjacency,
one-hot source, and one-hot target. The actual Calibri layer-12 port emits
contextual tokens [B,T,320]. These have incompatible semantics, despite both
being vectors. Calling the existing relational NRA on 320D hidden tensors
without a trained projection is unjustified.

Stage 3A therefore collects actual last-layer 320D contextual activations from
the **real frozen compressed 43M CPU decoder** as a pinned physical TRAIN-only
dataset. The next independent experiment will learn a semantic bridge
from TRAIN-only activations to relational inputs. This stage contains no
adapter weights, invented graph labels, or active intervention.

## Physical provenance / leakage protection

- Research branch is based on successful NRA2 Stage2 PR #614.
- Reuses **the entire real S194 -> S190 -> S187** 456+128+32 CPU/J7.
- Loads physically trained NRA2 FP16 from immutable successful standalone
  run #43446925758, pinned with SHA-256.
- Requires Stage2 real CPU J7 exact generated-token parity evidence
  run #43455911398, pinned with physical SHA.
- The hook receives actual decoder post-layer-12 forward output, and
  captures the last contextual vector in the *first prefill* before any
  answer/gold material is consulted. It does not modify the forward output.
- Each feature is tagged only with SHA256(prompt), not a gold answer.
- After all original inference, only the previously declared 32 S190
  **TRAIN** prompt SHA IDs are selected.
- Saves 32x320 FP16 physical tensor with explicit SHA ordering, no labels,
  raw prompts, expected token IDs, or QA/TEST.
- Full real 616 generated-token bitwise SHA digests must equal both S194 and
  Stage2; also all 584 protected and novel32 first/full counts must match.
- Real core and NRA weights remain frozen; output gate 0, no active NRA
  graph inference or generation change. No main, LIVE, Render, Supabase, APK.

This artifact is **not** a complete standalone graph-adapter training set.
The S190 source questions are owner/numeric paraphrases, not pure transitive
chain data. Before training an adapter, the next stage must create TRAIN-only
paired *real Calibri hidden states* and *well-defined relation-graph targets*
with a true held-out split, no label leakage, control for shortcut features,
then verify the previously trained NRA's ability to interpret learned graph
representations. 32 unrelated prompt vectors alone cannot justify a
generality claim.

## Acceptance for Stage 3A only

- Real 32x320 actual frozen Calibri FP16 features physically stored/reloaded.
- 32 distinct train prompt SHA IDs; all feature vectors finite and unique.
- Exact S194 & Stage2 616-token J7 unchanged, 584/584 protected.
- No extra production model parameter, all candidate acceptance flags false.
- CI GREEN and report artifact. Any failure -> REJECT stage.

## Still required for model acceptance

Train true feature-to-graph adaptor, shadow inference on untouched prompts,
then separately train gated residual integration and verify full new +EOS,
584/584 strict J7, paired same-CPU speed >= 90% and conservative all-in
physical <=0.7 bpp. No implicit promotion.
