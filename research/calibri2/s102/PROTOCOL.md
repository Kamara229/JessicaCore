# CALIBRI 2.0 S102 — hybrid S101 + protected J7 method (RESEARCH ONLY)

## Decision based on actual immutable CI evidence

This is **not** a merger of two sets of numerical weights. S101 contains the
strongest demonstrated niche-task quality: S99 = 353/456 strict full+EOS
authorized TRAIN; S100 frozen one-shot = 42/64 newly authored noun/owner
contrast prompts (S82/S95 = 0/64), although S100 old S82-relative CPU speed
0.8076 **failed**. S101 incrementally cached that exact S99 policy with
353/456 exact response parity, 0 drift, 1.39158x CPU throughput vs S82
measured on 48 frozen TRAIN cases, and inclusive 0.699573397602 bpp.

S87-J7 separately **proved a useful TRAIN safeguard**, not a quality gain:
64 real Apple MPS gradient steps with 8-step transactions; 2 transactions
regressed one of the original 196 S82 successes and were fully rolled back,
ultimately preserving 196/456 and 0.685140 bpp. J7 learned no new exact
TRAIN answer; it was trained from **S82**, not from S101's S93 core/bank.
Do not copy its parameters/optimizer directly into S101. Port the **control
mechanism** only.

## Actual S101 runtime ABI (to preserve)

1. **One S93 36,380,480-parameter paid CB59/CB61 compressed core.**
   Core codebooks are fixed 5-bit indices, 16-bit centroids, immutable row
   scales and tied aliases. Only two paid dictionaries
   `model.layers.{3,9}.mlp.down_proj.weight` varied in S87/S91/S93.
2. **S94 lossless alternative numerical profile bank**, with exact S87/S91
   codebooks and layer-3/9 Green Tunnel tensors. The S93 base profile remains
   in physical compressed core + S93 neural sidecar. No complete second/third
   36M backbone is stored.
3. **Frozen S95 prompt-only neural router** chooses one of S87/S91/S93.
   No question-id/answer routing table may enter inference.
4. **Frozen S98 neural noun copier** may override only the second generated
   content token when its context-based confidence passes 0.95.
5. **S101 exact KV-cache CPU decoder** must be used for both the baseline
   and candidate comparisons, with source prompt keys held for S82 pointer,
   original first/later policies frozen, and EOS included. Cached and
   uncached 456-TRAIN outputs were previously equal exactly.
6. Green Tunnel is currently *only in decoder layers 3 and 9 out of 12*.
   This hybrid pilot does NOT attach new adapters to all layers, which would
   risk exceeding the strict physical budget. Full-layer adapters are a
   separate controlled hypothesis, not a prerequisite.

## Physical budget / hybrid feasibility

The S101 ALL-IN = 0.699573397602 bpp. At 36,380,480 source parameters,
the exact remaining physical margin is approximately **1,940 bytes**.
That rules out adding another rank-8 adapter, a duplicate S87-J7 controller,
an unaccounted router, hidden QA data, or any extra inference tensor.

The only sensible initial trainable scope is **existing S93 paid FP16 centroid
cells + existing S93 layer-3/9 internal Green Tunnel**. S87/S91 alternative
bank weights, S95 router, S98 copy scorer, original S82 controller, CB59 5-bit
indices and source row scales stay immutable. Repacking a changed S93 CB61
may increase/decrease zlib bytes, so every candidate must recompute **real
physical bytes** and fail closed above 0.700 bpp. Saved optimizer state,
TRAIN metadata and guard reports are research-only and are not included
in the inference bundle.

Under the S95 route, only prompts routed to S93 can be improved by modifying
S93. Changes to S93 are switched away on frozen S87/S91 routes and must not
accidentally contaminate them. Existing second-token copied decisions are
non-differentiable argmax/threshold decisions: naive S87-J7 token-CE through
bare logits cannot be assumed to optimize the *actual* S101 output. This
needs a separate MPS objective aligned to the frozen router and S98 policy
(or a clearly isolated first/EOS intervention), not blind J7 weight copying.

## Protected transaction protocol: extension of J7

The original J7 no-forgetting baseline was **196 S82-correct TRAIN rows**.
The *new immutable S101 baseline must protect all 353 exact full+EOS correct
TRAIN rows*, plus **every previously correct first token** among the full
456. All 456 authoritatively known TRAIN rows are checked using actual
serialized CB59/CB61 PyTorch CPU and the S101 KV-cache policy.

For each future atomic MPS update block (suggested 8 steps):

1. Save exact copies of both trainable S93 centroids, all 3/9 internal GT
   parameters and the **complete Adam momentum/step state**.
2. Perform gradient updates **only** on authorized existing TRAIN examples.
   Protected TRAIN replay may be used. Do NOT read any historical sealed QA,
   including S79, S85, S96 or S100 (whose 42/64 result is no longer untouched).
3. Physically serialize candidate core and exact new sidecar; account for
   bank, router, scorer and metadata within 0.7 bpp.
4. Reconstruct physical CPU runtime and regenerate all 456 full+EOS answers
   **using the identical S101 cached S95/S98 policy**. Compute lost/gained
   individual identities, not merely aggregate score counts. Verify every
   old first token too.
5. If any regression or physical overflow: revert *both* parameters and
   Adam state byte-exact, confirm restored 353 on CPU, halve LR (or smaller
   trial block), and mark rejected proposal. If safe: commit temporary TRAIN
   checkpoint. No inference parameter outside paid S93 scope changes.
6. Final candidate must provide **>=1 net new correct full+EOS answer** on
   TRAIN, 0 losses among 353 and old first tokens, all-in <=0.7 bpp, measured
   same-run cold CPU speed >=0.90x S82 and exact CPU/MPS functional parity.
   Passing TRAIN is **not** independent model acceptance.

After training and before any claim of generalizable quality, freeze a new
candidate SHA and perform **one** fresh preregistered, uninspected independent
evaluation of novel questions. The old S100 sample can be referenced only as
historical evidence; it must never be reused to train, select hyperparameters,
or claim a second independent victory. Include natural responses and a
matched 36M FP32 control when available, not an incomparable 135M model
presented as parameter-matched FP32.

## Current S102 deliverables (stage A + first guarded MPS stage B)

- `hybrid_retention.py`: strict 353-full+first identity checks and atomic
  optimizer+weights rollback, with numerical Adam self-test.
- `hybrid_frozen_anchor.py`: independent read-only reconstruction of 456
  S101 CPU answers, frozen profile/sidecar provenance, numerical/physical
  headroom and correct-identity diagnostics. Produces a **research evidence
  artifact only**, not a stored inference lookup table.
- GitHub Actions macOS-15 pinned CPU + real-MPS workflow; source QA-embargo check.
- **Stage A confirmed by CI #42845071504**: 353 correct full+EOS, 387 prior
  correct first tokens, S95 router selected S93 on 436/456 TRAIN prompts
  (333 of the 353 correct); S87 15/15, S91 5/5. Exact available physical
  margin is 1,940 bytes, same frozen CPU ratio 1.39158 vs S82. Numeric
  synthetic J7 rollback succeeded.
- `hybrid_mps_transaction.py`: first **REAL Apple MPS 8-step S93 joint
  codebook+internal-tunnel update** with 2 wrong + 4 protected S93-routed
  TRAIN examples each step. Train through frozen S82 first/later policy;
  S98 discrete override is evaluated in true final S101 greedy CPU path,
  not falsely advertised as differentiable. Freeze S87/S91 alternatives,
  S95 router, S98 copy and zero gates. Physically repack changed S93 source
  and sidecar; compare ALL 456 S101 cached CPU outputs against original,
  enforce 353 full+EOS / 387 first preservation, <=0.7 bpp, rollback BOTH
  model and Adam on failure. This is a pilot: **results not yet verified**;
  no new quality or acceptance should be claimed until its own CI completes.
- Future Stage C: if the pilot fails, use exact failure categories to revise
  MPS objective without touching inspected QA, re-run fresh training only.
  Before any model acceptance require same-run S82 native CPU speed, GPU/CPU
  parity and a preregistered new held-out QA.
- Never touch `main`, LIVE Jessica, Render, Supabase or APK; never merge
  or promote without explicit user instruction.
