# CALIBRI 2.0 / NRA3 — neural integration contract (S91)

**Status:** draft research-only, not trained and not production-accepted. The source of truth is the original protected 12-layer, ~43M-parameter Calibri 2.0 with Green Tunnel. All baseline 616 J7, original parent S26/S57/S59/S61/S62/S63/S65 and original checkpoint remain immutable. No LIVE/main/Render/Supabase/Jessica/APK/merge/deploy.

## Two binding design requirements

1. **Inherit NRA2's best elements:** preserve all **eight separate neural mini-areas** (quoted/reported, negation scope, passive, temporal revision, hypothetical/dream, coreference, competing clauses, uncertainty) and the distinction between mentioning an event and asserting its truth. They now consume an explicit shared **neural event-and-argument graph**, not merely independent shallow word-ending flags. Existing original S59 mention proposals remain a frozen comparative reference; original S65 quoted-assertion exclusion must not be bypassed.
2. **NRA3 is part of Calibri neural weights:** new module is registered under **model.model.nra3_neural_weights**, a real PyTorch child of the 12-layer Calibri model. Learnable layer router, syntactic predicate-argument arcs, eight neural area heads, event-state heads and neural memory slots all appear in the **same Calibri state_dict/checkpoint**, not in a standalone runtime sidecar, external parser, or separate reasoning service. The source code lives in a research namespace for isolation; this is not a requirement for a separate production model.

## S91 neural structure

The only input is **original frozen Calibri hidden states** [B, 12, T, 320] in the original native BPE token coordinates and the input-only BPE valid mask. There are **no manually marked event indices, oracle POS/dependency labels or target-answer fields** in a forward call.

1. **Four learned 12-layer routing streams** (morphological, syntactic, event semantics, knowledge). Each learns a distribution over all 12 original layers; no 8th-layer bottleneck. The streams share a 320→24 hidden projector and fuse to 32 neural dimensions. Stream names are intended research interpretations; they are NOT yet empirically demonstrated layer specialisations.
2. **Neural event proposals:** a trainable source-token event/mention logit. A mention is not treated as true just because it was proposed. A future question-conditioned readout must select relevant events *after* question parsing; S91 does not yet have question conditioning.
3. **Neural grammatical relations:** eight trainable pairwise role-specific arc score matrices from an event predicate token to an argument token (actor, patient, recipient, negation scope, speech source, temporal anchor, coreference, event relation). No hardcoded suffix or comma rule is the ground-truth parse.
4. **Event-state heads:** six separately scored nonexclusive aspects, including negated, reported, hypothetical, uncertain, passive and asserted. This is a preliminary neural representation; mutually contradictory states need explicitly supervised consistency constraints in a future training phase.
5. **NRA2 mini-areas retained:** eight independent trainable miniheads, now conditioned jointly on the 12-layer event representation, role context and neural memory context.
6. **Knowledge as neural structure:** a small 16-slot, 32D trainable event-prototype memory is a submodule parameter, read through learned attention. There is a prototype-consolidation loss against a frozen earlier snapshot. **These randomly initialised slots are not yet a verified knowledge store.** Before any claim of retained knowledge, train structured relations and demonstrate preservation of actual old skills vs new skills across a blinded first-read.

New S91 params: **22,311** (nominal FP16 **44,622 bytes**). This is only a **module-weight count**, NOT an all-in bpp/CPU cost, because integration, runtime buffers, Green Tunnel and decoding costs have not been measured.

## Research-only physical integration

- Attachment to a research Calibri object calls add_module on model.model, and the S91 neural weights appear in its state_dict. The original 12 layers and parameters are not replaced or mutated.
- An **explicit scoped** all-12-layer hook collector observes existing outputs after original forward, detaches activations before NRA3, and returns **None** from every hook. The graph is computed from original layer outputs but **does not alter original logits or hidden states**; gate=0, candidate_accepted=false, activation_allowed=false. No hooks during the 616 J7 protected original-generation phase.
- Child-module weights are persisted as part of the research Calibri checkpoint; attaching is not the same as *activating* the candidate. Any future decoder integration requires separate explicit review and independent gates.
- **S91 CI uses only a 12-layer toy Calibri** to prove state_dict registration, exact output identity, all-layer access and parameter gradients. It is NOT a physical 43M/GT or active-J7 test.

## Mandatory evidence gates

| Stage | Experiment | Hard condition |
|---|---|---|
| S91 | In-model neural ABI, role arcs, eight miniheads, prototype weights | Unit CI; no old weights changed, gate 0 |
| S92 | Genuine original Calibri43M/GT **all 12 hidden layers** on wholly NEW source TRAIN-only data, after original 616 J7 replay | BPE SHA and identity; no gold/training imports |
| S93 | Separate external teacher **offline training-only** (Russian UD/morphology) + manually checked TRAIN labels for actor/action/patient/recipient/negation/quote/time and coreference | Teacher never loaded at inference; train, diagnostic and future firstread strictly disjoint by syntactic construction |
| S94 | Multi-objective training: token event mention + labelled event→argument arcs + event states + NRA2 miniheads; train only new embedded neural params | Frozen base 43M/GT; baseline S59 and S65 unchanged |
| S95 | Relation-based consolidation of already acquired events (memory snapshots, distillation, rehearsal); adversarial role-swap/passive/negation pairs | Old relation preservation and new skill acquisition both measured |
| S96 | Freeze exact checkpoint and source/annotation hashes, then **one-time entirely new, independently annotated if available**, prospective first-read | Source-only model logits persisted before opening gold, zero retuning; report end-to-end events, role F1, false positives, source TOP1 and abstention |
| S97 | Causal generation, original protected 616 J7, total bpp ≤0.7, CPU ≥90% CB59, all-in FP16/overhead | NO activation until **all** gates succeed |

**Previous S79 and S86 novel48 tests are SPENT** and may not be reused for training, threshold changes, model selection or another new-candidate score. S83 synthetic DIAG16 was used to examine S85/S90; treat as development diagnostics, never independent first-read. S90 NEW selector regressed vs original S59 on diagnostic (13/16 vs 15/16), so do not silently replace S59. S78/S85 previous same-template metrics were overly optimistic; never claim grammar generalization from S91 architecture alone.

## Why this may help

By making actor→action→object, voice, negation, source attribution, and temporal changes explicit **learned variables**, NRA3 has a stronger opportunity to capture transformations that preserve meaning despite syntactic variation, and to reuse durable relations instead of isolated token templates. This is a research hypothesis. Real improvement is established only by blinded and, where possible, human-adjudicated new syntax tests; standalone CPU/bpp and integration checks remain independent requirements.
