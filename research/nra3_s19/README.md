# NRA3 S19 — Systemic Russian grammar-to-event-graph architecture (research only)

## Why S18c4 is not enough

The frozen original Calibri43M + Green Tunnel / native43 CB59 / S11 router
and the 2,597-FP16 S18c4 directed-role head reached **micro-F1 40.17%**,
versus **23.11%** for the 2,309-FP16 S18c1 head, on a *one-time* handmade
NEW32 EVAL (104 positive verb→argument links). The run was GitHub Actions
#43923185797; **EVAL32 is now permanently spent**. On that tiny sample,
future-unrealized was **12.90% F1** and two independent predicates **22.22% F1**.
These numbers do not establish natural-language generalization.

S18c4 is a scorer for ordered word pairs, not a grammar parser or discourse
reasoner. It consumes two frozen 128-D token vectors + raw 18-bit surface
features; it does not build a syntactic tree, learn event polarity, distinguish
negative event vs rejected participant, or link pronouns to entities.
The old TRAIN96 includes 12 repeated "Завтра" future patterns and 12 repeated
"Пока" two-clause patterns. S95 baseline layer8 representations consumed the
**whole prompt**; prefix-only online causality has **not** been established.

## Proposed layered adapter (not yet trained)

Plain text + tokenizer offsets + frozen Calibri/GT 320-D states
    → morphology/UPOS/multi-class tense+case predictions
    → predicate identification and dependency-head/label predictions
    → directed predicate→participant role predictions (including NONE)
    → event-level factuality: ASSERTED / NEGATED / FUTURE_UNREALIZED / CONDITIONAL / UNKNOWN
    → participant-level polarity: POSITIVE / REJECTED
    → entity/coreference links + event↔event links
    → compact event-graph vectors exposed to a separate NRA read-only shadow
    → gated candidate output only **after** J7, CPU, bpp and independent quality gates.

Neither a word ending nor a conjunction is sufficient by itself to infer a
semantic role. The architecture must jointly resolve competing parse hypotheses.
For example, "не клиенту, а юристу" negates a participant selection; it must
not negate "отправила". "Если ... подпишет" is conditional, while "Когда ...
подпишет" marks temporal dependence: neither implies the future events occurred.

### Distinct interfaces

1. RawModelInput contains **only** raw text and text-derived token spans.
   Neither TRAIN tags nor a human gold graph are valid inference inputs.
2. TrainOnlyGoldGraph contains event anchors, directed roles, event
   factuality, rejected/positive argument polarity, coreference and
   inter-event links. This is **target supervision only** and never a feature.
3. Every event link has explicit source/target event IDs; shared arguments are
   represented by separate directed links to the same token. A pronoun's
   referent is a separate coreference link, not a substituted role label.
4. Event NEGATED means the event is explicitly denied; argument REJECTED
   means just that candidate participant was denied while another participant
   can be affirmed. CONDITIONAL and FUTURE_UNREALIZED are not ASSERTED.
5. All word spans are validated against the existing raw-only Unicode
   tokenizer source. A model cannot be given known predicate locations or
   human morphological categories as inputs.

### What S19 actually implements now

The typed, source-only graph contract, strict invariants and **nine distinct
human-authored TRAIN-only** annotated examples:
analytical future, inverted future, separate verbs, coordinated future verbs
sharing an AGENT, conditional clause, future temporal clause, event-level
negation, rejected recipient, past temporal clause with cross-clause coref.

The first S19 source-only CI #43932427314 passed all eight tests and sealed
source TRAIN9 gold/offsets SHA256
fd424c8584a7a8ee719bd44b9540141cd94c687b892bb6d559ef231e78327bb2.
The typed graphs contain 15 events, 40 positive/rejected argument links,
five coreferences and six typed event links. No model quality is measured.

This **does not** modify S18 source96, train any network, run a model, calculate
F1, use old evaluation prompts, or connect to Jessica, LIVE, Render, Supabase
or APK. This is a foundation for model/data implementation, not proof that
the new grammar architecture works.

## Training/evaluation sequence; research-only future work

**Phase A — source audit (current)**: seal nine human TRAIN annotations with
SHA256, unit-test offsets, token/gold separation, scopes and graph invariants.
Expand with externally sourced/licensed text and independent human annotation
under an explicit TRAIN split (include real-world ambiguity and negatives);
avoid using any consumed EVAL for selecting examples or feature tuning.

**Phase B — compact learned grammar head**: train UPOS, morphological features
and dependency arcs on TRAIN, paired with true original Calibri/GT features.
Report TRAIN-internal *group-disjoint* lexical, predicate and construction
splits. Prevent any annotation/lemma/gold POS from entering inference. Do not
pretend raw ending heuristics are POS labels or that older S95 activations prove
causal streaming.

**Phase C — joint role+event+coreference head**: infer predicate anchors,
roles and event factuality; learn argument-scope polarity and event-event
links separately. Compare against *both* frozen S18c1 and S18c4 on
TRAIN-internal development; include explicit controls for wrong source,
role reversal, pronouns, negation and future non-occurrence. Freeze all
hyperparameters and checkpoint SHAs before any new independent EVAL.

**Phase D — NRA integration in shadow only**: encode nodes/edges into small
bounded-width vectors with explicitly accounted trainable FP16 parameters.
No gold/labels in this bridge; disable decoder output gate. Test whether
NRA gains exact graph recall or coherent answer/EOS on protected J7 and new
TRAIN-only tasks. Do not overwrite original Calibri weights or Green Tunnel.

**Phase E — fresh independent certification**: preregister a *new, genuinely
unseen* natural-Russian gold corpus and a one-time scoring protocol only after
the candidate is frozen. Report per-subtask F1 and calibration, UAS/LAS,
event-scope accuracy, coreference link F1, exact graph match, two-word EOS,
protected 616 J7 no regression, physically complete **bpp <= 0.7** and
real measured **CPU throughput >= 90% of CB59 baseline**. Original parent
J7 passed previously, but the S18c4 candidate is NOT itself J7/bpp/CPU
certified. A failed gate means no activation.

## Hard restrictions

- Existing EVAL32, earlier S17c/S15c/S12c EVAL, EVAL128 and reserved tests
  remain off-limits for learning, feature discovery, retuning or rerunning.
- Only study an independent branch from S18c7; keep PR draft and unmerged.
- No Jessica ordinary chat tests, no LIVE, no Render, no Supabase, no APK.
- No semantic claims about trained model results from the nine TRAIN targets.
- Do not spend all-inclusive bpp/CPU budget using unaccounted FP16 tensors.
- Raw model features can be computed from plain text only; no oracle input.
- Strict left-to-right inference is a **separate unproven requirement** until
  prefix-equivalence is actually verified with genuine original model calls.

### Reproduce the source-only contract

    python -m unittest -v research.nra3_s19.test_event_graph_contracts

All these tests are *schema validation only*, not model evaluations.
