# NRA2 S72 — compositional situation micro-areas (DRAFT, research-only)

## Why the present S65 readout stalls

S66 original 43M + GT first-read 32 source tasks: recipient F1 **75.47%**,
rejected F1 **75%**. In quotation-only/no-real-transfer cases, 5 false
recipient links out of 8. Training more of the same S65 global 3-way
attribution head is not a convincing path to situation understanding.

The bottleneck is representation: one utterance may contain quote + negation +
passive voice + multiple time frames simultaneously. A sentence-level
exclusive situation class cannot model such overlaps. A mini-area must
attach to each **event hypothesis**, not to the whole sentence or a literal
keyword. It must receive the original real Calibri frozen H8 source word
embeddings as well as source-only S65 36D discourse cues.

## Eight simultaneously composable micro-areas

1. `quoted_or_reported` — textual claim vs assertion by narrator.
2. `negation_scope` — whether NOT negates event or an unrelated clause.
3. `passive_role_assignment` — agent vs recipient with reversed order.
4. `temporal_revision` — earlier failed attempt vs later successful event.
5. `hypothetical_or_dream` — dreamed, conditional, interrogative, planned.
6. `reference_resolution` — source/recipient across pronouns, descriptions.
7. `competing_event_clauses` — two predicates in one utterance.
8. `uncertainty_abstain` — ambiguous evidence; preserve genuine NONE edge.

Each mini-area has *different trainable parameters* (12D event encoder,
5 event/factuality correction outputs, 2 event-target outputs). A
multi-label sparse router can activate up to four areas for a single
event hypothesis. Areas are **not rules** that automatically change
a claim's truth; only supervised OLD TRAIN evidence learns their effects.
An area name is a contractual intended specialization, **not proof**
that untrained random weights already distinguish the situation.

## Integration boundary

`Original frozen 12×320 Calibri H8 + old S65 source event36 + old S65 logits`
→ `shared low-rank source/target representation`
→ `8 sparse composable micro-areas`
→ `shadow event×target, factuality and rejected logits`.

Parent model, original Green Tunnel, S132, S62/S63/S65 and J7 answers
are all **immutable**. S72 does not call any production decoder or
attach a hook. A new untrained area's correction is exactly zero.

For real offline classification, the whole prompt is visible. A real
incremental/streaming version would need event-relative prefix-safe H8
and separately proven online causal scope; **S69's all-step S132 phase
is not production-certified**, nor does S70 S95 feature sensitivity
demonstrate actual impact through S111's frozen zero-bridge.

## Prospective training and evaluation (NOT executed by S72)

* Construct *only new labelled TRAIN* event graphs with multi-hot
  per-event situation labels, not heuristic classifications at inference.
  Derive gold for event/recipient/REJECTED only inside the loss.
* Use **counterfactual minimum pairs** across the same names/verbs:
  real transfer / quotation / rumor / denied transfer / passive real
  / dreamed transfer / earlier failed then real later / unresolved.
  Include examples with overlapping conditions, not single-marker templates.
* Keep all paraphrases and counterfactual siblings for one semantic
  scenario on the **same** side of grouped TRAIN development splits.
  Group by situation seed, lexical template and named participant chain.
  The novel prospective evaluation set must be locked before training,
  and read **exactly once** after architecture, hyperparameters and gates
  are frozen. Existing S66 new32 and EVAL128/32 are already spent.
* Fit only the S72 heads, router and auxiliary area BCE on OLD TRAIN;
  never gradient-update original Calibri/GT/S132/S65. Compare against
  the fixed S65 readout under exactly the same original 43M inputs.
  For each area, ablate it and measure case-specific improvement. Compare
  full model vs same-parameter generic low-rank MLP to check whether
  *specialization*, rather than more parameters, is useful.
* Before accepting: do not lose protected baseline J7; measure
  independent prospective recipient/rejected precision, recall and F1,
  quotation-only false links, passive NEGATED scope and abstentions,
  evidence of **pairwise** and compositional generalization; require
  genuine retained 616 protected J7 under integrated candidate, accurate
  all-in on-disk bpp <= 0.7 and whole-CPU ratio >=0.9. No passing claim
  until all physical CI and fresh independent evidence exists.

## Explicitly rejected shortcuts

- Do not route by a gold label, hand-crafted SHA lookup or expected answer.
- Do not use S66 first-read 32 or spent EVAL128/EVAL32 for iterative fitting.
- Do not treat isolated zero-initialized experts as trained or competent.
- Do not attach S72 to all 12 decoder layers before causal J7 regression.
- Do not activate a candidate, alter main, LIVE, Render, Supabase, Jessica
  or APK; create DRAFT PR only.
