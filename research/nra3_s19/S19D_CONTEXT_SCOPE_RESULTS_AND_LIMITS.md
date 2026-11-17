# S19d — Contextual event scope, actual original Calibri43M+GT CPU

Research-only checkpoint: **GitHub Actions #44128707585 SUCCESS**, 34
source/synthetic unit tests passed and twelve grouped TRAIN-only folds
completed. Physical candidate 9,332 FP16 params total (6,528 frozen learned
S19b grammar + 2,804 trainable 5-word context/convolution and status head).
Combined original full-TRAIN96 candidate SHA256:
928f735a2018604db2ffe649799778df3805162b79a989a36e282dc3cb61c774

## Main finding: local versus contextual event classification

| Predict BOTH correct predicate and event status | S19c local | S19d contextual | Difference |
|---|---:|---:|---:|
| TRAIN-internal lexical-heldout micro-F1 | 63.55% | **93.20%** | +29.65 percentage points |
| TRAIN-internal leave-construction-out micro-F1 | 45.50% | **47.51%** | +2.02 percentage points |
| NEGATED TP / 12, lexical-heldout | 0 | **11** | +11 |
| FUTURE_UNREALIZED TP / 12, lexical-heldout | 4 | **9** | +5 |
| NEGATED TP / 12, leave-construction-out | 0 | **0** | 0 |
| FUTURE_UNREALIZED TP / 12, leave-construction-out | 0 | **0** | 0 |

Source: original S18 96 **synthetic TRAIN** sentences, 600 word tokens and
108 gold predicate-status events counted across each of two different
grouping strategies. Lexical-heldout counts: TP96 FP2 FN12 (S19d);
syntactic leave-construction-out: TP43 FP30 FN65 (S19d). Every CV fold
retrained the learned grammar on TRAIN-side rows and trained the new
contextual status head only on those rows. No held group, prior EVAL32,
EVAL128 or other spent independent EVAL was used for gradient updates.

**Interpretation:** the local S19c factuality head could not reliably
see sentence-level negation and future cues. S19d's five-word context
window plus raw18 features significantly improves role-neutral
event+truth recognition *within held-out lexical families that reuse
the same templates*. This is a productive architecture result, but is
NOT evidence of robust Russian-language understanding. The unchanged
near-zero out-of-template generalization demonstrates severe
dependence on seen source templates.

A leave-entire-construction-out fold for NEGATED or FUTURE excludes
**every example of that label** from its training rows. A supervised
classifier cannot learn absent labels from that fold alone. To
establish transfer, the NEXT train set must include diverse negative,
future and conditional examples across unrelated syntactic forms.
The nine new S19 TRAIN-only graph examples are currently **source-only**
and cannot yet train this neural adapter: no genuine original frozen
43M+GT CPU hidden for those nine texts has been captured.

This model does NOT learn rejected argument roles, true conditional
event graphs, temporal discourse links, coreference or NRA2 semantic
bridge. Its symmetric context convolution and full-prompt parent
representations are NOT strict causal incremental decoding.

## Next research action

1. Capture real original Calibri43M+GT wordpiece-aligned hidden320 for
   separately sealed **new TRAIN-only graph corpus** (starting with the
   nine S19 texts) using the existing frozen S18b CPU read-only capture
   and original J7 parent identity protections; never synthesize a
   hidden tensor or substitute prior EVAL data.
2. Supervise argument-level rejection separately from event-level
   negation, using raw-only model inputs and human graphs as loss
   targets. Add conditional/temporal and pronoun relation heads only
   after genuinely varied TRAIN input.
3. Validate group-disjoint new lexical/syntactic TRAIN development;
   then preregister a *new* independent one-time Russian EVAL after
   weights/hyperparameters are frozen. Old EVAL32 is permanently spent.
4. Keep gated shadow-only until active protected J7 616 retention,
   physically all-inclusive bpp <= 0.7 and whole CPU >= 90% of CB59
   are independently certified.

Never change LIVE, main, Render, Supabase, Jessica ordinary chat or APK.
