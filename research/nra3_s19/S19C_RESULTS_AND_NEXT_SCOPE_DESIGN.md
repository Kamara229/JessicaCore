# S19c — First learned event factuality adapter on real Calibri43M + GT

**Result (research only): GitHub Actions #44126483606 SUCCESS.**
The CI completed 28 unit tests, all 12 fixed TRAIN-internal grouped
folds, and physical FP16 checks. No independent/EVAL corpus was accessed.

## Mechanism

Freeze S19b's **6,528-FP16** learned raw-text/frozen-Calibri grammar parser
and train an additional **68-FP16** 16->4 event factuality head.
The full physically accounted 6,596-FP16 payload is SHA-sealed.
The grammar input is **real** frozen original Calibri43M+Green Tunnel
post-layer8 320-D hidden, plus 18 raw text surface indicators. No lemma,
UPOS, dependency, predicate identity, S18 annotated roles or truth labels
are inference inputs. The event status is predicted for EVERY word:
NONE, ASSERTED, NEGATED, FUTURE_UNREALIZED.

The S18 original 96 TRAIN labels have 108 verb events:
84 ASSERTED, 12 NEGATED, 12 FUTURE_UNREALIZED. These are
**hand-authored synthetic TRAIN**, not a representative natural-Russian set.
S19b grammar is re-trained on **only TRAIN-side rows** in each fold, then
frozen before the event head learns on the same rows. The held group
never participates in a grammar or event gradient, checkpoint selection,
or hyperparameter choice. The final all-TRAIN model reloads the immutable
S19b physical grammar artifact. Old EVAL32/EVAL128 and prior spent EVAL
are never loaded and cannot be rerun.

## Actual measurements: not independent quality

| Metric | TRAIN-internal 4 lexical-fold aggregate | TRAIN-internal 8 syntax-fold aggregate |
|---|---:|---:|
| Correct predicate **and** truth status, micro-F1 | **63.55%** | **45.50%** |
| Gold verb events | 108 | 108 |
| TP | 68 | 48 |
| FP | 38 | 55 |
| FN | 40 | 60 |
| NEGATED correct out of 12 | **0** | **0** |
| FUTURE_UNREALIZED correct out of 12 | **4** | **0** |

All values are TRAIN-internal for the same 96 synthetic sentences, NOT
held-out independent language competence; no direct comparison with S18c7's
40.17% directed **ROLE** F1 is valid because the output target is different.

**Most important falsification:** the adapter is NOT reliably handling
negation or future status. In the leave-*entire construction*-out negative
or future folds, the corresponding training split contains **no examples
of that status**, so no fully supervised head can be expected to learn
that class from this training data alone. Even across leave-lexical-family-out
folds with such labels present, NEGATED has zero correct predictions.
Raw morphological endings and single-word frozen embeddings are
insufficient for robust event scope. Do not tune on the consumed
independent EVAL or claim this is solved by architecture alone.

## Next engineering milestone

Research-only **S19d** should use the already SHA-locked *new nine
TRAIN-only* S19 event graphs and extend them with varied new human-authored
TRAIN examples. First collect genuine raw-text-only Calibri43M+GT
CPU hidden and tokenizer alignment for these **new** sources with the
original checkpoint frozen and explicit J7 protection. Do not rely on
gold predicate anchors or labels to construct model features.

Next, build a **contextual scope parser**:
- per-token raw negation and analytic/finite future cues, but condition on
  a learned predicate/dependency head; negate the actual predicate,
  not an unrelated participant;
- separate **event truth** vs **argument-level rejection**;
- represent temporal vs conditional discourse relations and coreference
  as separate **predicted** node/edge heads, not manually injected answers;
- train using strict split isolation; freeze and preregister a *new,
  genuinely independent* human test only after development.

Do not activate, promote or merge any candidate before protected J7
on its **active** path, fully inclusive bpp <= 0.7 and actual full
CPU speed >= 90% of CB59 baseline are confirmed. Original Calibri,
GT, LIVE, Render, Supabase, Jessica ordinary chat and APK remain untouched.

Latest physical SHA256 S19c combined grammar/event:
a3a8a4de32e8669c851437ad09340512609d193a607bcadc58dc5be73cb80d2a.
