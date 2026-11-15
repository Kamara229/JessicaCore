# NRA2 3.0 / S9 — new Russian UD TRAIN72 with immutable source contract

**DRAFT, RESEARCH ONLY, no production writes and no Jessica regular-chat tests.**

The S8 parallel hypothesis selector demonstrably collapsed to **GLOBAL** even after decreasing in-sample loss on twelve S4 examples. Repeating the same twelve examples is not meaningful evidence of adaptive inference. S9 therefore expands the explicit **TRAIN-only** source, without suggesting any language competence or evaluating a reserved holdout.

## What is physically prepared

Twelve hand-specified lexical families, each transformed into six controlled, linguistically tagged Russian sentence patterns = **72 NEW TRAIN examples**, completely disjoint from the twelve previously trained on in S4:

- active SVO agent / action / recipient / object;
- fronted object with changed order but equivalent semantic roles;
- recipient subject receiving an object from a source (nsubj is NOT agent);
- passive with nsubj:pass patient, agent instrumental and recipient dative;
- adjective–noun attachment with an object before the subject;
- adverbial time expression, a new ADV relation head.

Each token has explicit surface and lemma, UPOS, morph features, UTF-8 codepoint offsets, UD 1-based dependency head / relation and semantic role (AGENT, PATIENT, RECIPIENT or SOURCE). The text and all gold labels belong to optimizer-side TRAIN. Zero coreference, multi-hop, negation or broad grammatical coverage is claimed. The data is **formulaic/controlled**, not a natural-language benchmark.

The source is encoded in a strict canonical JSON and SHA256-preregistered by CI. At least one old S4 line or any duplicated or altered annotation causes a hard rejection. The new true-Calibri fast-tokenizer interface requires exact chat-template prefix and matching original prefill IDs, disallows BPE tokens crossing two UD words, and never supplies gold roles to Calibri's frozen 43M model. CI uses only a **mock char fast tokenizer** to validate the contract: **no genuine original Calibri layer8 capture or 5bit weight training is performed in S9**.

## Evaluation discipline

The five previously consumed independent EVAL128 tests and **the reserved sixth S3CI independent EVAL128 remain untouched**. No S9 heldout test is generated, inspected, or scored in this stage. A future genuinely independent Russian holdout must be separately preregistered **before training** and never sampled from these S9 TRAIN lexical families; lexical-family leakage and templated distribution similarity must be considered. This prerequisite remains open.

## Next meaningful work

S10: obtain genuine frozen Calibri 43M CPU S93 post-GT layer-8 hidden320 for the new SHA-pinned 72 TRAIN texts (never gold); verify original 616 J7 unchanged. Then design a small **physically accounted** trainable branch router inside NRA3 and train it with balanced supervision on diverse examples. The existing S8 static gate is rejected; neither zero-gate nor old accepted skills may be changed until a truly new heldout evaluation demonstrates quality gains and the whole model meets physical <=0.7 bpp and CPU >=90%.

No main/LIVE/Render/Supabase/APK changes; no model acceptance. output_gate=0, candidate_accepted=false.
