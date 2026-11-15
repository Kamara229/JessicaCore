# NRA2 3.0 S15 — first source-only SHA prereg of NEW Russian24 EVAL (after S14 weights locked)

**RESEARCH ONLY / DRAFT / source-only / ZERO candidate promotion.**

S14 PR #724 completed original physical native-C + real43M CPU TRAIN-internal study and *repeated the same exact saved FP16 bytes from scratch* in GitHub Actions #43740072530 then #43740415883 SUCCESS. S14 physically frozen **2,581-byte rank8 `upos_locked`** selection SHA: `5b96e2174e33da5232fcc2a4a397b6fab4370b6879adb684d75fea8a503f1279`, original S8b 43 CB59 native-C matrices SHA `d710aed710a44719b6003434a2dd83150861928ce167364cb92329beabf7be97` and original S11 39-FP16 six-cycle router SHA `05ce14e20c9f4e7e09f85dbd9bd46f0dc0a700d72f1a96ffa921b2e822f235c3`. Last S14 head `ab95483ef5dc985b8a32941163fb248bc9640ef3`, frozen before **any** S15 evaluation source was authored.

S14 internally selected a `upos_locked` fallback preserving original parts-of-speech logits by physical zeroed UPOS residual matrix and bias. The 24 internal developer rows were previously exposed to S11; their semantic-role gain (82/136→86/136) is **not independent EVAL**.

**S15 independent one-time source registration**: 24 new full-sentence Russian constructions hand-authored post-S14 with manually specified Unicode offsets, 17-way UPOS classes, morphology, rooted dependency arcs, 9-way semantic argument roles, and six explicit within-sentence pronominal coreference pointers. Include novel names/stems, transitive transfer and agent/recipient/patient inversion, passive agent case, fronted object, adjective attachment, adverbials, semantic experiencer vs agent, instrument and destination/source, two negated past predicates, future predicates (event **NOT** yet actualized), noun/proper-noun/pronoun distinctions and two-clause anaphora. No event is counted as having occurred from mere grammatical role annotation. Text identities are checked for exact SHA non-overlap against old original S4 12 TRAIN and S9 72 TRAIN.

Crucially **source-only**: this PR does NOT load/run any original CALIBRI43M, GT, S8b, S11 or S14 models, infer hidden, run old tests, score EVAL, compare candidate outputs, train or tune. Do not open/reuse the **already consumed** S12c EVAL16, five previously consumed original EVAL128 or reserved sixth S3CI EVAL128. This source is not a representative free-form corpus and a single 24-example score cannot prove broad Russian grammar/semantic competence.

First CI action checks that S14 previous frozen successful SHA precedent exists, then **commits SHA256 of complete 24-row text + UPOS + morph + UD+semantic+antecedent+event labels** before any model sees prompts. The first success SHA is hard-pinned in a **separate commit** and tested in a second source-only CI. Any later score shall use ONLY that immutable corpus and the **already frozen S14 physical bytes** in a new branch and shall not use EVAL24 for model selection/retraining. Hard-target whole parent+GT+NRA **<=0.7 bpp, >=90% CB59 original CPU, protected active 616 J7, full response EOS** remain unproven. Standalone raw S14 area+FP16 0.859934 bpp remains above 0.7. Source-only CI success never promotes an unqualified candidate.

No main, LIVE Jessica, regular Jessica chat tests, Render/Supabase/APK modifications. output_gate=0 candidate_accepted=false activation_allowed=false.


### Verified first full-source prereg SHA and strict permanent pin

GitHub Actions **#43741182553 SUCCESS** (3/3 S15 source-only contract tests) generated the complete **24 new UD + semantic + event truth + coreference** SHA `640bdfb0d817d1467eaf308fc1ba5dd9043361721af293b9666d89f4a2c7276d` before **any** S15 model evaluation, artifact **#14172506310**. The canonical entire-source SHA is now hardcoded into the source and CI with a positive exact-equality guard; a second source-only CI must reproduce it. The first prereg is model-free, not an EVAL model quality claim. Never train/retune using these 24 cases, or re-open the consumed S12c EVAL16.
