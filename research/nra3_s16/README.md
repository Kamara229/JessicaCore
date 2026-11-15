# NRA2 3.0 S16 — directed predicate → argument / agent–action–object mechanism (TRAIN ONLY)

## Why

S15c one-time independent 24-Russian-sentence original43M/GT CPU + true packed-CB59 test passed **technically** (CI #43743397331) but S14 **did not improve semantic role accuracy**: 79/152 → 79/152. UPOS remained exactly 74/152, and past-observable dependency heads changed only 46/74 → 47/74. Roles on asserted statements improved by 4 but those for future and negation deteriorated by 2 each, net zero. The 24 EVAL are now **consumed** and must never be viewed by S16 training or repeatedly rescored.

S14 predicted role labels for individual tokens; this does not explicitly encode **WHO DID WHAT TO WHOM** and does not represent a predicate-to-argument link. S16 introduces an explicit, directed, causal relation operator rather than more tuning of role classification on spent holdouts.

## S16 concrete architecture

A new rank8 source makes two physical projections of a frozen original NRA3 relation state (size 128 each): one maps **predicate** states and another maps **argument** states. There are **four separate directed role-bilinear matrices**, each scoring a predicate-to-argument edge: `AGENT`, `PATIENT`, `RECIPIENT`, `SOURCE`, plus `NONE` class bias. The prediction is a *directional relation* `predicate → argument`, not a bag of overlapping roles. Predicate's position in the original sentence does **not** change edge direction:

- If the verb occurs after the subject, the currently observed predicate may link to the earlier agent (`predicate_now`).
- If the verb occurs before the object, the currently observed argument may link back to the earlier predicate (`argument_now`).

Neither pathway may use any wordpiece at a future position, and both exclude diagonal/self edges or padded positions. This is a strict online causal representation; agent labels for subjects preceding a verb cannot be assigned as a completed predicate link until that verb is actually observed. Passive subjects may be **PATIENT** and an ablative/from argument may be **SOURCE**, not incorrectly treated as an AGENT. This stage does **not** determine whether a negated event actually happened, resolve pronouns, or generate answers.

All new neural weights are physically accounted: `128×8` predicate projection + `128×8` argument projection = **2,048 FP16**, four `8×8` role matrices = **256 FP16**, and five biases = **5 FP16**. Exactly **2,309 new parameters / 4,665 bytes** including versioned header and full SHA256, distinct from the untouched original 43 physical native-C CB59 5-bit matrices, S11's 39param FP16 selector, Calibri43M checkpoint and Green Tunnel. The new rank8 is FP16, **NOT** claimed 5-bit. All future CPU and full inclusive bpp measurements must charge these 4,665 bytes.

## S16 research-only TRAIN preparation

Use **ONLY** preexisting S9 Russian 72 TRAIN sentences: 12 lexical families × six constructions, canonical SHA `b09d60a77a17d30c2dbf846e40ab97860def8635e2bf31a5b0ab8981832b744d`. Construct gold predicate→argument directed edges from each annotated word's UD dependency head and roles after original native CPU outputs, never pass gold to inference. Each gold record explicitly has predicate index, argument index, role, earliest causal time when both words exist, previous word, and orientation; fail closed on any future-token link. Initial S16 source-only CI checks the complete 72-row directed graph gold SHA, physical roundtrip, no leakage/overwriting, and no activation.

No S16 real physical 43M training, EVAL, or quality improvement is claimed from source-only CI. **Next distinct S16b** may train this operator in research GitHub Actions on the original SHA-pinned S10 Calibri43M/GT 72 TRAIN hidden, fed through unchanged 43 CB59 native-C matrices and frozen S11 39param router. Physically bill 2,309 FP16 params, report actual directed **edge precision/recall/F1** and hard orientation/role confusion **only in TRAIN**, and compare to frozen predicted-role heuristic baseline. Never use S15c EVAL24 or S12c EVAL16. Only after S16 weights frozen could a new independent EVAL be pre-registered; avoid claiming broad language understanding on tiny synthetic tests.

**LIVE/normal Jessica chat tests, Render, Supabase, APK, main, original Calibri/GT, protected J7** all untouched; `output_gate=0`, `candidate_accepted=false` and no all-in <=0.7 bpp or >=90% CPU quality certification.
