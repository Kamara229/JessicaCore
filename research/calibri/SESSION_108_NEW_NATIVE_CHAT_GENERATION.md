# v108 — independent new prompts, genuine CPU model, locked morphology

Jessica ordinary-chat relay #c3b7e886-93c0-46ca-9cee-ff880ed9ffdf
returned server_processed=false. Jessica's answer that it lacked access
to v103/v104 artifacts and v107 source was transparent; **no actual test
ran through Jessica**. It must not be marked as a failure of the
detector, as a pass, or as scientific evidence for the model.

GitHub research Actions #41750015244 completed v107 source-locked
cross-run **historical** observable results TP=12, FP=0, TN=6, FN=0
against source-specific visually read outputs. Reviewers and detector
developer had previously seen source data. These counts are **not**
independent blinded validation.

v108 tests a fresh, fixed **precommitted** set of nine original
chat-template messages (six factual/math with gold, three open-ended).
No prompts overlap v96, v103, v104. Execute 32 actual KV-cached
greedy tokens from fully original FP32, physical locked CB59 native C,
and physical same-byte v102 native C on identical prefixes.

Keep the v106 fixed character 5-gram and compressed-byte thresholds
unchanged. Write full exact generated strings, 32 output token IDs,
source hashes, first-response boundaries, fake role lines, source
ground-truth and crude gold-presence signals for later reviewer audit.
Never infer coherent text from distinct-token counts, or factual
correctness from a token appearing in a hallucinated sentence. The
original FP32 reference can be wrong. A frozen morphology flag is a
**repetition diagnostic**, not a semantic-grading model.

v102 source SHA is checked against the completed original historical
v102 Actions artifact, not regenerated on these canary prompts. Frozen
CB59 source bytes remain 11,229,303 and the paid whole-model archive
rate 0.667839 bpp <= 0.7. The frozen official v89 TEST stays closed;
no model or detector fitting, source-weight changes, candidate
selection, production deployment, Render, Supabase, APK or Jessica Live.

The experiment produces NEW outputs, which are eligible for separate
human linguistic and factual annotation. Until that review, v108
reports alarm counts WITHOUT sensitivity/specificity, accuracy claims,
or quality pass. Facades v108/spec, generation, pipeline, launcher,
unit guard, isolated GitHub Actions. Avoid sending a Jessica-chat
request for local source files it does not possess.
