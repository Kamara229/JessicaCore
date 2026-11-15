# NRA2 3.0 S7 — three parallel semantic-relation hypotheses, one CB59 model

**Research-only. No LIVE or main changes. Zero output gate and no candidate promotion.**

S7 integrates **three concurrent, structurally distinct inference hypotheses inside the existing unified NRA3 relation layer**, inspired by CALIBRI S159 source-bound parallel slots and S177 adaptive candidate comparison. This is *not* three copies of CALIBRI, and not a stochastic 5/10-vote correctness oracle. The three alternatives attend over a single current-prefix hidden sequence:

1. **Global causal**: long-range entity/relation evidence.
2. **Local causal**: only the previous 4 token positions.
3. **Predicted semantic-role complementarity causal**: affinity between *model-predicted* AGENT / PATIENT / RECIPIENT / SOURCE / DESTINATION / INSTRUMENT / EXPERIENCER / LOCATION roles, without gold roles or label lookup at inference.

Their messages are evaluated in one vectorized operation. An adaptive, differentiable **agreement gate** uses the existing CB59 `rel_q` projection to compare the alternatives and sends the weighted mixture to **one existing GRUCell update**. No new learned matrices, no duplicated GRU rounds, no new operators, and **no added trainable parameters**. Original `NRA3Config.parallel_hypotheses=1` preserves previous S6 execution by default; S7 explicitly selects `parallel_hypotheses=3`.

Both the differentiable dense TRAIN path and the original Calibri `direct_kernel_v59.c` genuine 5-bit native CPU path share the **same pure tensor hypothesis function**. Source code does not reconstruct CB59 2D weights at native inference. Optional S7 per-token hypothesis mixture weights are returned in an existing features ABI and not sent into CALIBRI. Padded tokens emit zero gates and zero hidden relations; future-token attention is masked.

## One S7 physically paid TRAIN-only study

Use only the SHA-pinned **S5 original frozen 43M post-Green-Tunnel layer8** CPU hidden320 of **12 previously registered Russian TRAIN** sentences (S5 run 43699866531) and the physically trained **S6 CB59 6,106,154-parameter** source (S6 run 43702088137, SHA `f1248a74ba238a2a5da49310d1af3a1070b41cee592fddac60028ec95fb06b20`). S7 adds a **direct supervised UD dependency-head relation loss** between word relational representations to the existing grammar/semantic objective. All gold arcs remain strictly optimizer-only. No coreference/multihop graph truth and no independent held-out evaluation. The intermediate FP32 matrix surrogate is restricted to TRAIN RAM; post-training source remains original **43 K32V8 physical 5-bit matrices**, immutable packed code indices, unchanged complete physical CB59 size and unchanged 1D original tensors. Native C CPU must reproduce the train-only loss on the physically reloaded source.

Report *only* in-sample TRAIN loss change and preliminary **NRA-only** CPU latency (one vs three hypotheses, the original 43M full pipeline excluded). Full 616 J7 after active NRA output, true whole-Calibri+Green-Tunnel+NRA CPU >=90%, complete real physical all-in <=0.7 bpp, and real unseen full-answer+EOS generalization **not** established. The prior S6 50% improvement is TRAIN fit of only 12 rows, **not** model language accuracy.

Existing consumed five EVAL128 are not reloaded; the unopened sixth S3CI EVAL128 is not read. LIVE Jessica, ordinary Jessica-chat tests, Render, Supabase, APK, baseline CALIBRI and GT unchanged.

### Next necessary gates

Expand preregistered genuine TRAIN Russian grammatical constructions (especially pronouns, multiclause, negation, passive, temporal edges), add independently separated novel EVAL designs with zero reuse of spent/held sets, train cross-entity **relational links** rather than only UD heads, and measure matched whole-pipeline CPU/bpp/J7 with output gate *still* zero before any specifically approved guarded rollout.
