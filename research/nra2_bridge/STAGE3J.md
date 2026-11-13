# NRA2 S3J — explicit entity/token binding before more optimizer steps

## Independent observed evidence (S3I, 2026-11-14)

S3H successful read-only frozen CPU capture #43517070838:
[128,85,320] real 43M layer12 FP16 tokens, physical reload verified,
S194 original protected 616 token generation SHA, 456/456 + 128/128
full EOS, gate=0 and unchanged core/Green Tunnel/NRA. The successful
artifact is `nra2-stage3h-real43M-128factorial-tokens320-616J7`.

S3I GitHub Actions #43517148959 **successfully ran**, physical
FP16 learned bridge SHA 800d7f751c768dac315b962300e58af90c72dc46c4612647fa6825be46cb8481.
Optimizer had exactly 64 TRAIN and all four CTRL16 scores were
committed only after physical checkpoint. Observed:

| S3I split | exact graph | directed edge F1 | frozen NRA reachability | source | target |
| --- | ---: | ---: | ---: | ---: | ---: |
| TRAIN64 | 3/64 | .6351 | 50/64 | 64/64 | 64/64 |
| CTRL_MATCHED16 | 0/16 | .3515 | 8/16 | 5/16 | 6/16 |
| CTRL_WORDING16 | 0/16 | .2486 | 8/16 | 8/16 | 7/16 |
| CTRL_LENGTH16 | 0/16 | .2650 | 8/16 | 6/16 | 8/16 |
| CTRL_BOTH16 | 0/16 | .2451 | 8/16 | 2/16 | 6/16 |

Frozen independent NRA on **oracle graph inputs** was correct on all
64/64 TRAIN and four 16/16 controls (simple reachability only).
S3I initial TRAIN loss 4.4492, final 0.7865; **this was not
semantic generalization**. No new decoder full+EOS gain, all-in
physical bpp or paired CPU speed has been demonstrated. Candidate
rejected; no integration or model acceptance.

## Specific failure hypothesis

Prior learned six static node queries pool 320D contextual token
states, but have no hard requirement to represent *named nodes* R0...R5.
The model learns source/target on TRAIN but hallucinates many edges
(208 FP vs 201 TP on TRAIN) and collapses on all four OOD controls.
This is a **hypothesis**, not proof that token bindings alone solve it.

## S3J token-identity prerequisite — verified

S3J native tokenizer **preflight** GitHub Actions **#43518808282 SUCCESS**.
It uses the original pinned `HuggingFaceTB/SmolLM2-135M-Instruct`
tokenizer revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`
and its exact `apply_chat_template` + `encode(add_special_tokens=False)`
contract. All 128 S3G prompts are now mapped from literal
R0..R5 node mentions to actual token offsets, including multiple
occurrences per entity, with token-order and mask tests. This uses
zero graph labels, zero 43M model weights and does not predict
adjacency/reachability. Importantly **this is only tokenizer plumbing**:
no learned semantic binding, no heldout accuracy improvement, and
no new FP16 candidate is demonstrated. Do not confuse S3J offset CI
success with the rejected S3I neural adapter.

## Next falsifiable experiment — predeclare BEFORE observing scores

Do not repeatedly tune S3I on already examined CTRL16. Design a
new **S3J** research protocol that aligns named entities to token
positions without using ground-truth adjacency or answers at
inference. The first work item is to obtain a reproducible
text-to-token-span alignment from the **same frozen tokenizer** used
for the 43M prefill, recording all occurrences of each node name
and edge connector. A node ID is not assumed to correspond to
one token, and whitespace is not guessed.

Proposed architecture:
1. Read genuine frozen full-token [T,320] plus masks and tokenizer
   alignment for each prompt, never labels or gold answer tokens.
2. Compute a *node-specific* learned pooled representation from
   contextual states at **all mention spans** for each of R0...R5,
   with explicit slot present masks. Resolve repeated mentions
   with role-neutral pooling; **do not infer graph edges from query
   source/target**.
3. Predict directed adjacency from pairs of node representations,
   conditioning on connector/context spans, and source/target query
   roles from their independent spans. Only forward-pass label-free
   text/spans may enter the bridge.
4. Pass differentiable graph, src, dst into frozen 2241-param NRA;
   compare exact edges and oracle first, then actual reachability.
5. Introduce **counterbalanced graph/question pairs**: identical
   edge descriptions with different (src,dst) queries and opposite
   labels, so shortcuts from query direction or “yes/no” word choice
   cannot prove graph parsing. New heldout **graph topologies** must
   remain disjoint across train and validation, even for paired
   questions within a topology.
6. Test prompt/order invariance (shuffle edge serialization) and
   node renaming; heldout graph topology and novel wording are separate
   axes. Keep a fresh, unviewed final split after all architecture
   decisions; prior S3F and S3I controls are exploratory only.

Safety gate: original 616 CPU J7 SHA; all 584 protected answers;
unchanged Calibri/Green Tunnel/standalone NRA; output gate=0;
physical FP16 model+normalizer reload before control scoring.
If independent novel full+EOS eventually improves, only then
measure packed **all-in** bpp <=0.7 and same-CPU speed >=0.90x.
Never modify LIVE Jessica, Render, Supabase, APK, main, QA/TEST.

**Status:** SPEC ONLY. No S3J training run, no new performance
claims, no release candidate. This document intentionally avoids
tuning on S3I's exposed controls.
