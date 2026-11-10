# CALIBRI S122 — independently varying two-token owner labels (TRAIN only)

## Purpose and boundaries

S101 compressed 36M baseline retains 353/456 originally correct complete+EOS outputs and 387/456 initial tokens. Its S82 new-TRAIN family uses 24 words drawn from 40 existing S82 noun candidates and four originally TRAIN color names and owner/text frames. The S82 code constructs two owner labels of the **SAME color**, while modifying the noun association. Thus the historical *TRAIN data generator* omits combinations where the two owners carry unrelated colors and nouns. This coverage gap is observable from the original S82 training source **without using S118**.

S118 (the first, once-only new 128 QA holdout) evaluated S115 negative: 30 full vs baseline S101 31. The S118 question frames, names, nouns, data, examples, and outcomes are **permanently embargoed as training/evaluation input**, and are NOT read or imported by this branch. None of the original old QA/TEST is read either. S122 is a **new TRAIN curriculum only**, not a QA benchmark or a proposed model improvement. We may develop on the new TRAIN data, but all improvements require later separate, frozen-model, preregistered, once-only independent holdout.

## Exactly precommitted design, no search based on inference

New data are derived exclusively from previously permitted S82 TRAIN vocabulary: precisely 24 original TRAIN nouns, all four original TRAIN colors from S30, all four original S82 TRAIN frames, all four original S82 TRAIN pairs of owner names. No new names/frames are borrowed from other studies or one-shot QA.

Define 64 new counterfactual TRAIN pairs. For each pair i (0..63):

- Pick one of the four S82 historical TRAIN frames and its original owner name pair using i modulo 4.
- Independently choose the two different colors, deterministically cycling through all 12 ordered distinct color pair combinations.
- Choose two different words using a fixed deterministic permutation over the original 24 TRAIN nouns.
- Write question for owner A, then the opposite counterfactual after swapping A's and B's labels. Each correct target is a two-token phrase + EOS; each phrase must exactly match original tokenizer semantics.
- Check all 128 new TRAIN question SHA256 values are unique and **disjoint from the original historical 456 TRAIN prompt hashes**. Fail closed if any original source changes or collision occurs.
- Output ONLY summary counts and the digest of the generated TRAIN question hash set. Do not save raw question strings, gold tokens, per-case IDs or any inference-time oracle lookup.

The initial implementation only authors and independently validates this new TRAIN material using the real original tokenizer. **No GPU training, no new weights, no additional QA.** Once validated, a separate explicitly predeclared future experiment may train the original S111 340-parameter dual donor conditional Green Tunnel on these new original-TRAIN-derived counterfactual pairs, using strict whole original 456 exact greedy full+EOS retention, no first-token forgetting, FP16 physical <=0.7 bpp, >=0.9 original S101 CPU speed, and independent untouched once-only QA after selecting and freezing one candidate.

## Evidence so far

S119: three GEM projected 340D S95 prefill-only proposals 0.015/0.030/0.060 all forgot 1/2/4 old complete TRAIN outputs. S120: eight historical whole TRAIN pair gradient proposals likewise forgot 1/2/3 old complete; none eligible. S121 investigates stronger protected old-gradient nullspace; no claim of success unless its real physical 456 guard passes.

These results suggest existing 456 historical TRAIN proxy gradients do not yet translate to stable output preservation and novel transfer. S122 targets a **different causal hypothesis: independent attribute-binding TRAIN coverage**, not more aggressive optimization against the inspected S118 holdout.

All work stays in separate GitHub research branch. LIVE Jessica, Render, Supabase, APK and main remain untouched.
