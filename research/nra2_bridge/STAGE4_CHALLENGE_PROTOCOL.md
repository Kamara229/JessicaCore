# CALIBRI 2.0 · Stage 4: harder tasks after successful adapter qualification

**Status: preregistration, NOT model testing or a claim of success.**
This research branch depends on NRA2 S3F. No Calibri, Green Tunnel, NRA,
Jessica chat, LIVE, Render, Supabase, APK, main, QA, or TEST is changed.
The existing S194/S3E 616-generated-token J7 and protected 584/584
remain the acceptance baseline. The 48 JSONL examples are **PUBLIC
illustrations**: their answers are in the repository. They must NOT be
called blind evaluation or used as evidence of generalization.

## Sequence of gates

1. **S3E** must physically save and reload genuine last-decoder-layer
   [48,T,320] FP16 tokens, masks and prompt indices, preserve all 616
   original generated-token hashes, with true numerical-vs-bitwise
   cross-run FP16 audit. Never weaken a failed SHA or label check.
2. **S3F** must freeze the 43M decoder, Green Tunnel and existing 2241p
   NRA; train a token-to-directed-graph bridge on exactly 32 declared
   TRAIN examples; physically reload FP16 before any heldout scoring.
   Improvement on the reused 16 examples is **exploratory only**.
3. **Adapter advancement (not automatic acceptance):** register a fresh,
   structure- and phrasing-disjoint 96-example assessment, never used
   to tune S3F. Report graph exact-match, source/target accuracy, edge
   F1, and NRA reachability; verify that the oracle-input frozen NRA
   can solve each family before attributing a miss to the adapter.
   Repeat with independent seeds and a majority-class/template baseline.
   Failure to generalize blocks integration.
4. Only after a verified adapter gain, extend to **Stage4 harder tasks**.
   Predeclare 8 families; prepare 96 development examples (12 per family)
   and 192 *reserved* final examples (24 per family) by independently
   generated graph structures, node permutations, question wordings,
   and output lengths. The reserved set must not guide fitting,
   hyperparameter selection, data augmentation or prompt design.
   A repository-visible generator is not a cryptographically sealed
   external blind evaluation: use an isolated independent custodian for
   a genuine blind test, if a generalization claim is required.

## Eight more demanding task families

| Family | Added difficulty | Scoring |
| --- | --- | --- |
| Branching | Alternative routes and distractor edges | Exact yes/no, path |
| Cycle | Directed loops without infinite traversal | Exact yes/no |
| Reversed direction | Same network, inverse source/target | Exact yes/no |
| Counterfactual edge | Delete one link and insert another | Exact yes/no |
| Forbidden node | Reachability subject to a prohibited intermediate | Exact yes/no |
| Two questions | Two simultaneous independent reachability answers | Exact two words + EOS |
| Conditional edge | Activate or deactivate an edge using a flag | Exact yes/no |
| Weighted shortest path | Minimum directed path cost; inaccessible -> none | Integer or none + EOS |

The first three families mostly probe graph extraction and multi-hop
reachability, though cycles need an oracle check. The next four require
intervention, constraint or compositional processing **before** frozen
NRA reasoning; failures cannot automatically be blamed on the bridge.
The weighted family is a *new computational skill*, not evidence that
the current 2241-parameter reachability NRA is defective. Score it as
a separate Calibri capability, with a suitable graph-algorithm oracle.

## Evaluation protocol

- Report per family, path length, positive/negative label, graph
  structure, phrasing, and token-output length; macro-average across
  families, never hide a failure under one aggregate number.
- Freeze all candidate weights, adapter config, tokenizer, prompts and
  threshold **before** touching reserved cases. Save a physical model
  SHA. Avoid exact-answer/template and question-hash lookup.
- Independent algorithms build answer keys *after* frozen input capture,
  never passing gold, full answer, expected label or prompt SHA into
  inference/routing. Verify every generated key with an independent
  implementation.
- Require full generation ending in the expected EOS, not merely a
  correct first token or decreasing TRAIN loss. Distinguish graph
  correctness from text response correctness.
- Use matched graph-only oracle vs input-bridge tests to localize:
  (a) representation extraction, (b) adapter entity binding,
  (c) graph reasoning, (d) answer decoding. Record abstentions.
- Do NOT integrate into LIVE until candidate all-in physically charged
  bpp <= 0.7, native same-CPU speed >= 0.9 of reference, protected
  584/584 unchanged, original 616 hashes unchanged where applicable,
  and genuine *new* full+EOS gains verified without forgetting.

## Examples in repository

`stage4_hard_public_examples.jsonl`: 48 reproducible questions,
6 per family, with independently derived answer keys. This is a
**smoke/calibration fixture only**, not validation or a blind test.
`test_stage4_public_examples.py` checks each answer via independent
BFS or Dijkstra, asserts directed cycles in the cycle family,
balances yes/no, and prevents mislabeled blind evidence.

**Do not run these examples on the candidate until S3F succeeds;**
reserve new structures and natural paraphrases for the later evaluation.
