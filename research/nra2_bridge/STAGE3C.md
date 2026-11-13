# CALIBRI 2.0 NRA2 — S3C physical real hidden-to-graph pairing

## Why S3B was not sufficient
PR #616 / CI #43484392072 proves a differentiable shadow 320D-to-2-edge
adapter can be trained on 24 source-derived proxies and evaluated on eight
untouched examples (7/8). But those S190 item/number prompts have no
independently grounded relational graph annotation. The 7/8 is **not**
a graph-language understanding result and the protected 584 were **not**
replayed in S3B. Zero output gate is still essential.

## S3C independent paired graph data (no adapter training)

- Based on the actual **frozen 43M CPU decoder**, reuse entire pinned S194
  full 456+128+32 real generated-token J7. All original 616 produced token
  SHA digests must match S194 bit-for-bit (S3A remains pinned as provenance).
- Install a second read-only layer-12 port without output injection. After
  the original S190 32 TRAIN probes, take 48 additional original-model S93
  CPU prefills, never passing a graph label or target answer to the model.
- Precisely described edge statements R0 -> R1, source node and target
  node. Each prompt has its own independent directed-chain adjacency,
  source, target and deterministic reachability target, with both positive
  and reversed-direction negative queries.
- **32 TRAIN** have four or five active nodes. **16 untouched structural
  heldout** have six active nodes and different query wording templates.
  SHA and exact edge-order topologies are disjoint. These held-out data are
  not opened by an optimizer or used for model selection.
- **Only after all original CPU decoding and new prompt CPU prefills**,
  generate the independently specified relation adjacency tensors, source/
  target node vectors and binary path labels. Physically store separately
  from 48x320 actual model FP16 states; prove bitwise physical reload.
- No weights trained in S3C, no direct answer gains, no active NRA output,
  no inferred benchmark success. This is a *data acquisition/grounding*
  stage, NOT a graph generalization benchmark pass. Prompt conventions and
  node tokens could still offer shortcuts and must be controlled in later
  paired-training evaluation.

## Next gate (not executed by S3C)

Train only the hidden320-to-graph representation adapter on the 32 graph
TRAIN examples. Freeze pretrained NRA + 43M Calibri and all tunnels.
Evaluate complete graph reconstruction and binary reachability *only*
on the 16 untouched longer-hop structures against explicit trivial-label
and keyword-baseline controls. Retain separate class balance and edge
permutation probes. A semantic claim requires real measured results
better than shortcuts, not just a falling training loss.

Only after shadow pass can a **new** nonzero output-gate candidate be
considered, with physically saved weights, independent full 616 J7,
no lost 584, new correct full+EOS, same CPU ratio >=0.9, all-in bpp <=0.7.
Research branch only: NO LIVE Jessica, Render, Supabase, APK or main edits.
