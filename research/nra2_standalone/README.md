# NRA 2.0 — independent relational core, exploratory v0

**Isolation:** Research only. No imports from Calibri 2.0, no shared neural weights, no LIVE Jessica, Render, Supabase, APK, or original NRA S195–S205 changes. This branch contains no integration hooks.

## Function under test

Learn whether a directed chain establishes a transitive left_of(source, target) relation, rather than memorize explicit pairs. Each synthetic example consists of randomly relabeled graph nodes, directed edges, and a source/target query. A trainable GRU message-passing area learns to propagate relations. Inference receives **no answer, label, or example ID**. This first task tests only one narrow capability; it does not establish natural-language understanding, memory storage, or general reasoning.

## Independent curriculum

- TRAIN: chains with 5 nodes; balanced positive/negative queries separated by 1, 2, 3, or 4 edges. 300 actual CPU gradient updates, mini-batch 96, frozen seed.
- Exploratory development: new independently generated chains with 6 nodes and **5-edge** transitive relation, never included in training. Balanced negative reversed questions. Metric: full binary decision, correct/total, split by label.
- Goal for first-stage development qualification: at least 98% TRAIN-distribution evaluation and 95% five-hop development accuracy **after physical FP16 save/reload**. Failure stays recorded; no fake PASS.
- Limits: synthetic graph family only; development is not sealed QA, not evidence of broad generalization. A future **independent, untouched** test set with distractors, non-chain graphs, and altered relation templates is required before claiming robust generalization.

## Interfaces and growth

RelationalNRA(edges[B,N,N], source[B,N], target[B,N]) -> logits[B] is independent from the 12-layer Transformer. Current capacity is width 16, five recurrent propagation steps. Both width and propagation depth may grow in **separate capacity experiments**, each with fresh training, ablation, FP16 checkpoint proof, and independent evaluation. Do not count untrained growth as improved model capacity.

## Planned phases, not performed here

1. Complete independent neural specialization and proper held-out task families.
2. Build a **read-only adapter** for frozen 320-dim Calibri activations; train NRA separately on Calibri-derived training representations, never on QA or TEST.
3. Compare one read/write integration with 6-to-12 or multi-stage alternatives; start with zero-impact gate and require 584/584 strict J7 plus at least one genuinely new full answer with EOS, CPU at least 90% baseline, all-inclusive conservative bpp at most 0.7.
4. Only after evidence consider gradual joint training and scoped growth. No implicit promotion or LIVE release.

Run on CPU with:
- python -m unittest -v research.nra2_standalone.test_experiment
- python -m research.nra2_standalone.experiment --out /tmp/nra2-first
