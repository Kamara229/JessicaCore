# NRA2 S3F: token-level attention for real frozen Calibri hidden state

## Experiment predeclared before OOD16 scoring

S3D #43506534289: 32/32 exact TRAIN graphs, 0/16 exact
six-node untrained graph; NRA frozen graph-oracle 16/16.
S3E #43512049506 captures full token-level 320D readouts
from the genuine frozen 43M CPU layer12 (not logits or synthetic features).
Its output is pinned physically, and is not editable as a side effect.

In S3F, train only a 32-width token cross-attention graph bridge:
six learned node queries, four attention heads, bilinear directed-edge
head, one shared source and target head. No hidden graph labels passed to
the bridge. Frozen 2,241-parameter independent graph NRA applies 5-step
relational message passing to the soft graph emitted by the bridge.
Exactly 240 AdamW optimization steps on only first 32 TRAIN examples,
plus TRAIN-token-only per-feature normalization. Original Calibri and
Green Tunnel not loaded here, and output gate exactly zero.

Save FP16 bridge weights plus TRAIN normalizer; physically reload
**before** the single exploratory score on 16 S3C/S3D OOD six-node
examples. Record exact graph, directed-edge F1, source/target index,
NRA reachability, and frozen-NRA graph-oracle 16/16. Compare with
S3D fixed prior OOD16: 0 exact graphs and 8 reachability.

IMPORTANT: the OOD16 was already observed by S3D, and S3F's input
architecture was selected after seeing S3D's failure. So even an
improvement cannot prove independent generalization. It is strictly
exploratory and needs a *fresh, predeclared* heldout split with 4/5-node
matched lengths, 6-node longer graphs, and disjoint phrasings to isolate
causes. This experiment uses 32 examples, which is extremely limited
for text-token relation extraction; poor generalization does not prove
S3E token hidden states lack relational information.

**Never accept candidate** on CI success or loss reduction: active
decoder gains, all-in bpp<=0.7, CPU speed>=0.9 and protected
584/584 + new full+EOS still require separate native decoder testing.
No tests in Jessica ordinary chat, no LIVE, Render, Supabase, APK,
main, production, or sealed QA/TEST changes.

## Verified prerequisite (2026-11-14)

S3E read-only CI #43512049506 completed SUCCESS. Artifact
`nra2-stage3e-real43M-48graph-tokens320-616J7` is physically
preserved. The successful run reported bitwise-equal 48 final vectors
and graph labels, 456+128 unchanged full+EOS baseline, 616 unchanged
original generated-token SHA digests, and no trained candidate.
S3F must never use the previously failed S3E CI #43509178812.
