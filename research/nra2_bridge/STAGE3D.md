# NRA2 S3D — real frozen Calibri hidden320 to graph adapter: first structural holdout

S3C corrected run #43486292082 **SUCCESS**, 48 physically saved genuine
43M CPU hidden320 vectors; 32 declared 4/5-node TRAIN, 16 independently
annotated six-node structurally novel probes. Its full original
616 S194 CPU outputs had identical digests, 456/456 + 128/128 protected
and 0/32 new full+EOS. No active NRA gate. This is pinned input evidence.

## Shadow-only measurable S3D

- Source: exact S3C FP16 archive #43486292082 (not earlier S3C run).
- Freeze pretrained 2,241 parameter relational NRA, Calibri 43M and Green
  Tunnel (core not even loaded for S3D); gate remains exactly zero.
- Training: physically decoded 320D hidden, 32 TRAIN only, declared AdamW
  240 steps on 6x6 directed-edge logits, 6-source and 6-target logits and
  frozen NRA reachability with 4/5-node graphs. Normalization entirely from
  TRAIN. This is a small 24-width bridge with significant overfit risk.
- Save bridge + normalization in physical FP16, bitwise reload. Evaluate the
  physically reloaded model on original 32 and *exactly once* on 16 six-node
  untrained structures and new prompt templates. No selection or tuning from
  the heldout. Graph oracle on true labels must score 16/16 to confirm the
  separately trained frozen NRA understands the graph-format input.
- Report all separate metrics: exact full adjacency matches, source and
  target identity, directed-edge F1, and frozen NRA reachability. A simple
  class-majority baseline scores 50% by construction. Do NOT treat
  TRAIN accuracy or graph oracle accuracy as Calibri semantic ability.
- CI **success means reproducible experiment artifacts, not acceptance**.
  The experiment deliberately records a negative structural-generalization
  result if accuracy is low. Real semantic success requires reliable complete
  heldout graphs and endpoints plus reachability, far above trivial baselines.

## Acceptance remains out of scope

No active Calibri generation test, no improvement in 32 new complete+EOS
answers, and no physically inclusive bpp or paired same-CPU speed measured
by S3D. Any future active candidate must pass independent original
584/584 retention and gain true full+EOS answers with bpp <=0.7 and
same-CPU speed ratio >=0.9. There is no merge or deployment permission.
Never touch LIVE Jessica, Render, Supabase, APK, main or sealed QA/TEST.
