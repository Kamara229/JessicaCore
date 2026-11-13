# CALIBRI NRA2 S3E: isolate the information-loss hypothesis

S3D #43506534289 proved overfitting: TRAIN32 32/32 graph exact,
untouched longer/new-phrasing OOD16 0/16 exact graph, 8/16 reachability.
The frozen independent 2241p NRA can solve correct oracle graphs 16/16;
it is not yet known if one final hidden320 retains the source entities
and link order. Do not claim adding more adapters, parameters or hops
will necessarily fix the feature bottleneck.

**S3E performs data capture only**, with no training or new inference:

- Reuse exact 48 S3C synthetic graph prompts (32 TRAIN/16 unseen 6-node).
- Load the exact frozen S194 43M CPU model and run its native 616-token J7
  before graph-specific read-only prefills, preserving all generation digests.
- Add a second read-only hook on decoder layer index 11 that saves the
  entire genuine contextual FP16 [T,320] output, not just its final row.
  Masks and sequence lengths are physically saved, with no graph gold,
  answer or feature edits inside model hooks.
- Verify final row against S3C #43486292082 within two adjacent FP16
  representable steps (NOT cross-run bitwise: record changed rows and max delta),
  graph labels match bitwise, 616 original outputs match S194, NRA frozen,
  zero gate, and physical FP16 and masks reload exactly.
- Capture is **not learning** and cannot establish a useful increase in
  relational reasoning. Do not use OOD16 labels to select any model or
  architecture after looking at its results. Any attention adapter should
  be specified before the heldout is evaluated once.
- Read-only research branch only; no LIVE Jessica, Render, Supabase, APK,
  production, main or sealed QA/TEST.

Next: separately defined small token-level attention bridge with 32 TRAIN
and an independent evaluation. Explicitly report exact-graph accuracy,
source/target accuracy, NRA reachability and unchanged S3D control.
Improvement on these 16 is only exploratory: the same OOD16 was already
examined by S3D, so a truly independent final control must be gathered
before claiming generalization.
