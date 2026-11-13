# NRA2 S3Q — preregister independent soft/hard check, not a retroactive DEV fix

## Confirmed observation

S3P CI #43524516584 (success) reused unchanged physical S3O FP16
SHA `40a8903d5419cd8641b5d01e75e429e10b449304b3ba140d2245bf7f8a9d00b6`,
original S3N CPU 192 token features and original frozen 2241p NRA.
On the **already inspected** S3M DEV64, each of 6 missed cases
(indices 33, 35, 36, 38, 44, 46) was a **false-positive on a
cycle_sink graph**. All 64 predicted graphs and endpoints were
otherwise discretely correct. Soft edge probabilities: **58/64** NRA;
hard edges using >0.5: **64/64**, regardless of source or
destination soft/hard. Soft endpoints were not causal.
Non-edge probabilities around 0.02077 changed NRA logits even
though all edges satisfied the usual classification threshold.

This is a **post-hoc diagnostic**, not independent evidence or
an accepted model improvement. Do not select hardening on DEV alone.

## S3Q locked BEFORE future native CPU/model scoring

- **16 newly generated directed topologies**, disjoint from S3G
  and S3M, four families (chain, fork, cycle with directed sink,
  diamond); 64 pairwise role-balanced questions; 32 positive/32
  negative; 4/5/6 nodes. Unique prompts and topologies have SHA256.
- Independent 43M CPU state acquisition must use original
  S194 frozen CPU source, maintain original 616 J7 hashes and
  protected 584 full+EOS, and physically verify [64,T,320] FP16.
- **No training, finetuning or parameter updates**: exact S3O
  physical FP16 and frozen independent NRA weights/pinned SHA.
  No re-selection of normalization or tuning of probability calibration.
- Predeclare original soft edge probabilities vs **predicted hard
  edges** with strict rule `sigmoid(edge_logit) > 0.5`. Source
  and target remain **soft probabilities in both arms**.
- Use identical frozen NRA. Count full binary reachability
  correctness, stratified by family and by truth label, and graph/
  source/target agreement. Do not choose an improved threshold,
  retrain on S3Q, or inspect future labels before the source/weight
  commitment and CPU acquisition are pinned.
- S3Q uses same query templates as S3M, so evidence would show
  **graph topology transfer** not unseen language transfer.
  This is **open preregistered research**, not cryptographically
  sealed final blind QA. Future novel wording, truly separate final
  evaluation, full native decoder EOS, all-in physical bpp and
  measured CPU throughput are still needed.

## Safety

CI only checks deterministic input splitting and stores commitment.
No model, NRA or original Calibri weights are loaded or modified.
Output gate=0; no tests in ordinary Jessica, no main, LIVE, QA/TEST,
Render, Supabase or APK modifications. No acceptance claim.
