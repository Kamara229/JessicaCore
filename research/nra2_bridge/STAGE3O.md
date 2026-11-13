# NRA2 S3O — learned semantic query source/target roles on new 43M CPU states

## Existing observed proof and failures

S3K #43519730567 learned token/instance-aligned directed edges on original
chain graphs; S3L #43520095966 had exact graph 16/16 across each of four
already observed research CTRL16 groups. Novel wording broke source and
destination: 7/16 (wording) and 8/16 (both), despite all graphs exact.
S3M #43521067551 added 128 TRAIN and 64 topology-disjoint DEV pairs
including chain/fork/cycle-sink/diamond and both directional queries;
S3N PR #629 native frozen 43M CPU [192,T,320] capture is a prerequisite.

## Architecturally locked before DEV evidence

- Reuse full **learned S3K** graph branch on genuine frozen 43M layer12
  token hidden, initialization SHA pinned to the original 68664-byte
  physical FP16 from CI #43519730567.
- Augment with a small **neural source/target head**:
  * find question-clause text span via original S160 tokenizer;
  * identify the two entity-name mentions by genuine offsets,
    without treating textual occurrence order as semantic roles;
  * pool contextual 320D hidden for both mentions, nine tokens to
    left/right and whole question;
  * shared MLP predicts a source-role score for each mention.
    Opposite scores determine destination. No hardcoded "from/to"
    word-to-answer or graph adjacency oracle is used.
- Train exactly **128 new S3M TRAIN prompts**, using 240 AdamW steps,
  graph LR 0.0003, role-head LR 0.002 and original S3K normalizer,
  fixed before opening any DEV outcome. Golden adjacency/source/target
  labels appear *only in training loss* for TRAIN128.
- Physically commit and reload the entire learned bridge + normalizer
  in FP16; verify SHA. Only then perform a single research readout on
  topology-disjoint **64 DEV prompts** (no gradient, no model selection).
  DEV is public research, **not an untouched independent final blind**.
- Verify that the standalone independent 2241p NRA remains frozen,
  original S194 616 J7 native CPU output hashes and 584/584 protected
  full+EOS are unchanged, and all outputs have zero LIVE gate.
- Diagnose exact graphs, endpoint source/target accuracy, predicted
  reachability and oracle true-graph NRA reachability on both splits.
  New graph families may exceed NRA reachability capabilities, so do
  not require 64/64 true-graph oracle unconditionally.

## Acceptance guard

This PR is not authorization to deploy or update LIVE. A successful
CI, falling TRAIN loss, or DEV metrics are **not** final model success.
After any meaningful topology-disjoint transfer, reserve an absolutely
fresh independent final evaluation, test actual native new decoder
FULL+EOS generations (not graph proxy), preserve every existing
answer, and physically measure all-in packed bpp <=0.7 and matched
CPU throughput >=90% of reference. Until all gates pass, the adapter
remains a shadow with output gate=0.

No writes to main, QA/TEST, LIVE Jessica, Render, Supabase or APK.
