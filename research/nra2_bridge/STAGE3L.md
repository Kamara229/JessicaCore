# NRA2 S3L — physically frozen S3K and honest four-control outcome

## Verified real research execution

- **S3K successful CI** #43519730567, physical FP16 bridge
  SHA `16cc39dc64667260260da2954d3a56ff7f65f8846284087822f2fddea4b49c3a`,
  68,664 bytes, true frozen S3H 43M native layer12 [128,85,320].
  On TRAIN64: graph exact **64/64**, edge F1=1,
  endpoints **64/64** each, frozen NRA reachability **64/64**.
  This was the training set only; no generalization proof.
- **S3L successful CI** #43520095966, artifact
  `nra2-stage3l-frozen-s3k-4x16-exploratory-only`.
  NO optimizer, frozen S3K physical FP16 SHA verified unchanged,
  no checkpoint selection, and all 616 original S194 J7 hashes
  inherited from S3H, unchanged core/GT/frozen 2241p NRA, output gate=0.
- The exact four control labels were previously examined in S3I.
  These results are *exploratory*, not an untouched final blind test.

| Group (each n=16) | Exact directed graph | Edge F1 | Reachability | Source | Target | True-graph NRA oracle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Matched size+wording | **16/16** | 1.0 | **16/16** | 16/16 | 16/16 | 16/16 |
| Novel wording | **16/16** | 1.0 | **7/16** | 10/16 | 6/16 | 16/16 |
| 6-node length | **16/16** | 1.0 | **16/16** | 13/16 | 16/16 | 16/16 |
| Both wording and length | **16/16** | 1.0 | **8/16** | 6/16 | 5/16 | 16/16 |

This is a meaningful **structural** improvement on this small
single-chain graph task relative to S3I **0/16 exact graph in every
control**. The bridge no longer guesses edges from static node
queries; it uses real native token offsets for all named node
mentions and neural classifiers on the contextual text windows
between consecutive mentions. No gold graph or hardcoded arrow
parser enters inference.

However, new wording can reverse the textual order of endpoints
(`Can {dst} be reached from {src}` vs `from {src} to {dst}`).
S3K currently binds the last two mentions as textual candidates and
learns their roles from the familiar TRAIN64 grammar. It is not
trained on the reverse wordings. The correct graph plus wrong
source/target leads to weaker reachability even though true-graph
oracle NRA was **16/16**. Fixing frozen NRA is NOT indicated.

## Preregistered next experiment (do NOT tune on S3L's exposed controls)

New S3M source inputs must be captured from the original frozen
43M CPU model, using **fresh graph topologies**, not S3G controls.
Make counterbalanced pairs: the SAME graph followed by multiple
questions with independent `src,dst` choices and opposite
yes/no results. Each TRAIN grammar must include genuine forward
(`from src to dst`), reversed surface order
(`can dst be reached from src`), negated and structural wordings.
All splits must be topology-disjoint, no QA/TEST, with 4/5/6
active nodes and branching/cycles to rule out single-chain shortcuts.
Write/read exact real token-offset alignment, but never present
graph gold, expected answer or prompt SHA to inference.

Design a **language-conditioned neural query-role scorer** that
examines the local contextual phrase near each mention plus role
words, rather than using last-two mention order as semantic labels.
Freeze physical Calibri/GT/NRA and protect original 584/584 and
all S194 616 J7 hashes. Train only a preregistered TRAIN split;
commit FP16 and norm before any holdout. Hold back a completely
untouched final topology+wording split that is evaluated ONCE,
without hyperparameter/architecture choice using its scores.

Only after a genuinely independent increase in real full+EOS
responses should the research candidate proceed to same-CPU
all-in bpp<=0.7 and speed>=0.9 baseline and possible integration.

**Current candidate NOT accepted.** There is NO evidence yet
of any new native decoder full+EOS gain, full packed physical bpp,
matched speed or independent final blind semantic generalization.
Do not modify LIVE Jessica, Render, Supabase, APK, main, QA/TEST.
