# NRA2 S3S — predeclared S3Q new-graph native CPU soft/hard comparison

S3P #43524516584 diagnosed all six S3O research DEV64 errors as
cycle-sink false positives caused by tiny nonzero *soft edge*
probabilities. Predicted 0/1 graph edges were otherwise 64/64 exact;
same frozen NRA returned 58/64 using edge probabilities and 64/64
using deterministic binary predicted edges. This prior S3M DEV
observation alone is post-hoc and NOT an independent result.

## Frozen BEFORE first S3Q test score

S3Q #632 CI #43525211488 physically preregistered 16 completely
new directed graph topologies and 64 corresponding prompts, with
32 yes/32 no, no header/question-template/length shortcuts and
topology disjoint from S3G and S3M. Commitment SHA:
`5e9ea9cd66d5dedf0aa2cf574dcf78c8e0a7522c892f775fd8e9e014c46a26cb`.

S3R PR #633 CI #43525586699 is the sole genuine original frozen
43M CPU [64,T,320] FP16 feature source for all 64 prompts. It
repeats S194 original 616 generated CPU J7 and 584 full+EOS
retention. It creates graph gold only **after** full original CPU
acquisition, and runs NO S3Q scoring itself.

This preregistered S3S trial is one fixed paired two-arm experiment:

**Shared frozen elements:** exactly the original S3O neural graph/
semantic-role adapter FP16 SHA
`40a8903d5419cd8641b5d01e75e429e10b449304b3ba140d2245bf7f8a9d00b6`,
its original physical training normalizer, the same independent frozen
2241p standalone NRA, same 64 S3R real frozen CPU token vectors.
The exact **same SOFT source and destination probabilities** go to
NRA in both conditions; no gold endpoints are ever fed to inference.

**Single independently varied factor:**
- A: original soft edge weights `sigmoid(edge_logits)`.
- B: `(sigmoid(edge_logits)>0.5).float()` predicted hard edges.

Output is unchanged frozen NRA logit `>0`, with no training,
finetuning, threshold selection or reruns designed to maximize S3Q
score. Test 64 total and four families each n=16, plus each family
positive/negative n=8; report graph exactness and endpoint
correctness separately, both false positive and false negative
counts, and every divergent/question error. Record independent gold
graph+gold endpoints only as labelled attribution/control, never as
the primary model score or decoder output.

CI is allowed to run after successful S3R original CPU artifact
and physical original S3Q commitment. No ordinary Jessica chat
tests. This **public research set reuses S3M question wording** and
therefore any signal measures graph topology transfer, NOT wholly
unseen language. Success, if any, is NOT independent closed final
QA/TEST, NOT an original native decoder new full+EOS improvement,
nor proof of all-in bpp <=0.7 or matched CPU >=90%.
Output gate remains zero and model candidate not accepted. Original
Calibri/Green Tunnel/NRA/QA/TEST, main, LIVE, Render, Supabase and APK
never change.
