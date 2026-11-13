# CALIBRI 2.0 NRA2 — S3B independent bridge smoke (research only)

S3A (#615, CI #43480311111) physically proved 32 unique real 43M CPU
last-layer 320D FP16 TRAIN vectors and full unchanged 456+128+32 output
token digests. It did **not** train an adapter or improve any answer.

## S3B narrow mechanics, not acceptance

This experiment builds an actual learnable 320D-to-soft-graph read adapter,
connected to a **frozen** independently trained RelationalNRA 2,241-parameter
graph unit. Its output is never injected into Calibri (strict gate=0).
The 32 S190 TRAIN prompt identities are split deterministically by
SHA within each of four families: **24 TRAIN, eight genuinely withheld**
(6/2 each). The eight are never optimized or used to normalize features;
they are evaluated once after 60 predetermined training steps. A fresh
physical FP16 adapter is serialized and bitwise reloaded, but pretrained
graph weights and original 43M weights remain unchanged.

The only available 32 real TRAIN prompts involve *owner/item paraphrases*
and *numeric comparison paraphrases*. They are **not a supervised graph
representation dataset**. To avoid inventing graph edges, S3B is explicitly
a limited *binary proxy* with transparent labels obtained from original
S190 training prompt construction: requested Owner B vs Owner A, and
numeric first number exceeds second. It tests that a real hidden320 adapter
can be optimized while gradients flow through a frozen graph core. It
does **not** demonstrate owner/item graph reconstruction, structured
reasoning, semantic generalization, or correct multi-token speech.
S3B deliberately has no active decoder output or generation effect.

S3B DOES NOT re-execute the full 43M J7 on any newly changed decoder.
It loads the physically validated S3A original 584/584 J7 report,
which stays relevant because S3B does not load or change the original
decoder; it does not substitute this prior report for acceptance of a future
active candidate. No candidate same-CPU speed or all-in bpp is asserted.

## Advancement decision after S3B

Only proceed to a truly semantic NRA bridge after creating a *real paired*
TRAIN data set containing the frozen Calibri hidden tensors, independently
grounded relational graph structure and source/target entities, plus
held-out structural forms and negative checks against shortcut cues.
Then evaluate the trained bridge and frozen NRA on those held-out graphs.
Only after semantic success consider a separately trained nonzero
output gate, with fresh physical 616 J7, 584/584 retention, new full+EOS
gains, same-CPU speed >=90%, and all-in physical bpp <=0.7.

No QA/TEST, no production, no LIVE Jessica, Render, Supabase or APK change.
