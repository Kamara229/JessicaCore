# NRA2 S3M — semantic source/target role grounding: preregistered research

## Why

S3K frozen neural token/entity adapter #43519730567 physically trained
with 64/64 exact TRAIN graphs. Subsequent S3L #43520095966 (PR #627)
had 16/16 exact directed graphs on every previously exposed 16-case
control, but failed to assign correct source/target under unfamiliar
reversed wordings: reachability 7/16 wording and 8/16 both shifts.
This suggests the next bottleneck is **query-role interpretation**,
not directed graph reconstruction or the frozen 2,241p reachability
NRA. S3K/S3L controls were already viewed and may not be reused
to select architectures.

## New topology-disjoint source data

`stage3m_role_balanced_graph_data.py` builds **48 genuinely new
graph topologies** disjoint from all prior S3G topologies, with
4/5/6 active nodes and chain/fork/cycle-with-sink/diamond families.
Each topology has four semantic questions: 2 yes and 2 no, including
literal source-first and target-first phrasing; directions are reversed
for the same underlying graph. Four question formats each contain
both true and false examples, preventing a wording-to-answer shortcut.
Oracle directed reachability is computed independently by BFS.

- **TRAIN128:** 32 topologies x 4 counterbalanced questions, 8 graph
  topologies from each family. Only here may new role-head updates
  take place.
- **DEV64:** 16 topology-disjoint graphs x 4 questions, 4 graph
  topologies per family. Research diagnostic only, not sealed blind.
  Commit FP16 role-head + normalization SHA and all training
  decisions *before* scoring DEV. At most one planned DEV scoring
  pass; **no tuning on DEV results**.
- **FINAL:** deliberately **NOT GENERATED OR SCORED** at this stage.
  True independent final testing must be preregistered and set aside,
  with graph structures and phrasing unseen when selecting architecture.
  This fixed-seed research generator is not secret or cryptographically
  sealed and must never be described as a final blind benchmark.

## Next experiment, not completed by this PR

Physically acquire authentic frozen 43M CPU layer12 [T,320] token
sequences for all 192 new prompts with native S160 tokenizer offsets,
preserving original S194 616 generated token SHA, protected 584/584
answers and unchanged model weights. Build a trainable **semantic
query-role head** that reads actual query-context phrases and entity
mention spans and predicts which entity is the source/target. Do
not pass graph gold, oracle labels, expected answers or SHA to model
inference. Preserve S3K's demonstrated node-bound directed edge
branch; independently test its invariance to graph edge serialization
and renamed nodes. Freeze original Calibri, Green Tunnel and
independent NRA; output gate zero.

Only when actual untouched FULL+EOS improvements are observed should
we consider physical all-in packed bpp <=0.7 and matching CPU runtime
>=90% of the verified baseline. Not even future CI success or
TRAIN/DEV exact graphs is sufficient for model acceptance.

## Boundaries

No changes to main, LIVE Jessica, Render, Supabase, APK, QA/TEST or
any protected model artifact. This stage contains no neural training,
no real CPU token capture, and no final-blind quality claim.
