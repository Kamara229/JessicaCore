# CALIBRI NRA2 S3I — diagnose bridge generalization, not another bigger adapter

## Frozen prior evidence
- S3E successful native 43M CPU #43512049506, real full token [48,T,320], original 616 real CPU J7 retained.
- S3F successful infrastructure #43514000830, but failed model generalization: TRAIN 32/32 graph, OOD 0/16 graph and reachability 7/16 (worse than S3D 8/16).
- S3G #43514565056 preregistered and unit-validated unique 64 TRAIN and four groups of 16, separating topology, wording, node count and combined shifts.
- S3H #43516017432 is the required upstream source for physical [128,T,320] frozen CPU hidden states, masks, labels and protected original 616 hash guard.

## Locked before any control metric

**No architecture search now.** Reuse precisely S3F's six learned node queries, 4-head 320→32 attention, bilinear directed-edge output, source/target heads, frozen 2241-parameter NRA reachability loss, seed 2026111401, 240 AdamW steps, lr .002, weight decay .02 and exactly the S3F loss weights. Only the 64 TRAIN graph prompts may influence model weights and normalization statistics.

**Commit the physical FP16 bridge, TRAIN-only mean and scale, and their SHA before using control gold for any score or selection.** The complete label archive is loaded into memory to extract the first 64 rows, but the four control slices are not passed to the optimizer. This is a guarded research diagnostic, not cryptographically sealed gold. Do not run gradient steps after commitment. Score once each:
- CTRL_MATCHED16: novel topology, same size & wording.
- CTRL_WORDING16: novel topology and wording, same size.
- CTRL_LENGTH16: novel topology and six nodes, same wording.
- CTRL_BOTH16: both length & wording shifted.

Required metrics: exact graph, edge F1, source/target, frozen NRA reachability and true-graph oracle for each group. Keep prior S3F 0/16 exact as a historical negative, NOT a directly comparable matched split. All four control results are research diagnostics, NOT independent final model acceptance.

**If CTRL_MATCHED16 fails:** address node-identity binding, graph serialization, class/structure shortcuts and shortage of diverse training supervision before changing parameter count or adding more parallel attention. Use a real token-to-entity binding mechanism and syntactic invariances, never gold or hardcoded answer lookup during inference.

**If CTRL_MATCHED16 passes but WORDING fails:** study phrasing variability and text-token alignment.

**If LENGTH fails:** investigate permutation/variable-node structure and extrapolation; control for longer prompt token lengths.

**If all controls pass:** build a genuinely fresh unseen final set, evaluate unchanged 584/584 retention, new real full+EOS outputs, total physical bpp <=0.7, matched native CPU speed >=0.9 reference and exact prompt independent generalization before any integration.

Output gate remains zero. Calibri weights/Green Tunnel/NRA stay unchanged. No Jessica ordinary chat tests, LIVE, Render, Supabase, APK, main, sealed QA or TEST.
