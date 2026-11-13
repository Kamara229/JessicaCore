# NRA2 S3G — isolate the token-bridge generalization failure

## Evidence, not a passed adapter

Successful execution of S3F #43514000830 **does not mean a successful model**.
The bridge fit TRAIN32 32/32 exact graphs, endpoints 32/32, reachability 32/32;
on reused OOD16 it yielded exact graphs **0/16**, reachability **7/16**,
directed edge F1 **0.18095**, source **7/16**, target **4/16**.
S3D last-vector bridge had OOD16 0/16 exact graphs and 8/16 reachability;
frozen oracle-fed 2241p NRA: 16/16. FP16 bridge physical reload succeeded;
no candidate was accepted, no new full+EOS gains demonstrated.

**Diagnostic interpretation:** capacity and lower TRAIN loss are not enough.
S3F trained on 4- and 5-node chains with familiar prompts, while its
single heldout simultaneously changed node count to six, graph
topologies and phrasing. We must separate those distribution shifts
before inventing another bigger attention adapter.

## Next controlled experiment design

Generated (but NOT scored) by
`stage3g_factorized_graph_controls.py`:

| Split | Size | Nodes | Prompt style | Role |
|---|---:|---|---|---|
| TRAIN64 | 64 | 4/5 | familiar | candidate training only |
| CTRL_MATCHED16 | 16 | 4/5 | familiar | unseen topology, matched input |
| CTRL_WORDING16 | 16 | 4/5 | new | wording shift only |
| CTRL_LENGTH16 | 16 | 6 | familiar | graph size/path-length shift only |
| CTRL_BOTH16 | 16 | 6 | new | combined shift |

All 128 unique directed chain topologies, balanced 50/50 reachable and
unreachable per group, each template contains both outcomes. A separate
independent reachability function verifies the labels. The fixed seed and
group counts are committed **before capture or training**.

**Required follow-up before scoring:**

1. Capture genuine frozen 43M CPU layer12 [T,320] hidden states for
   these 128 prompts using an additional read-only stage with
   immutable physical FP16, masks, source SHA, original 616 J7 parity.
2. Train new adapter only on the 64 declared TRAIN rows. No dev or
   control labels, gold, template ID, or question hash at inference.
3. Freeze physical FP16 and thresholds BEFORE any four controls.
   Run each control exactly once, reporting graph exact, edge F1,
   entity source/target, NRA reachability and per-family errors.
4. Run a graph-oracle check on each control before blaming the bridge.
   Report train, matched and shifted metrics separately.
5. The controlled splits are research diagnostics and visible in
   code. They are **not an untouched external blind benchmark**.
   If a new architecture is chosen after inspecting their failures,
   build an entirely new, isolated final holdout.

No changes to original Calibri or Green Tunnel weights, output gate
remains 0, no Jessica ordinary-chat testing, QA/TEST, main, LIVE,
Render, Supabase or APK. Do not move Stage4 harder trials into model
evaluation until an adapter passes an independent unseen-structure
validation and the full+EOS / size / CPU guards.
