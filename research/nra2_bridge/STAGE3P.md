# NRA2 S3P — isolate six S3O DEV errors: frozen neural NRA input interface

S3N true native 43M CPU run **#43522555326 success**: 192 token-contextual
states FP16, original S194 CPU 616 token hashes and protected 584 preserved.
S3O run **#43522663612 success**: physical 148,434-byte FP16 neural
graph + query role adapter, SHA
`40a8903d5419cd8641b5d01e75e429e10b449304b3ba140d2245bf7f8a9d00b6`.
On already exposed, topology-disjoint but non-blind DEV64:
64/64 exact graphs, 64/64 both semantic endpoint roles, 58/64 frozen
NRA predicted reachability. With gold 0/1 graph + endpoints, same
frozen original NRA 64/64.

**Scientific question:** does the remaining 6/64 error arise from the
relational model's limitation on complicated graphs or from the
continuous probability interface? Do not presume an error in reasoning
when exact binary graph and endpoints recover 64/64.

S3P is a predeclared **post-hoc diagnostic** on the previously exposed
DEV64 with no fitting or parameter changes. Eight 2x2x2 tests feed
the same independent 2241-param frozen NRA with independently
soft/hardened *predicted* edge probabilities, source probabilities,
and target probabilities. Hardening is deterministic >0.5 edge
binarization and endpoint argmax (no oracle). Compare to an additional
three oracle-only label-controlled tests: gold graph + gold endpoints,
gold graph + predicted soft endpoints, and predicted soft graph + gold
endpoints. They exist only to attribute the six failures, not for
inference or parameter choice.

The artifact records all 6 error indices, graph family, length,
question style, logit margin, true-edge minimum probability, false-edge
maximum probability, true source/target probability, 8 intervention
outcomes and immutable SHA provenance.

**Research caveat:** binary/hardening 64/64 on this already exposed DEV
is NOT generalization, does NOT license selection of a threshold using
DEV and does NOT mean that NRA2 can solve the actual native 43M
decoder's new questions. Any new interface should be committed before
a fresh, truly independent final set, then verify actual native
generation full+EOS, zero forgetting, compressed all-in bpp <=0.7,
matched CPU >=90%, and no QA/TEST/LIVE writes. Until then gate=0.

No training, no Jessica ordinary-chat tests and no changes to
original Calibri, GT, NRA, main, Render, Supabase, APK or QA/TEST.
