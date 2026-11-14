# NRA2 S3AO: preregistered once-only four-way EVAL128 comparison

## Immutable provenance (locked BEFORE any S3AO scoring)
- Research-only parent: S3AN #43572358979 successful. Native *original frozen* 43M CPU S160 layer12 [128,97,320] FP16 EVAL128, no candidate scoring; 616 original S194 J7 output digests unchanged.
- S3AK committed TRAIN256/EVAL128: #43542086287, commitment SHA `e1c8b0f9ad533482c351c5ac392f5923a59759c57c62d55080d692cbebf5354c`. EVAL128 source hash `e0e30f96665241520b71dce791d6305eb9b44a5b2eb9d1132804927c058d407e`. 32 previously unseen graph topologies, four unused wordings, source-first 64, destination-first 64, yes 64, no 64.
- S3AN physical token FP16 SHA `0b75c0f271cf580b04c38af665c0d52c373f37444f9dc13c2588afdbd59b4863`, last hidden SHA `8497866b4176384c87400703b0e1f483753e17c0a14e21489378a14b60084a65`, gold-label SHA `dee82a2d226f55a9f1408d9bb8f0a5a1574dc2f8209ec77aecb2b93e4cb532ab`.
- S3O base FP16 SHA `40a8903d5419cd8641b5d01e75e429e10b449304b3ba140d2245bf7f8a9d00b6`; S3Z FP16 SHA `60a7121b8270f647909776b58dc21b07c6cb885fd6ebeb6cb3158b23221f6e9f`; S3AG FP16 SHA `2c23de87efc2369ed6fc0880604420070c116a4ee70b66a4eb0a946f1afeb118`; S3AM FP16 SHA `c5c400fc8249ea474263c9388f03e1496320bc8f6b793f93d155a962c3bf0eef` (#43544245980).
- Independent NRA 2241 parameters unchanged; original S3O graph branch and normalizer unchanged bit-for-bit. All four candidates loaded as frozen FP16. No training, no threshold tuning, no checkpoint selection, no hidden QA/TEST.

## Predeclared evaluation
1. Run exactly once after source and candidate SHA assertions: frozen same CPU input for four role heads S3O, S3Z, S3AG, S3AM. Edge sigmoid > **0.5** (fixed), source/destination softmax, immutable 2241p NRA reachability logit > 0. Preserve output gate **0**.
2. Primary: S3AM `reachability_hard_correct` on EVAL128 compared in *paired* way with S3O baseline. Secondary: both semantic roles correct; graph exact; forward vs destination-first (64 each); yes/no (64 each); topology families and four grammar styles; false-positive and false-negative; question-level SHA and per-arm correctness.
3. Report `S3O_only_correct`, `S3AM_only_correct`, `both_correct`, `both_wrong`; also report S3Z/S3AG comparators, without tuning, replacing the baseline, or selecting a model using this data. Exact graph and logits must be identical across arms. If role gains do not translate to reachability gains, record this rather than claiming improvement.
4. Set `candidate_accepted=false` irrespective of scores. Do not reuse this EVAL128 for later training, model selection, hyperparameters, or new blind claims. Any comparison is an exploratory scientific one-time proxy, NOT a new full decoder + EOS gain.
5. Actual full decoder + EOS, physical all-in bpp <=0.7 and paired CPU speed >=0.90 with zero loss on S194 616 must be proven independently on a fresh, unexposed evaluation before acceptance.

No writes to main/LIVE, Jessica ordinary chat, Render, Supabase, APK, QA/TEST, Green Tunnel or the original 43M decoder. Research branch + GitHub Actions only.
