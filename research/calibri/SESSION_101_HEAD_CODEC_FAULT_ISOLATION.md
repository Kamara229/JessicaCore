# CALIBRI v101 — is model.head damage a codec bug or a rate loss?

v100 Actions #41744903506 technically SUCCESS. Full original-model oracle
2x2 on 4 new same-token-prefix prompts: CB59 native top10 0/40;
FP32 final-norm activation + CB59 head 1/40; native norm + FP32 head 0/40;
BOTH exact original pretrained final norm and FP32 head 40/40 with exact
teacher text. The last cell by design fully bypasses both native components,
a sanity check NOT an achieved compressed quality improvement.

v101 isolates *bitstream encode/decode parity* and *original-weight
information loss*. Read original immutable CB59 SHA and alias map; confirm
one source for tied embedding/output head. Visit each physical 5-bit packed
head row in 256-row temporary scratch blocks; decode every index, reencode
and verify physically identical source bits, reconstruct temporary numeric
rows from stored FP16 centroids and scales, compare each against pinned
original pretrained FP32 teacher output head weight. Compute aggregate
relative weight RMSE, cosine, row error fractions, codebook usage.
For two new synthetic prefixes, obtain FP32 and native final-norm hidden
activations; use them as IDENTICAL inputs to (a) C original native packed
GEMV and (b) explicit Python decoded head (full-vocab predictions). Confirm
C kernel and Python agree on absolute logits and top1, and separately
measure FP32 exact source versus packed approximation on ORIGINAL teacher
activations, then exact FP32 head on wrong native activations. Verify
native C embedding rows equal same decoded tied head rows at significant
boundary token IDs. **DO NOT modify source or save dense head matrices**.
A perfect C/Python roundtrip does NOT prove quality preservation; bad
original FP32-vs-CB59 comparison indicates intentional/irrecoverable-at-
that-rate information loss, not necessarily a software implementation bug.

Do not open old v89 TEST or train/test for candidate tuning. Research
facades v101/spec, codec, pipeline, launcher, tests + dedicated CI only.
No change to live Jessica, main, Render, Supabase or APK.
