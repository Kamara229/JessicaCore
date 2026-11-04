# CALIBRI v104 — physically paid C-native eight-bank whole tied head

v101 real frozen source decoder parity proved CB59 5-bit head was
encoded/decoded correctly: max C vs Python logit difference 8.583e-6,
but pretrained FP32 head relative weight RMSE 0.662516. v102
physically same-byte tied head refit (larger source-only codebook
sample and LS row scale) improved original weight RMSE to 0.648297,
preserved CPU speed (~0.999 ratio) and 0.667839 bpp, but it did
NOT recover full native output: all 4 new synthetic next-token top10
overlaps zero and real generated text repeatedly meaningless.
v103 added chat-templated public BoolQ TRAIN independently labeled QA,
because naked/non-chat FP32 teacher itself made incorrect answers.
v103 findings must be considered before claiming semantic success.

v104 tests a genuinely distinct **physical source and native operator**:
partition the 49,152-vocabulary original tied embedding/head by eight
contiguous regions of 6,144 rows. Fit distinct K=32, V8 FP16 centroids
from original pinned pretrained head weights in each bank (4,096 source
vectors per bank). Reassign all 5-bit head indices and optimize physically
FP16 per-row scalar on source-only weight L2. Preserve one original tied
head/embedding shared source index stream and per-row scales, all other
pretrained matrices *identical*. The original CB59 base manifest, length
and CRC/SHA remain parseable, but its last head's codebooks are replaced
with the bank0 source; **original v59 standard C would misdecode rows 6144+**.
Thus wrap the complete recoded 11,229,303-byte source in a new CB104
container, plus ALL 8 FP16 32x8 bank centroids (4,096 bytes) plus
24-byte header, 4-byte CRC and 32-byte SHA. Total 11,233,459 bytes
(check authoritative constants), ~0.6681 bpp (<0.7). Never count only
5-bit indices, ignore sidecar bytes, or invoke original standard C for
the new banked head.

v104 native C kernel precomputes LUT for 8 banks, then decodes packed
codes per row using correct bank. A separate C embedding kernel reads
only selected token rows with identical bank index and shares original
packed trained head handle. There is no dense 2D trained source, no
FP32 teacher call, or external sidecar free bits at **candidate runtime**.
All other layers use original CB59 packed-native C operators. Exact
full-vocabulary C vs temporary Python banked-decoder same-input parity
plus C demand embedding vs decoded head row are asserted.
Only the research diagnostic constructs tiny temporary 256-row numeric
scratch, and the CPU reference FP32 teacher is not used by candidate.

Compare genuine pretrained head FP32 weight relative RMSE, four fresh
same-prefix original synthetic contexts, two new genuine-chat format
gold arithmetic/geography generations and actual CPU KV-cache native
32-token/s against frozen original CB59. All result strings remain in
artifact for independent semantic review. **Source-weight RMSE alone is
not quality**. No source selection by QA, no old v89 independent TEST.
The v104 candidate remains an isolated PR and artifact, not promoted
to Jessica main/Render/Supabase/APK/Live.

Facade layout: spec.py, physical.py, encode.py, runtime.py,
metrics.py, pipeline.py, launcher + focused CPU C unit tests and
Actions workflow. If native speed <90% baseline OR actual outputs
remain incoherent, quality/speed FAIL and reject v104.
