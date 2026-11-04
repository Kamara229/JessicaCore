# CALIBRI v102 — first source-paid head repair experiment, not another FP32 oracle

v101 Actions #41746010640 completed technical PASS. It checked the
entire original 49,152×576 original tied head. Original 5-bit indices
roundtrip correctly and C native GEMV matches explicit Python decoding
within **8.583×10⁻⁶ maximum absolute logit**, so this is NOT a C
decoder/bit order bug. But genuine pinned trained FP32 head weight differs
from CB59 recovered head by **relative RMSE 0.662516**, mean row cosine
only 0.74158. All teacher-vs-native prompt vocabulary top-10 sets
remained disjoint, including single-factor activation/head tests.

This is an intentional irreversible *information-loss* problem.

## Concrete v102 experimental replacement
Instead of merely adding a teacher-activated correction or patching a
single operator, **re-code the whole tied input embedding+output head**
as ONE physical codebook with the *same original 5-bit K32V8 runtime ABI*.
No extra byte, no standalone head duplication. Different original
source-only K=32, 8-D centroids fitted from **32,768** original
pretrained weight vector samples rather than 4,096, and two candidate
whole-block row-scale policies (RMS and newly fitted scalar least-squares).
Select one using ONLY original full pretrained source squared error.
No labeled QA, no synthetic generated text, no independent TEST is used
for refit or codebook choice. Repack all 49,152 rows, 5-bit index
streams, FP16 centroids, FP16 row scales and whole source CRC/SHA.
Entire original CB59 manifest, every other trained weight matrix,
source dimensions and C kernel ABI must remain bitwise unchanged.

Candidate is a separate 11,229,303-byte research archive; never
overwrite frozen CB59 baseline, nor push new source to main, Render,
Supabase, live Jessica or APK. Rate remains exactly original 0.667839bpp,
well below user ≤0.7bpp ceiling.

CI independently verifies source-only RMSE, actual FP32 hidden +
original/candidate native head and full model output top10 on NEW
synthetic contexts. Also run actual 32-token true cached CPU generation
on two contexts with BOTH separately compiled original/candidate
native C runtimes and pretrained FP32 teacher, record *actual decoded*
generated text, paired CPU token/s and ratio. Technical CI success
is NOT semantic success. If full-model text remains nonsense or speed
falls below 90% original reference, record scientific FAIL and
do NOT promote. Native speed is only a preliminary 2-prompt result
with hardware/run noise.

Source-only whole block rewrite cannot guarantee original semantic
quality. It establishes an operational, physically paid iteration point
for future smarter dictionaries/hybrid operators and identifies whether
source coding, not repeated patching, can recover useful output.

Facade files v102/spec, recode, metrics, pipeline, launcher, unit guard;
isolated draft PR and GitHub Actions workflow. Historical official v89
independent TEST stays sealed.
