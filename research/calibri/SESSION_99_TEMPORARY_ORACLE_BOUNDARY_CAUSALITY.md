# CALIBRI v99 — stop global language corruption before spending any more bits

Prior v98 Actions #41743033442 SUCCESS with physically frozen CB59
(11,229,303 bytes, 0.667839 bpp), but no meaningful native generation:
embedding relative RMS drift 0.727, first layer 0.957, layer9 1.302,
final norm 1.435. **All four** same-prefix FP32/native next-token
top-10 sets were disjoint; native output used repeated low-frequency
unrelated tokens. This is a serious scientific FAIL despite CI PASS.

**Question:** If we supply an EXACT original reference hidden activation
at one boundary in the packed runtime on the SAME token prefix, is
meaningful next-token support recovered, or are downstream packed layers
and/or tied output-codebook broken too?

On four FRESH synthetic contexts, observe teacher-native top-10
overlap before intervention. Pre-register 7 boundaries:
input embeddings, layers 0/8/9/10/29, final norm.
For each, temporarily patch only the full reference stage activation,
and compare next-token top-10 support and top-1 against FP32.
On two contexts also generate 12 actual tokens. On EACH autoregressive
step run full original FP32 teacher on the identical, changing student
token prefix, then intercept that one teacher activation and inject
into native next-token pass. This is a **very expensive physically
UNPAID oracle** and completely invalid as a speed or size benchmark.
Every forward hook is removed after use (including errors), pretrained
packed weights NEVER change, previous official TEST stays sealed.

Gains could inform where the model is no longer recoverable; they are
NOT compact operator improvements, semantic certification, or evidence
that a decoder with access to FP32 would be a real compression method.
No automated winner selection; preserve actual texts for manual review.

Facade design: v99/spec.py -> patch.py -> behavior.py -> pipeline.py.
Isolated CPU unit tests and GitHub Actions, research draft PR stacked
on v98; no live main/Render/Supabase/APK/Jessica modifications.
