# CALIBRI v100 — separate hidden-residual and output-head failures

v98 frozen CB59 native logits share ZERO top10 with pinned original FP32
on four new same-prefix samples, with large drift at embeddings, layer0,
layer9 and final norm. v99 Actions #41743949266 succeeded technically,
but no single original FP32 hidden activation patch restored coherent
text or reference top1 in 28/28 interventions. Exactly one patch
(layer10 on creative prompt) recovered one reference top10 token.
Previous v97 original FP32 head-only substitution also failed
coherent language. A paired interaction may explain distributed damage.

v100 uses four NEW synthetic diagnostic prompts with exactly original
archived 11,229,303-byte native CB59 and pinned original FP32 teacher.
On same token prefix compare these preregistered 2x2 intervention cells:
(A) unmodified CB59; (B) original FP32 final-norm activation, native head;
(C) native final-norm state and temporary full original FP32 lm_head;
(D) BOTH FP32 final-norm activation and full original FP32 lm_head.

Save next-token top10 predictions and first-token match on four probes.
For two probes, save actual greedy 12-token generated sequences of
all four cells plus independent FP32 text. When supplying exact original
activations, invoke original FP32 teacher on EVERY student-generated
prefix. Both oracle types are UNPAID, potentially expensive and NEVER
physically deployable within 0.7 bits per original parameter. They cannot
count as native-speed experiments or new compressed model quality.

No training, no model variants saved, no changes to frozen weights,
official v89 TEST unopened, no live Jessica/Render/Supabase/APK changes.
Modular facades v100/spec.py, factors.py, generation.py, pipeline.py
plus launcher, unit guard and isolated GitHub Actions workflow.
