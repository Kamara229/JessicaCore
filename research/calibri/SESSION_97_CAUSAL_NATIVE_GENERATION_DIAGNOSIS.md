# v97: freeze size/speed, locate source of broken native generation before another weight rewrite

v95 completed #41737705182: selected original CB59_LOCKED on 32 BoolQ +
24 SciQ FIT, independent 64 BoolQ + 32 SciQ VALID returned equal
CB59 35/64 and 6/32; FP32 30/64 and 13/32. No quality gain.
Old v92 worsened BoolQ on v94, and native generations largely
degenerated into repeated syllables despite 0.667839 bpp.
Do not present technical PASS as scientific quality improvement.

v97 changes **no physical archive**, model weights or source runtime.
Read the immutable verified full native CB59 11,229,303-byte archive.
Load actual FP32 original weights only as a temporary diagnostic oracle.
Original model's output probabilities are *never* a target for fitting
or selecting a candidate: evaluate actual publicly labeled BoolQ TRAIN
correctness and print the entire generated text, with lexical repetition
and honest human-review-needed flags.

Causal ablations: temporarily replace the packed input embedding,
output tied head, both simultaneously (they are tied and might exert
distinct influence at input/output), early/late MLP down_projections,
and middle attention value projection. In all cases restore in finally
and assert native packed module handles restored after each trial.
All source SHA and complete actual all-in bpp are unchanged.

**Important caveat:** Exact FP32 oracle swaps are *not physically
permissible at <=0.7 bpp* and are *not native CPU speed comparable*.
They are upper-bound causal localizers. A module improvement is a
hypothesis about which native compact codebook/operator to redesign,
not an achieved improvement or license to deploy. Small synthetic
answer-reference present in first sentence is not certified truth.

New original public BoolQ train slice: train[8720:8736] (unseen in
previous v94/v95 fit/VALID). Four fresh generation prompts.
The official frozen independent v89 TEST is never opened or tuned on.

Small facades: spec.py -> oracle.py -> behavior.py -> pipeline.py,
with independent fake-model unit tests and a dedicated Actions workflow.
The workflow performs original CPU-only probes, uploads traceable
results, never touches main/Render/Supabase/Jessica Live/APK.
