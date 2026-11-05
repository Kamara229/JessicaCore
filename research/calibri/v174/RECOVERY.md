# CALIBRI continuation from v172/v173 — 2026-11-05

## Verified completed evidence

- v172 head `cf7e0d4b61769dfa67b96bea2dc1fe3ea0245845`, PR #309,
  [target run 41996650582](https://github.com/Kamara229/JessicaCore/actions/runs/41996650582), artifact 13400004156.
  All 211 matrices refitted, 11,229,303 bytes, 0.6678394131307638 bpp.
  40 questions: FP32 12 machine-correct / 23 parsed / 17 ungraded;
  original CB59 and CB172 0 parsed / 40 ungraded. This is NOT a semantic
  claim that FP32 gets only 12 right. Generated CB59 texts are repetitive
  fragments. CB172 CPU 11.9818278993 tokens/s vs CB59 11.9970219336,
  ratio 0.9987335162 on that run. No demonstrated answer recovery.
- v173 head `707e8d6510fe6f13dfff54c6133aa056a4a50055`, PR #310,
  [target run 42001114597](https://github.com/Kamara229/JessicaCore/actions/runs/42001114597), artifact 13401918441.
  Same compressed weights, native versus dense diagnostic: first argmax
  agrees 8/8; entire generated text agrees 6/6. Four sampled real matrices
  have relative native/dense error 3.65e-7 to 4.12e-7; embedding error zero.
  Both compressed-source paths repeat `speci` rather than answer; FP32
  correctly answers Mars/Tokyo and gives a meaningful handwashing response.
  This strongly supports representation loss as the cause on these probes,
  not a blanket mathematical proof that every possible runtime path is correct.
- Target workflows succeeded technically. Unrelated inherited v0.7/v0.8/
  v0.9/v0.11 workflows include failures; the repository CI is not all green.

## Next experiment and interpretation

Historical v110/v111 already found broad damage on four INT6-control
prompts; v111 separated input embedding, body and head. This is an
expanded behavioral replication with original FP32, 96 questions, a longer
generation cap, and attention versus MLP interactions. It does not claim
that family localization itself is new, nor assume one easy module repair.

v174 runs a full 2^3 intervention over tied embedding/head, all attention
matrices and all MLP matrices. Every family is either original FP32 or the
exact original CB59 decoded values. All eight combinations answer the same
96 historical development questions. No weights are fitted and no gold is
sent to model inference. The separate private v89 TEST remains unopened.

The 64-token cap records EOS/truncation. The previous conservative extractor
is retained: lexical gold occurrence and ambiguous prose require review;
unparsed answers are not asserted to be wrong. All raw responses are saved.
The suite is diagnostic development data, not a new independent final test.

These are explicitly dense interventions, not new <=0.7-bpp models. The
source file remains unchanged; original FP32 weights present in hybrid arms
are counted explicitly as UNPAID parameters. No native speed claim follows
from dense timings. v173 supplies native/dense equivalence evidence.

Use the completed eight-arm result to choose a representation change:
an isolated family failure points to its coding geometry; failure only in
combinations points to interactions/accumulation. A successful hybrid is
not a compressed success. Any subsequent actual candidate must physically
fit the unchanged 11,229,303-byte budget and be evaluated through native
execution against FP32 on real answers and speed. No KL/NLL/PPL gate.

Only research files and this research workflow are added. No merge to main,
no LIVE Jessica, Render, Supabase, APK or Green Tunnel modification.
