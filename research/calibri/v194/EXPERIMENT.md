# CALIBRI v194 — ACTUALLY PHYSICALLY COMPRESS original 360M and RUN packed native QA

**Purpose:** stop treating endless comparison of uncompressed
135M/360M models as progress on CALIBRI's central goal.
The prior original v193 SHA-pinned FP32 question audit
completed successfully. The 360M original got 46/96 machine
confirmed, while 135M got 30/96, with 0/12 reversed numeric
pairs both correct (both originals), and 0/12 360M reversed
fictional bindings both correct. The factual strict parser
undercounts full correct sentences; no post-hoc metric
mutation allowed.

## Stage A — real physical entire source bytes

- Load FULL ORIGINAL pretrained
  `HuggingFaceTB/SmolLM2-360M-Instruct`
  at frozen HF SHA
  `a10cc1512eabd3dde888204e902eca88bddb4951`
  in FP32; verify 361,821,120 **unique** parameters,
  including tied input/head alias.
- Use existing fully authenticated **CB59 5-bit code
  per 8 source weights, 32-entry per-tensor FP16 codebook**
  with original-source-only KMeans approximation.
  Also physically store every row's FP16 scale, raw
  FP32 trained 1D norm and metadata.
- Write ACTUAL binary, SHA256 and CRC, validate alias,
  all tensor names and physical full-archive byte length.
  Require bytes <=31,659,348 (0.700 bpp of **original
  unique 360M**, not the 135M model count). Include
  complete source binary in a GitHub artifact if pass.
- 2D matrices are NEVER allowed to remain FP32 outside
  binary as uncounted native trainable source tensors.
  This baseline is simply the 360M instance of the
  135M CB59 rate-first codec, not novel compensation.
  It is an honest **baseline** to improve upon.

## Stage B — actual C-native packed inference, NOT a simulation

- Download ONLY the complete physical source artifact;
  do not load FP32 original 360M weights in this stage.
- Verify actual source SHA and full byte count, tied alias
  and every source-trained 2D module replaced by native
  C packed GEMV or selective embedding row decode;
  resident source 2D trained dense parameters must be 0.
- Precommit 16 distinct frozen public v193 question IDs
  (4 arithmetic, 4 reversed numerical comparisons,
  4 reversed fictional binding, 4 factual). Existing
  original 360M answers & original prompt hashes come
  from **successful v193 artifact #13446083971**.
  Reuse the EXACT frozen v192 answer grader SHA
  `e230e480c02c680ae02d17b0825b451cbf76b138ceb743f181cc6fb62ad4ac1f`,
  not new evaluator or gold tuning.
- Actually generate **16 complete greedy 32-token-cap**
  answers from packed-C resident real pretrained 360M
  model. Preserve every wrong, repetitive, EOS and
  ungraded answer including original FP32 baseline.
  Compare genuine full-answer quality and bounded
  opposing-pair success using identical original
  question chat formatting.
- Measure native wall seconds per case as diagnostic.
  Do NOT claim a +/-10% CPU speed gate without matched,
  equal-work paired native CPU benchmark to an actual
  same-360M native source baseline on same runner.

**IMPORTANT:** physical source rate <=0.7 alone does NOT
prove answer correctness or CPU speed. This experiment
may FAIL quality. Do not hide negative results or
misrepresent source-only RMSE as answer accuracy.
The 135M archive physical numbers (v186) cannot be
used as a substitute for a genuine 360M measured source.

## Experimental limits
New physical 360M archive is generated ONLY in isolated
research draft PR and GitHub CI. Do not merge into
main or touch LIVE Jessica, Render, Supabase, APK,
Green Tunnel or independent private v89 TEST.
No Jessica-chat model tests or training on public
DEV answers; no NLL/KL/PPL acceptance gate.
