# CALIBRI v193 — NEW 96 original-model QA, v192 grader FROZEN before answers

Purpose: decide whether the original **uncompressed** pretrained
135M/360M FP32 models are genuinely able to produce correct
and distinguishable diverse responses in controlled data, not
whether any compressed CALIBRI model achieves target parity.

Prior actual experiments:
- v188 historical public 192 actual original 135M FP32 responses:
  66 machine confirmed, 57 ungraded (v188 parser).
- v189 historical same 192 original 360M FP32: 105 machine
  confirmed, 44 ungraded (same strict v188 parser).
- v190 genuinely new 64 public questions/model: v190 precommitted
  bare-answer grader scores 9/64 135M and 18/64 360M,
  47 and 38 ungraded. Original model 0/8 contradictory
  number-greater-than pairs both correct.
- v191 **retrospective posthoc** structural answer pattern
  scores 34/64 vs 43/64. NOT independent confirmation.
- v192 grader locked in independent GitHub CI #42094376648
  and hashes sealed **before v193 prompts exist**:
  exact SHA `e230e480c02c680ae02d17b0825b451cbf76b138ceb743f181cc6fb62ad4ac1f`.

### v193 preregistered experiment

1. Create exactly 96 **NEW** public question texts and GOLD,
   24 arithmetic (addition/subtraction/multiplication),
   24 explicitly reversed numeric Yes/No (12 pairs),
   24 original-owner color binding (12 pairs),
   and 24 short diverse factual STEM/geography/biology questions.
   Verify not identical to 96 v169 historical or 64 v190
   public-dev prompts. Reject duplicated SHA or unbalanced
   gold in opposite pair. Freeze ALL question/gold SHA
   before importing original model/performing inference.
2. In a separately committed GitHub Actions job, verify
   immutable `v192/contract.py`, `recognize.py`,
   `grade.py` SHA256 aggregate matches exact sealed
   v192 manifest. ABORT on any unexpected source drift.
3. Load the original source pretrained FP32 135M and 360M,
   validate pinned revisions, unique parameter counts and
   tied output embedding to avoid double-counting, use each
   original chat template and exactly 32 greedy tokens.
   Do not train either or read the private v89 TEST.
4. Preserve ALL 192 verbatim generated responses (96 each),
   token IDs and original prompt-template SHA; use **only
   frozen v192 independent three-way answer grading** and
   both-correct opposite pair successes. UNGRADED stays
   ungraded; a model saying Yes to every input can
   score at most one member per contradictory pair.
5. Publish all negative cases and exact 3-way raw counts.
   This is a modest *public development* source-capability
   audit, not human semantic accuracy or a sealed blind
   private evaluation. Do not adapt the frozen grader to
   the observed answers. New pattern improvements would
   require a new future version and an additional
   truly untouched set.

No compressed model or bpp speed measurements in v193,
no KL/NLL/PPL acceptance and no Green Tunnel research
parameter additions in reported bpp. 360M theoretical
complete archive ceiling for <=0.700 original bpp:
31,659,348 bytes based on unique original 361,821,120
parameters; this is NOT physically achieved yet.

All work on isolated unmerged research PR. Do not change
main/LIVE Jessica, Render, Supabase, APK, Green Tunnel,
private v89 TEST; Jessica relay does not run model tests.
