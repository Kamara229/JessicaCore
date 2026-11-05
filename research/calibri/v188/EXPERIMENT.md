# CALIBRI v188 — source FP32 ability on all 192 historical public DEV prompts

The original uncompressed SmolLM2-135M is a relatively weak
135M-parameter instruction model. Late CALIBRI v185/v186 are
physically compact and run at reference native speed, but neither
recovers broad coherent answers. The small original FP32 control
in prior v185/v186 used only 24 historical public DEV questions.
We therefore need a complete, honest portrait of the original
FP32's **actual free-generated answer ability**, rather than
attempting to match an arbitrarily weak 6/24 machine grade.

- Run **all original 96 varied historically established public
  development questions**, each with BOTH exactly precommitted
  `direct` and `brief` prompts, for **192 actual FP32 answers**.
- Domain counts are frozen: arithmetic, fictional binding,
  yes/no, factual recall, and logic. Opposing fictional
  binding and yes/no pairs are graded BOTH sides, not just
  marginal Yes/No accuracy.
- Freeze model revision and input hashes *before any inference*.
  Require 134,515,008 actual tied ORIGINAL FP32 unique parameters.
- Fixed 32-token greedy generation, full original text, EOS,
  actual token IDs and token diversity. Conservative machine
  grade: unparsed/ambiguous answers are UNGRADED, not wrong.
  Gold word occurrence alone is never counted as proven
  semantic success.
- Preserve all 192 raw outputs, by-domain and by-prompt-format
  counts, first-token diversity and within-question answer
  consistency across the two prompt templates.
- Do not create source weights, no teacher-target tuning, no
  model optimization, no quantized model file, no DEV-based
  candidate selection and no new external prompts.
- This is **HISTORICAL DEVELOPMENT** data already used for
  debugging; it is NOT an independent blind TEST of
  generalization or model capability. The independent
  frozen private v89 TEST remains sealed and unread.
- Its purpose is to define the actual baseline capabilities
  before selecting a genuinely stronger original pretrained
  source or a next multidimensional physically paid
  representation. It cannot establish any compressed-model
  answer quality, 0.7bpp storage or CPU throughput.

The v187 real 4D cross-row 2×4 source pilot is recorded as
negative for operator speed (~71.85% median CB59 on sampled
matrices) and almost unchanged reconstructed-weight error
(median 4D/CB59 ratio 0.99906); its evidence does not
justify claiming recovered intelligence.
The concurrent v186 v185-derived compact source achieved
3/12 heldout opposing first-token pairs solely in
fictional Yes/No possession and 2/24 historical DEV
machine-confirmed, with arithmetic still repeated 5.
Neither is ready for LIVE Jessica.

Research-only draft PR, no main, LIVE Jessica, Render, Supabase,
APK, Green Tunnel, or private v89 TEST changes. Jessica
is relay-only and performs NO model tests.
