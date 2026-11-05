# CALIBRI v190 — newly constructed balanced public DEV after historical teacher audit

Prior v189 actually compared original FP32 135M and original FP32
360M on 192 *previously encountered* historical development
prompts: machine-confirmed 66 vs 105, but no demonstrable
improvement in opposite Yes/No and logic domains; automatic
factual extractor failed to recognize correct replies such as
"Rome" and must NOT be treated as ground truth.

## Distinct fixed public prompts

Without looking at any model answer, predefine exactly **64 NEW,
untrained, publicly generated questions** in four balanced groups:

- 16 arithmetic examples, eight additions and eight subtractions
  with distinct 3-digit operands; require a single integer.
- 16 binary contradictory numeric comparisons, eight pairs,
  deliberately balanced **8 Yes + 8 No** and reversing each pair.
- 16 fictional color-folder binding questions, eight pairs,
  varying the owner's folder color with entirely new fictional
  names; both opposite members must be correct to count a pair.
- 16 factual single-answer questions (chemistry, planets,
  geometry, biology, geography), deliberately distinct from
  the exact previously reused historical 96 prompts.

Freeze every full question, gold answer, pair ID and SHA256
**before either teacher model is loaded or generates a token**.
Block any exact question reuse from the old 96 historical
development prompts. The model versions are immutable:
135M original old pinned SHA and **360M original
`a10cc1512eabd3dde888204e902eca88bddb4951`**,
361,821,120 unique parameters with tied output/input head.

## One strict common, precommitted, answer quality grader

Both original models get exactly the same QUESTION text,
own original instruction tokenizer/chat formatting, greedy
generation cap 32. Preserve every raw token ID and verbatim
text, EOS and source input hashes; raw output is primary.

Machine-confirmed "correct" means the WHOLE canonicalized
response is the expected number/color/Yes-No/fact, with
only a narrow allowed neutral prefix ("The answer is",
"Answer:", etc.). A response that merely **mentions**
the expected answer embedded in contradictory, negated
or otherwise verbose text must be marked UNGRADED, not
silently promoted to correct. A recognized different
domain answer is machine-wrong. The grade's limitations
are clearly disclosed: verbose semantically correct
paraphrases may remain ungraded, and this is not a
human semantic accuracy evaluation.

Count marginal confirmations, parsed wrong vs ungraded,
and **BOTH** answers in each opposite pair verified,
rather than assuming different first tokens guarantee
a meaningful answer. Importantly a constant Yes
predictor can never pass a pair.

There is no original or compressed MODEL TRAIN in v190,
no source or CPU benchmark in this run, and no inference
about actually achieved 0.7 bits per original parameter.
For future 360M model compression the complete byte cap
is exactly **31,659,348**, computed using its *own*
361,821,120 unique original parameters.

The 64 cases are PUBLIC DEVELOPMENT data (not the sealed
private v89 TEST); no future quantizer, decoder or source
training is allowed to optimize directly to these gold
answers. A separate genuinely untouched TEST remains
required before any claim of generalization. Negative
results must be preserved verbatim. No KL/NLL/PPL
model-quality acceptance.

All changes remain in an isolated research draft PR
and its GitHub Actions; do not touch main, LIVE Jessica,
Render, Supabase, APK, Green Tunnel or private v89 TEST.
Jessica only relays user messages, never runs experiments.
