# CALIBRI v96 — audit semantic correctness before further changes to compression

## New evidence from completed v94

- CI #41735013116 **technically SUCCESS**, but actual quality check **FAIL**:
  BoolQ 128 different public train-split questions: original FP32 71
  correct, locked native CB59 62, v92 full-block rewritten source only
  **51**. The previously observed 12-example v92 improvement 5 -> 8
  **did not generalize** to 128 new cases.
- SciQ four-option selection: CB59 14/64 and v92 16/64; original FP32
  only 10/64; BUT the same constrained A/B/C/D token prompt suffers
  answer-position bias (original FP32 predicts A 59/64, CB59 C 45/64,
  v92 C 52/64). Do NOT treat this as clean world-knowledge accuracy.
- Raw generated responses are mostly unintelligible repeated syllables
  or punctuation: e.g. `mersemersemerse...` and `--------------------------------`
  on trivial arithmetic. 16-prompt CB59 unique-token ratio .13086, v92
  .10938, below .25. This is a stronger negative result than a
  standalone low diversity number: **useful text is not being formed**.
- Physical source stays 11,229,303 bytes = 0.667839 bpp, unchanged
  native C runtime; measured v94 full KV CPU ratio is 1.00107, i.e.,
  no apparent rewrite-specific speed regression on these prompts.
  **All historical compression and speed achievements remain protected**.

## Why v96

Before any claim of a quality-preserving compression method, we need to
rule out two separate sources of failure: (a) weak original 135M FP32
model versus (b) damaging 0.668 bpp native compression. v94 only
checked the original FP32 on constrained Yes/No and A/B/C/D labels,
not actual natural-language generation.

v96 *freezes ALL sources*, generates on the same **fresh synthetic**
8 unambiguous short-answer canaries and 4 open-ended prompts using
FP32 original, native CB59 and byte-identical native v92 rewrite.
Grade only actual output string vs known ground truth for the 8
answer canaries. Save full generated text and token IDs for
interpretation, and repetition/character-quality alarms for 12.
Do NOT claim to automate subjective human evaluation of open-ended
text: results explicitly indicate human review outstanding.

No KL, NLL, logits imitation, or teacher distribution comparison. The
original FP32 is a **correct-answer reference**, not a requirement
to copy its probabilities or even its wording. Compression is judged
by correctness and useful coherent responses.

## Failure and recovery response

If FP32 forms coherent language but CB59 degenerates, the primary
bottleneck is information loss or structural distortion in the
0.667839 bpp representation — not an output-bias correction issue
alone. The next bounded research step would probe tied embeddings,
attention sensitive layers and codebook aliasing, but only within
the frozen original native 5-bit/8 vector CPU and ZERO extra bits.
If FP32 cannot answer the canaries, benchmark design and source
model ceiling are issues; do not falsely blame quantization alone.

Quality Doctor must treat obvious gibberish as FAIL regardless of
a higher unique-token percentage or more correct A/B/C/D guesses.
Factual correctness and coherent prose are more important than
mere random-token variation.

Never train on the synthetic canaries or the already-opened v89
independent TEST. Once canary results examined, use new untouched
public train-heldout examples for every subsequent model selection.
No CB59/CB88/v92 change, no main/Live/Render/Supabase/APK mutation.

This work runs in small facades: spec.py -> runner.py ->
grade.py -> pipeline.py; source integrity from established v94
provenance and native runtime API.
