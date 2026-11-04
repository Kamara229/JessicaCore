# CALIBRI v107 — cross-run check for v96/v106 character-level detector

v96 original real 32-token outputs had nonsensical native generations and
a misleading strict-prefix score for the original pretrained FP32 model.
v106 inspected v96 original 12 responses per model and constructed a
POSTHOC descriptive two-signal detector based on duplicate 5-character
windows and zlib compression; in v96 it detected 11/12 CB59 and 12/12 v92
native repetitions, versus 1/12 previously. v106 thresholds were selected
after seeing v96: they have NO independent validity yet.

This v107 step is a strict CROSS-RUN, NO-FITTING evaluation of those
UNCHANGED thresholds on two other original completed GitHub Actions
research outputs that are SHA256-pinned and not derived from v96:
v103 (#41747347901) — four authentic chat prompts: 4 original FP32,
4 original native CB59, 4 same-byte v102; v104 (#41748235592) — two
other original chat prompts: 2 pretrained FP32, 2 native CB59 and 2
physically paid eight-bank CB104. Total 18 generated answers, with
12 clearly repeated nonsensical packed-native strings and 6 fluent
but not always factually correct original FP32 strings, which often
include artificial system/user role separators.

Re-read the FULL original recorded response strings, preserve source
prompt fingerprints and all 32 original generated token IDs. Apply EXACT
v106.signals thresholds as-is and count TP/TN/FP/FN against a
source-specific visible language coherence label, not factual
correctness. Important: output readability labels are manual review of
previously visible original samples, NOT independently blinded human
annotations, not new prospective TRAIN/VALID, and not the frozen v89
TEST. Even perfect agreement across 18 small examples does NOT justify
an autonomous Jessica self-repair pipeline or prove quality improvement.

No model is loaded, trained, changed, tested again, selected, or
promoted. Only Python standard-library original text audit through
v107/contract.py, crossrun.py, launcher, unit guards and dedicated
read-only GitHub Actions. Frozen source model archives, main/Render/
Supabase/APK/Live and original official independent TEST are untouched.
