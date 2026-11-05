# Completed CALIBRI v174/v175 stage — 5 November 2026

**Decision: v175 fails behavioral quality.** The compact candidate is not
accepted as a quality improvement, not merged into main and not deployed.
Both targeted workflows completed; no experiment from this stage remains
pending. 768 diagnostic v174 generations plus 288 v175 generations = 1,056.

## Verified sources

- v174: commit `9f7abe44373a4478fa02fd08788765b76d0d6b16`,
  [PR 311](https://github.com/Kamara229/JessicaCore/pull/311),
  [run 42004560915](https://github.com/Kamara229/JessicaCore/actions/runs/42004560915),
  artifact 13407006303. Four targeted tests passed.
- v175: commit `83775bdab217363b0fbe10b1c9527c9121cf24f1`,
  [PR 312](https://github.com/Kamara229/JessicaCore/pull/312),
  [run 42008214907](https://github.com/Kamara229/JessicaCore/actions/runs/42008214907),
  artifact 13406448460. Five targeted tests passed.
- Raw generation artifacts expire on 5 December 2026. This directory stores
  the measured summary and all 96 FP32 texts with explicit posthoc review.

## v174: full factorial source localization

Every arm answered the same 96 questions at a 64-token cap. The FP32 arm
had 36 machine-correct, 38 machine-incorrect and 22 ungraded outputs. These
are extractor results, not a complete semantic assessment (see review).

Each of the other seven arms had zero machine-parsed correct answers,
no EOS before the 64-token cap in all 96 cases, and no demonstrated recovery.
Even compressing just the tied embedding/head, just attention, or just MLP
was sufficient to cause failure on this suite. This strengthens earlier
four-prompt findings; it does not establish that every individual matrix
is equally critical or that all alternative low-rate representations fail.

The embedding/head+attention arm had token diversity 0.7132 despite
incoherent output. Diversity by itself must not be treated as intelligence.

## v175: actual same-byte source candidate

- Complete archive: **11,229,303 bytes; 0.6678394131307638 bpp**.
- SHA256: `cba50f7c03253f43a180e817e09979878967c71e176965a9d4ac391300defaa5`.
- All 211 matrices changed; 8,158,011 existing 5-bit group indices changed.
- Every matrix passed the regularized calibration-loss fallback, yet actual
  behavioral recovery failed. Local calibration gain is not answer quality.
- Native median speed ratio to CB59: **0.9987748977 (99.88%)**, measured on
  eight alternating fixed-work pairs on this CPU. Not a universal speed guarantee.
- CB175: **0/96 machine-confirmed correct**, all 96 hit the token cap without
  EOS; inspected outputs are irrelevant repetitions such as Airlines/sqlalchemy.
- Four real native matrix checks had relative errors around 3.5e-7–4.2e-7
  against literal decoded weights; embedding difference was zero.
- Across independent runs, original FP32 sequences agreed 96/96. Original
  CB59 dense v174 versus native v175 sequences also agreed **96/96**, extending
  v173's execution-path agreement evidence. This is not FP32 quality parity.

## FP32 content review (posthoc, assistant-reviewed)

All 96 texts were read. Results are kept separate from frozen machine grades:
46 clear correct; 41 incorrect; 2 cut off without a correct completed answer;
1 correct color with unsupported explanation; 6 indirect correct-order
restatements without explicit first-place answers. Depending on whether the
last two categories are accepted, a total varies; do not publish one inflated
unqualified accuracy. These are historical development prompts, not a blind
final benchmark.

- Factual: **14/16** clear correct. The other two confuse months/days/weeks.
- Arithmetic: **8/16** clear correct; two responses are truncated at 64 tokens.
- Binding: 12 clear correct + 1 correct color with unsupported elaboration,
  out of 32. Case 31 is a machine false positive: asked about Bina, answered
  about Orin, with a color contradicting Orin's described notebook.
- Yes/no story queries: **8/16**, with strong constant-answer behavior.
- Logic: 4 clear correct comparisons; 6 correct-order restatements kept as
  indirect; 6 incorrect. Presence of a reference name is not semantic success.

The extended limit corrected one truncation-related interpretation (80-27
completed as 53), but 26-16 completed as the incorrect 12. Increasing output
length does not by itself fix reasoning.

## Scope and next research boundary

This stage is closed. No new run is requested by this report. Reject v175 as
a quality recovery. Further work should test a materially different source
representation or joint quantized training within the same byte budget;
repeating minor K32/V8 index refits is unsupported by these outcomes.
The evidence does not prove the global <=0.7-bpp goal impossible.

No NLL/KL/PPL success criterion. Private v89 TEST unopened. No main, LIVE
Jessica, Render, Supabase, APK or Green Tunnel modification. Legacy inherited
CI includes unrelated failures; only the two targeted workflows are called
technically successful here.
