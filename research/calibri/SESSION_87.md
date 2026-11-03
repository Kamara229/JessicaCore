# CALIBRI v87 — root-cause test: causal operator vs local compression error

## Last verified experimental result, 3 November 2026
Real GitHub Actions **41579153088 SUCCESS** ran frozen pretrained model v86 on entirely unseen original TEST. Software succeeded, but **all research-quality gates did not pass**.

At 11,240,769 physical bytes = **0.66852133 bpw**, frozen CB85:
- SciQ teacher / CB59 / CB85 = **22 / 7 / 7** correct of 24.
- BoolQ = **11 / 13 / 12** of 24; adaptation regresses one question.
- WikiText KL **10.19007 → 10.10661**, AG News KL **10.05242 → 9.97320**: small improvements.
- Independent real KV-cached greedy distinct ratio CB59 = CB85 = **0.21875**, vs FP32 teacher **0.9375**.
- Single-CPU CB59 **23.02** vs CB85 **22.32** tokens/sec; no proof of speed improvement.

Thus v85's improvements to TRAIN-VALID KL/NLL are real but **do not restore QA or repeated-token generation**. We MUST NOT optimize against the now-opened v86 TEST or quietly retune the frozen candidate.

## Preregistered v87 scientific question
v83 located 98.12% local squared-error energy in layers 11/28/29 MLP down projection, but local error is not causal responsibility for broken end-to-end behavior. The 135M compressed model also has a native-compressed tied token embedding / vocabulary head.

We therefore temporarily insert **actual original FP32 pretrained operators into the packed student's own activation path**, and test end-to-end consequences. Fixed diagnostic scenarios:
- no-change sham CB59;
- pretrained output vocabulary head only;
- pretrained input embedding only;
- original FP32 three suspect MLP down operators only;
- output head + down operators;
- original embedding + head;
- all original embedding + head + down operators;
- separately, unchanged physically frozen CB85 control.

Unlike local teacher-forced error comparisons, these interventions measure **full-model causal downstream responses to replacing the operator on the real student trajectory**. Replacement operators draw the FULL original pretrained FP32 weights and are **UNCOMPRESSED DIAGNOSTIC ORACLES**, absolutely NOT new 0.7bpw candidates. Nothing is trained, compressed, promoted or deployed in v87.

## Strict new source and measurement
Only original WikiText **train** and AG News **train** via existing TRAIN-split handles \`wiki_valid\`, \`news_valid\`; choose **window indices 2 and 3**, disjoint from v84 VALID window 0 and v85 VALID window 1. 2×64 tokens/corpus and 126 scored next positions/domain. Record data token SHA, full-vocabulary teacher KL, next-token NLL, original FP32 top-1 agreement and two fixed new authentic KV-cache 16-token greedy prompts. This is diagnostic TRAIN, **NOT independent generalization/TEST**.

Original CB59 SHA and physical CB85 SHA verified before experiments and never modified; original 135M teacher pinned revision. Zero imported v86 test rows or TEST-based optimization. Outcome must be reported as scientifically diagnostic even if GitHub CI passes; no hidden source cost claims for oracles.

## Modularity / isolation
\`causal_rewrite_v87.py\` thin CLI → \`v87\` facade → \`v87.pipeline\` orchestrator → small \`v87.oracle\` and \`v87.metrics\`; four numerical/restoration tests. A separate GitHub Actions research-only job publishes an evidence artifact. New stacked draft PR based on v86. **Do not merge or change main/live Jessica/Render/Supabase/APK**.

Next direction depends on real measured causal impact: construct a genuinely compact, *physically paid* learned substitute for the highest-leverage function, then new disjoint TRAIN-VALID and independent TEST; don't substitute expensive FP32 teachers at inference.
