# CALIBRI v98: accumulative hidden-layer divergence, not more blind edits

v97 run #41741673757 completed with technical CI SUCCESS; JSON artifact
#13282136401. Original FP32 BoolQ 8/16, frozen native CB59 6/16
on new original public TRAIN; temporary FP32 oracle head 10/16
but predicted NO 15/16, a severe answer-class bias. Early MLP
oracle also 10/16. Native outputs remained largely repeated nonsense;
no single-block oracle restores actual coherent language.

**v98:** freeze exactly the original complete 11,229,303-byte CB59
source (SHA-256 in CALIBRI_FROZEN_BASELINE_2026-11-04.json).
For four NEW synthetic prompts only, run the original pinned FP32
SmolLM2-135M-Instruct and unchanged CPU native packed model on
*identical input token prefixes*. Capture embedding, EVERY transformer
layer output residual, final norm via short-lived forward hooks.
Measure relative RMS, cosine, step-to-step increases. At first
next-token logits, measure vocab entropy, top-1 dominance,
top-10 overlap and decode top candidates. Separately obtain and
preserve actual greedy 32-token generated outputs and repetition
alarms on both models, distinguishing errors on the same input
from feedback caused by differently generated tokens.

Report first drift threshold and biggest layer-level increases,
not simply largest absolute local error. This is an observational
fault-localization test and DOES NOT select a compression variant
or establish causal attribution without subsequent interventions.
No optimization, weights changed, QMoE claims, semantic score from
entropy, external TEST, or Jessica test. No CI success as quality PASS.

Facades v98/spec.py, hooks.py, metrics.py, pipeline.py, launcher,
CPU unit tests, isolated draft PR and GH Actions.
