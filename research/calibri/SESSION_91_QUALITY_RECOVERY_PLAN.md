# CALIBRI v91 — evidence-first quality recovery roadmap

## Established evidence
- Actual frozen 134,515,008-parameter model. CB59 11,229,303 B; CB88 11,413,683 B = 0.678805 bpp, with ~356 kB paid space remaining under 0.700 bpp.
- v89 original held-out independent TEST scientific FAIL (BoolQ and greedy diversity). Never use that TEST again for tuning or source selection.
- v90 CI #41591866792 PASS, research diagnosis only. BoolQ train[800:832]: CB59 14/32, CB88 8/32; disjoint train[6000:6032]: CB59 12/32, CB88 17/32.
- CB88 predicts No 32/32 and 31/32, a major class bias. Three fresh greedy diagnostic prompts: CB59 distinct-token ratio 0.0521, CB88 0.1146, both below 0.25 target. CB59 itself is already collapsing; do not assume that only the head correction is broken.
- Small 32+32 samples, constrained Yes/No answers, truncated context mean all conclusions exploratory.

## Mandatory diagnostic order (weights frozen)
P0 — Compare original pretrained FP32 teacher with CB59 and CB88 on identical true TRAIN cases. Verify evaluator/prompt, class recall, predicted class distribution, full context and labels. When teacher itself fails, distinguish task/model limitations from quantization damage.
P1 — Isolate four output-head modes using IDENTICAL CB59 hidden states: original, +FP16 bias only, +4D tensor only, +both. Numerically verify selected-token 4D contraction equals full operator. Quantify Yes-minus-No signed logit margins, per-case confidence and bias vs tensor contribution. If bias drives majority collapse, restrict/calibrate bias. If tensor drives it, examine normalization, factor geometry, gating and magnitude before any further training.
P2 — Compare all 2D original native packed projections to exact original FP32 on THE SAME TEACHER INPUT. Rank relative RMS, cosine and output scale by layer and operation. Separately record propagated hidden-state drift after every Transformer block. Local approximation error is not causal downstream damage.
P3 — Oracle causal rescue, diagnostic ONLY: temporarily replace one packed block, tied embedding or output head with its exact original FP32 counterpart; restore in finally. On 2 real truncated TRAIN contexts measure improvement in teacher KL, top-1 agreement and Yes/No margin; test top 6 distinct blocks plus first/last projection, embedding and head. Oracle FP32 replacements are not physically paid models.
P4 — Test genuine greedy KV decoding of original, bias-only, tensor-only and combined frozen head on the SAME 3 fresh TRAIN-only prompts. Log per-prompt token IDs, diversity and repeat share; a follow-up should probe entropy, top-1 gap, EOS probabilities and KV-cache/no-cache equivalence. Do not hide repetition with temperature or penalties.

## Remediation decision tree (after v91)
- If FP32 teacher has low BoolQ ceiling: fix/train-only prompt or evaluation protocol and limit claims before changing weight coding.
- If bias-only creates the No skew: reparameterize or zero-center bias, add logit-margin guard and objective balancing full-vocabulary KL + true-token NLL + balanced Yes/No recall; train on TRAIN, select on disjoint VALID, not frozen TEST.
- If the 4D correction dominates: rewrite operator with activation-aware normalization, low-rank factor conditioning, bounded residual, gated adaptation. Train candidate separately and retain physically complete block-accessible source.
- If quantized projections have high causal importance: redistribute physical bits toward the harmful layers using per-channel codebooks, adaptive scaling, error-correcting low-rank operator, or tied head/embedding correction. Remove equivalent cost from insensitive blocks and account all headers/metadata.
- If the error is spread across layers: combine a handful of modular operator fixes in small facades with joint validation for interactions; do not pile on undocumented whole-network dense weights.
- If greedy remains broken even with KL/QA improvement: directly investigate top-token saturation, native embedding/tied head, residual-stream drift and KV correctness. Decoding penalties are diagnostics, not restored probabilities.

## Gates
- File size <=0.700 bpp including correction, header, checksums; block reads and CPU native execution preserved.
- On disjoint TRAIN/VALID: KL improves against CB59/CB85, NLL not worse, SciQ/BoolQ class-balanced accuracy without regression or systematic majority response; greedy distinct-token >=0.25. Always report sampling uncertainty and negative results.
- CPU speed >=90% of native CB59 with the actual complete operator, not an unmeasured theoretical speed-up.
- No scientific PASS claim from GitHub CI alone. Pick, freeze, SHA-pin the best new source on TRAIN/VALID only and evaluate **once** on a truly **new** independent TEST, never the already observed v89 TEST.
- Research-only GitHub branch, small facade modules, no changes to main/Live/Render/Supabase/APK, no ordinary Jessica test without explicit request.

## v91 status
Inference-only frozen-source root-cause experiment. Its outcome decides the first measurable new operator correction. Refrain from retuning until its diagnosis is complete.
