# CALIBRI v69 — REAL whole-model functional gate after mixed v68 paired GQA layers

**Pre-registered 2026-11-03. Isolated research PR.**

## Why
The verified v68 experiment on pinned real pretrained SmolLM2-135M-Instruct at exact 11,229,303 B/0.667839 bpw proved exact source GQA function-preserving V/O scaling but had MIXED local attention output error: layer 0 **+3.66% worse**, layer 15 **−2.22% better**. The successful synthetic v67 test was not representative. The missing next decision is whether v68's fully packed 135M archive actually improves **end-to-end model outputs**, not local matrix L2 errors. Do NOT change quality claims on the basis of a two-layer result.

## Frozen test
1. Recreate v68 candidate by reapplying the identical source-only GQA analytical norm gauge to **two paired layers (0 and 15)**, with original v59 K32V8 quantizer, original physical v59 CB59 source and exact full manifest. Extend v68 research helper only to emit an optional candidate archive, not to change its algorithm.
2. Verify candidate physical length exactly **11,229,303 B**, SHA/CRC and all trained 2D/1D source keys, tied aliases. Source model original SHA pinned as in v68.
3. Separate teacher FP32 and two separate native compressed packed inference processes: identical **original CB59** vs **gauged v68 CB59**. For packed phases, create Llama *from config* and replace ALL 2D tensors with original direct C-K32V8 LUT GEMV/embedding. No original pretrained dense 2D matrices remain in packed runtime, and no teacher source weights loaded in packed processes.
4. Fixed post-hoc untouched WikiText2 **VALIDATION** split tokens offsets `4096:4224`: **4 disjoint windows of 32 tokens**. Compute 124 context→next-token positions, matched against the same original FP32 teacher output. This scope is **exploratory, small** and not a statistically significant language ability demonstration.
5. Report per-position teacher→packed KL nats, next-token heldout NLL/perplexity, original teacher top-1 agreement, normalized logit L2; paired native CPU 4-window prefill timings separately. Teacher data is NEVER used to choose gauge or quantizer. No full-runtime quality or speed claims before actual job completion.
6. Integrity tests failing are CI failures. Gauged candidate losing against original CB59 on KL/NLL is a scientifically valid negative result and is **NOT** a CI failure.

The whole-model archive is generated inside CI for packed inference; only audit JSON uploaded to Actions artifacts, not original model trained weights. Scripts stay in `research/` on draft PR, no changes to live Jessica, main, Render, Supabase, APK, tokens/secrets/settings.

## Acceptance / decision
A meaningful go for deeper **multi-dimensional neuron representations** requires:
- Exact physical source rate <=0.700 incl all metadata and aliases, all real compressed inference modules retained.
- On disjoint heldout model eval, KL and NLL both better than original CB59 at matched rate; report uncertainties and follow with larger unseen sets.
- Runtime slowdowns recorded; no production deployment. Even a positive 124-position result cannot establish task competence because original CB59 was severely degraded (v61 0/8 token matches).
- If no improvement: do NOT use blind global norm-gauge; pivot to supervised, TRAIN-only function-space gauge search / mixed per-matrix tensor decomposition and stronger end-to-end training.

## Status
Results pending CI at initial commit. Need check workflow, record actual signed deltas, and preserve negative results.
