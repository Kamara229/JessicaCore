# CALIBRI v58 — migrate from raw 14M Pythia to stronger open instruction model

A **single isolated research experiment** following completed CALIBRI v57 draft PR #187: real physically stored INT6 quality/speed on an actually **instruction-tuned** publicly available open-weights 135M model, not another variant of raw pretrained 14M Pythia. Previous v55–v57 repeatedly established tiny PPL improvements without stable gains in real 0/1 answer accuracy; a truly more task-relevant source model is necessary to test applicability. This stage does NOT claim performance on deployed Jessica; it only tests the compression algorithm on a more relevant open model.

## Frozen source and experimental protocol — BEFORE first CI run

**Teacher:** `HuggingFaceTB/SmolLM2-135M-Instruct` (Apache-2.0; 135M, LlamaForCausalLM, HF revision pinned dynamically by model_info().sha and saved). BF16 pretrained checkpoint loaded to **FP32 CPU** to compare full teacher weights against decoded model on exactly the same ISA/threads. The source **ties input token embeddings and output LM-head weights**. This is a key full-model compression trap: duplicating a tied 49152×576 embedding would artificially inflate checkpoint bytes and parameter denominator. Hence define and test a new **CB58** schema:
- Canonicalize exact source-tied tensors by original `state_dict()` order (same storage+shape/strides+offset). Store only UNIQUE source tensors in independently CRC/SHA-verified **CB46** sub-archive: **all 2D** floating source matrices rowwise signed INT6 / FP16 scales (the actual physical **CF15** codec), keep exact 1D original float vectors in source FP32. Include explicit alias-name mapping (embedding/output) and actual source tensor names.
- Add outer CB58 metadata, length checks, CRC32 and SHA256. Both standalone CB46 and CB58 checksums must pass; cold loader reconstructs all expected source keys and preserves tied embedding/LM-head data_ptr identity. Physical disk byte size includes ALL unique tensors, manifest, aliases, both integrity layers, not hypothetical size. The denominator of actual bpw must be **unique trained parameter count**, equal to `sum(p.numel() for p in original.parameters())`. Alias names are stored on disk; never count tied weights twice. Confirm source complete and signed six-bit schema before evaluating any heldout accuracy.
- This first new-architecture pilot uses deterministic **un-calibrated rowwise INT6** on all source 2D matrices (no pretend Pythia MLP calibration is directly portable to SmolLM2). Unique source-only per-row SSE priorities in fixed Python float64 `math.fsum`; one balanced max **12% of total repair byte budget per physical trained matrix**, greedily choose highest original-weight minus decoded-INT6 squared error per physically stored FP16 residual-row byte, first one row per tensor. The independent physically stored **CR47** patch is parent SHA bound, CRC/SHA-checked, exact tensor shapes/row indices. Max complete CB58 + CR47 + **15,480 B future POST correction reserve <=6.50 physical bits per UNIQUE pretrained model parameter**, not a fictional sparse/dense estimate. Restore chosen FP16 residual rows at cold load ONCE. No new linear layer/hook or runtime correction inference pass.
- Predeclare comparison: **uncompressed FP32 instruction teacher** vs **complete physical INT6 baseline** vs **same physical INT6 plus source-only balanced FP16 error recovery**. No model training, test/validation fitting or post-hoc tuning of row selection.

## One frozen TRAIN-only real-answer selection gate

After physical source and correction files are persisted, open only brand-new TRAIN portions:
- SciQ TRAIN[864:880], 16 labeled questions, support+question, four options, gold text not in prompt; hash-shuffle options and collapse exact duplicate answer texts (predeclared source-schema normalization from v57).
- SuperGLUE BoolQ TRAIN[608:624], 16 labeled questions, passage+question and yes/no option independently scored; gold labels not in prompt.

Fixed context prefix ≤128 and candidate ≤32 Pythia-compatible model-token IDs, zero-shot causal normalized **mean token logP(option | question prefix)**. For each of the 2 TRAIN families, measure total correct/16 and mean `gold-answer logP minus best wrong-answer logP` across the 16. Choose the extra CR47 source-only repair **only if** neither TRAIN task loses any correct answers, BOTH families have strictly positive mean gold-margin gain over physical INT6 PRE, and equally weighted average margin gain >=+0.001. Otherwise select **smaller plain INT6 PRE**, saving ~8MB correction. This fixed small sample gate is a heuristic, not a reliable population predictor or true calibrated confidence interval. No test/validation data accessed before final TRAIN choice.

## First independent heldout: actual 32 question answer correctness each

After **immutable** selection, for first time read:
- SciQ TEST[352:384], never used in CALIBRI TEST[:352] before.
- BoolQ VALIDATION[480:512], never used in CALIBRI validation[:480] before.

For original open instruction teacher, real full-state source-only physically restored plain signed INT6 and the balanced FP16 sidecar alternative, measure zero-shot **correct/32** and real paired gold-vs-best-wrong option log likelihood margin. Log per-case predictions, source revision, train/test question+options SHA256; all 3 variants measured even if sidecar loses TRAIN gate. Crucially: no training on test, no HellaSwag/ARC-tuned model choice. This is a small feasibility pilot and not a statistically significant quality finding.

## Real CPU speed and cross-host verification

Use two independent GitHub Actions runners **Ubuntu22.04 / Ubuntu24.04**, pinned CPU `torch==2.5.1`, `transformers==4.46.3`, NumPy, AVX2/2 threads; actual autoregressive full model **128-token prefill +32 KV-cache decoding tokens** original instruction teacher vs physical TRAIN-selected INT6 cold-restored model, with 2 warmups+7 repetitions. Hard user's speed slowdown ceiling: ≤20% warm median. No time estimate or success claims before actual Actions verification.

Independent third comparer checks all real CB58 and CR47 physical bytes+SHA bitwise identical across both CPU hosts; unique tied source parameter count and bpw ceiling; selection math from TRAIN-only original margins+correct; first-use heldout 32 correct answers each; timing raw 2+7 data. If any integrity fails, CI FAIL and report exact error; a real quality regression is a valid negative scientific result, not by itself a storage integrity failure.

**Important:** The reconstructed running model always expands **whole INT6 source to dense FP32 hot RAM** at startup for speed. No packed INT6 runtime memory saving, native INT6 GEMM, Jessica integration, production APK, Render, Supabase, or chat testing. Do not modify main, production, secrets, settings, or user data.

## Source implementation integration correction (before any heldout TEST access)

Initial two-host CI reached the new 135M model's physically stored source and residuals successfully, then failed **before loading any heldout data** in the inherited v57 helper `load_fixed_tasks`, which hard-coded FOUR QA families while this frozen feasibility protocol intentionally tests exactly TWO (`SciQ`, `BoolQ`). This is a pure scope mismatch, unrelated to observed predictions or task quality. Replaced that helper call with `load_two_tasks` which preserves v57's exact per-record schema/answer normalization but checks the frozen **2-family** set and exact 16 TRAIN/32 untouched TEST items. No correction algorithm, 6.50bpw cap, selection threshold, sample partitions, speed settings or heldout labels changed. Amendment disclosed before any HELDOUT data were read.
