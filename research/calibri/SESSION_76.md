# CALIBRI v76 — real TRAIN-fitted paid rank-4 weight residual within one source-complete <0.700 bpw archive

**Pre-registration: before the first v76 CI TEST run. Draft research PR only.** The user-facing Jessica chat/weights, main, Render, Supabase, APK, deployment and production secrets are not changed.

## Actual prior observations motivating this stage

The v75 real pretrained full 135M native compressed model CI tested 12 original 2D/8D source representations against **two independent TRAIN datasets**. Robust TRAIN minimax rule rejected all modifications and chose original CB59 unchanged. Subsequent TEST KL/NLL differences were exactly zero, as appropriately reported. More generally, original rate-first CB59 is 11,229,303 bytes, 0.667839413 bpw for 134,515,008 original UNIQUE trained parameters and is disastrously degraded on genuine text and labeled answers. Neither diagonal GQA gauges, fixed 2D Hadamard rotations, nor 8D activation-weighted index edits recovered real useful quality. A future claim must compare entire neural function, directly executable packed inference, and all physical bits.

**v76 changes the question.** Can a **physically paid correction to the pretrained output weight operator**, optimized using REAL packed-model hidden states and teacher full-vocabulary KL on *multiple TRAIN corpora*, improve heldout entire-model quality without exceeding the global 0.700 bits/original parameter cap?

## Codec: a genuinely complete packed trained source + low-rank factor operator + bias

- Read pinned original real pretrained `HuggingFaceTB/SmolLM2-135M-Instruct` revision `12fd25f77366fa6b3b4b768ec3050bf629380bac` and physically original CB59 archive from v59 verified Actions run 41507201766 (11,229,303 bytes, all trained weights/metadata/alias tables/CRC/SHA).
- **Do NOT change a single original CB59 byte.** Build ONE independent CB76 source file by appending a small self-describing correction payload (with versioned magic, dimensions, CRC32 and SHA256) to the complete original CB59 bytes.
- Model has original output head weight `W_out` packed 5 bits per 8 original source weights and tied embedding. Add an explicit **rank-4 original-embedding-dimensional residual matrix** `ΔW=U Vᵀ` with `U∈R^{49152×4}`, `V∈R^{576×4}`, plus vocabulary output bias `b∈R^{49152}`. These are new correction parameters, not reconstructed original 2D full dense source weights; the rank-4 matrix itself never materializes. The v76 runtime calculates `y=NativeCB59(W_out,x) + U(Vᵀ·RMS_normalize(x)) + b`. This normalized-feature variation is an intentionally different *low-rank functional correction*, not literally a simple constant linear perturbation of W; describe it accurately. No other pretrained 2D source is unpacked.
- All `U`, `V`, `b` entries physically stored as **FP16** in the one file. No optimizer state, original teacher source, duplicated packed model, dataset or hidden feature cached in the released source. Model runtime decodes only those paid FP16 factors into FP32 buffers for a small CPU Torch matrix computation on top of the still-native C packed GEMV. **No claim of end-to-end native C implementation for the paid residual**, and no speed parity expectation.
- Physical bytes, including all original trained model metadata and source weights:
  - complete original CB59 source: **11,229,303 bytes**;
  - FP16 U: 49,152×4×2 = **393,216 bytes**;
  - FP16 V: 576×4×2 = **4,608 bytes**;
  - FP16 bias: 49,152×2 = **98,304 bytes**;
  - CB76 header and CRC+SHA: **16+36=52 bytes**;
  - **single complete file 11,725,483 bytes**, `8 × 11,725,483 / 134,515,008 ≈ 0.69735` bpw, BELOW true full-source 0.700bpw cap `floor(0.7 × 134,515,008 /8) = 11,770,063 bytes`. Remaining physical budget 44,580 bytes.
- This is a simple paid rank-4 low-rank sidecar trained via full-model teacher KL, NOT a novel multi-dimensional neuro-weight theorem, not general enough for all layers, and not zero-bit magic. It is a falsifiable first budget/quality pilot for true learned, directly executable tensor residuals.

## Fit and guard: strictly TRAIN, two independent text domains

All original pretrained teacher and original compressed baseline student forward runs occur as **separate processes**; teacher logits and packed-student true hidden activations are collected only for TRAIN fitting/validation. No original dense pretrained source matrices are present in any student runtime. Candidate `U,V,b` fitted against actual original FP32 teacher full-vocabulary output probabilities using the actual **PACKED model final hidden-state trajectories**, not original-teacher intermediate tensors. This addresses the severe v73 error propagation issue.

- WikiText original `train` FIT tokenizer tokens **12288:12544**, VALID **14336:14592**, each 4×64 windows, 252 position logit outputs.
- AG News original `train` FIT records **8000:8100**, VALID records **8500:8600** (text only, first 256 model tokens in each), each 4×64 windows.
- Same pinned model tokenizer and Hugging Face dataset revision metadata for all. Dataset labels are never read.
- Optimize rank4+bias using original teacher-probability cross-entropy / full-model KL on **two FIT corpora** with fixed Adam learning rate 0.035, L2 regularization 0.01 on bias and 0.001 on U/V, deterministic seed 76.
- Before TEST, retain snapshots after **20/50/90 optimizer steps**, each as either: (a) bias-only (same complete CB76 physical bytes, physically U=V=0), or (b) rank4+bias. Evaluate 6 fixed physical FP16-snapped correction candidates on **two heldout TRAIN VALID corpora**, choose max minimum baseline-to-candidate full-vocab KL gain; require both gains **≥0.10 nats**, otherwise choose a physically complete CB76 with all correction factors ZERO (explicit negative outcome). Threshold, iteration candidates, rank and regularizers are fixed now, not selected on TEST.
- Save frozen selection, actual FP16 factor shapes, checksums, dataset revisions and source; TEST datasets are not read during this stage.

## Two new completely independent TEST domains, frozen before fit

Only **AFTER frozen TRAIN VALID selection**:
- WikiText original `test` tokens **70000:70512**, 8×64 windows, 504 scored next tokens.
- AG News original `test[3000:3100]` records, text only first 512 tokenizer tokens, 8×64 windows, 504 scored next tokens.

Original FP32 pretrained teacher process, original CB59 native-packed C process and paid CB76 native C **plus rank4 CPU factor correction** process, all separate. Measure genuine full-vocabulary teacher→candidate KL, actual heldout next-token NLL, teacher top1 agreement, real CPU prefill times, paired bootstrap window uncertainty, all exact physical source file bytes and CRC/SHA. A model can gain on both KL and NLL but still have abysmal absolute language competence and unacceptable inference speed. Record numerical positives, negatives, marginal gains, fitting reversals and overfitting. **Do not use TEST to change snapshots or selection.**

## Stop/go

- Source integrity, one-file physical budget, no original trained dense 2D hot source and actual native-packed C execution are HARD PASS requirements. Corruption must fail parser.
- A two-domain TRAIN VALID selection can return all-zero corrections: honestly report **no model improvement** and do not force a change.
- Successful heldout KL/NLL in two domains is a research signal only; practical usefulness additionally requires larger heldout tasks, genuine autoregressive answer generation and CPU token latency comparisons.
- The v76 residual cost is near the 0.700bpw cap; a stronger heterogeneous bit allocator/learned compact factor decoder must demonstrate value well beyond an output-layer calibration of 49152 logits before claiming a general compact-weights breakthrough.

## Results

No measurements at preregistration; CI logs and artifacts will be checked and findings appended without suppressing failure.
