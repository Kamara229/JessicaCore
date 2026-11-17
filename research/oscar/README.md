# OSCAR — Original Source Compression & Architecture Reconstruction

**Oscar** is a separate research-only converter for full FP32 transformer weights into a physically complete CALIBRI CB59/CB61 compressed source. It has its own implementation, tests and entry points under `research/oscar/`. It does **not** edit CALIBRI, NRA/NRA2, Jessica LIVE, main, Render, Supabase or APK.

## Why

Prior actual 43M experiments:
- S7A: real trained CALIBRI 12×320, effective 43.26M architecture reconstructed exactly as same-source dense FP32 and full Green Tunnel12: **8/8 identical complete tokens+EOS**. Recompressing it with a newly fit weight-only K32V8 dictionary destroyed outputs: **0/8**.
- S7B: re-encoding the same FP32 with original trained 32×8 dictionaries and row scales restored **8/8**, all 4,546,560 original 5bit groups matched (source-linked inversion). Critically, this is NOT independent public FP32 training.
- Their **honest** rate is 0.661 bpp using the 43,261,760 structural denominator, but **0.786 bpp excluding 6.88M fixed/zero-heavy expanded structural parameters**. This is not a proven general ≤0.7 deployment.

## Oscar v0: actual independent API, not S7B script copy

Function `encode_fp32(state, *, mode, prior_blob=None, ...)` accepts a **real FP32 state_dict** with tied embedding/output alias. Its input is **not required to be previously CB59-compressed**. It assembles a new authenticated physical CB59 source with new raw FP32 1D tensors, aliases, CRC32+SHA256 and exactly all 2D matrices physically encoded. Three explicit methods:

1. `cold`: original source-only fresh K32V8 5bit dictionary fitting (works with any valid FP32 tensor state, but quality not guaranteed).
2. `prior_locked`: original **trained** compatible 32×8 FP16 centroid dictionary and exact original row scales retained, but all packed 5bit codes **re-inferred from the supplied FP32 matrices**; no old index stream is read for encoding. New raw 1D FP32 tensors are taken from supplied FP32, not from old checkpoint. Thus for a same-source dense FP32 reconstruction, original trained function is preserved.
3. `prior_adapt`: genuine previously unseen FP32 weights with compatible shapes: **recalculate each row scale from the supplied FP32** and fit FP16 codebooks by initializing original trained dictionaries and then minimizing current source's own 8D squared error with deterministic source-vector samples and FP16-snapped centroid updates; rebuild ALL codes and full physical source, not merely old original weights. This is an engineering possibility to handle previously unseen weights, **not a guarantee of pretrained FP32 language retention**.

All modes physically repack the original full FP32 input, preserve unique/tied aliases, check source tensor geometry (85 original 43M or 211 native 135M as appropriate), complete binary decoding and corruption protection. No FP32 weights or external codebook prior is needed **at inference**; an optional original trained dictionary prior is used by the offline converter only.

### Two independent checks

**Unit tests**: independent from any specific 43M artifact, synthetic and non-invertible novel FP32 weights, changed 1D tensors, mismatched-prior failure, full 5bit code bitstream, CB59 tamper reject, strict warm same-dictionary exact decoded outputs. These do **not** constitute meaningful model quality proof.

**Real 43M benchmark**: original pinned actual S142 43,261,760 architectural model (12×320), with original S93 base CB61, real S87 GT3/9, all-12 GT carry (Q2 zero), S142 real progressive rank896 48 active channels per layer, S94 triple bank, S95 trained routing, S98 trained second-token copier. Extract same-source FP32 and run Oscar `cold`, `prior_locked`, optionally `prior_adapt`. Measure all 12 last-token hidden state drifts, full output logits, actual incremental KV generate tokens including EOS, inherited original 8 TRAIN questions and full physical compression rate including one Oscar CB61, S142 sidecar, original S94+S95+S98 physical files, runtime manifest. Reuse only previously disclosed TRAIN (NOT sealed unseen QA).

**Important limitation:** existing 43M public original pretrained independent checkpoint has not been identified or evaluated; our exact FP32 mirror came from CALIBRI! Warm inversion of already-quantized FP32 does not show that Oscar can independently compress a new pretrained FP32 model. A 135M 30×576 model cannot be mapped to 12×320 by simple tensor relabeling: it needs teacher–student alignment/distillation and a different quantitative codec if equivalent quality at ≤0.7 bpp is to become possible.

## Gate

Never report Oscar universally successful merely because the CLI exits green or a same-source round trip matches. Required future scientific gate: *independently pretrained FP32* source, genuinely unmodified test set, real answer quality including EOS and distributional correctness, jointly trained Green Tunnel if needed, physical **conservative** bpp ≤0.7 including all inference weights/code and measured native CPU ≥90%. Oscar v0 will report both nominal and conservative bpp and **will not promote** a model with conservative budget failure.

All source modifications isolated to `research/oscar/` and its workflow; no NRA, deployment or Jessica chat tests.
