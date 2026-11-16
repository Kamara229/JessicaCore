# CALIBRI 2.0 architecture transfer to the trained FP32 135M (without NRA)

**New, independent research experiment — NOT the same as S1 four-cycle codebook training.**

## Scientific goal

Take **the already pretrained original** `HuggingFaceTB/SmolLM2-135M-Instruct` revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`, 134,515,008 unique parameters, and express it as an inference architecture based on the actual CALIBRI 2.0 research components **excluding NRA2, semantic/grammar adapters, and S3 node/role adapter**. Then compare its real generated answers to the *same original FP32 source* on literally identical frozen prompts.

This is an **architecture transplant**, not training the 43M scratch Calibri model and not changing the original 135M backbone geometry to 12×320. The 135M original has its own native depth/width; all learned-new-side modules must be compatible with those dimensions. Copied 43M learned sidecar weights may **not** be silently reused with incompatible dimensions.

## Inclusion contract — modules identified from original CALIBRI 2.0 code

1. **Compressed core**: all 211 real 2D source matrices packed in original CB59 (0.6678394131 bpp), physical same-tensor aliases and immutable single source, original configuration/tokenizer preserved. Source S0 SHA `50eac24ba7d7a452c495f8adced2cdef3f9cc44e024217fc21d8f2cebb87413c`; S1 four-cycle alternative source SHA `55e350047b1485f88c43451f5ae3fbff8481506702e2823a0734567b11cf1fda`.
2. **S87-style internal Green Tunnel**, real rank-8 neural residuals at ≈25% and ≈75% through the native core depth, starting as exact identity (e.g. native depth 30 => indices 7/22); explicitly trainable and physically saved; not just renaming centroid learning.
3. **S126-style full-layer recurrent Green Tunnel**, shared low-rank state carried consecutively across **every original layer**, zero initial residual gates, explicitly auditable coverage and scoped state cleanup.
4. **S94/S95-style compact profile bank and prompt-only router**: register S0 and previously TRAIN-only S1 variants, store **only changed FP16 centroids and 5-bit group indices** as a compact physically paid delta, never store a duplicate full second CB59 archive as an alleged below-0.7-bpp product. Router inputs are token statistics (not question IDs or gold answers); until independently trained and guarded, default profile S0 and report S1 as a **shadow ablation**.
5. **S159/S160-style bound parallel source slots**: one real hidden-state prefill, one vectorized two-slot source-span scoring head; never independently combine tokens from separate source records. Until trained and accepted the slot head is shadow-only; it does not replace generative answers.
6. **Parallel candidate generation**: independently generated same-core branches (greedy and two sampled sequences, shared batched execution) with gold-free consensus, explicitly reporting extra latency rather than claiming speedup.
7. **Protection**: frozen source provenance; a full-identity protected replay and snapshot/rollback contract for subsequent *actual joint* core+tunnel training; no extra model weight accepted merely because its loss went down. Training-only TEST/QA must remain sealed.

### Excluded: NRA completely

No `NRA`, `NRA2`, `S3` grammatical node linker, role/semantic adapters, S3/616 skills, or code/weights from that branch. No fake stub named NRA. CALIBRI features only.

## Comparison order (frozen BEFORE first evaluation)

- **A — FP32 trained 135M**, exact pretrained original, ordinary autoregressive generation.
- **B — direct CB59 135M**, source S0, same questions, generation and decoding.
- **C0 — CALIBRI architecture on B**, compact bank + zero-init internal GT + all-layer GT + router + parallel bound 2-slot head (shadow), original full-answer fallback. **Mandatory C0 token-for-token == B** on all tested prompts; otherwise architecture transplant has a bug.
- **C1 — C0 with S1 alternate source as shadow profile**, fair ablation of previous four training cycles; not automatically routed into production, not evidence of learning from QA.
- **C2 — three-path candidate generation on C0**, concrete model computations/latency, not FP32 guidance and not necessarily better.
- **D — protected joint core+GT training** of native 135M to be evaluated in a SEPARATE training pipeline after correct C0 physical/functional compatibility is established. No claiming D as implemented or evaluated in C0.

### Requirements and precommitted evaluation

- Frozen benchmark contains manually authored new questions, exact expected answers, SHA256 of full prompts before first inference. No original v231/v232 or S1 TRAIN prompts allowed as optimization labels. No model training or checkpoint selection on the benchmark.
- Real complete text output (at least up to 12 new tokens), EOS, first-token, exact normalized answer and exact cross-model sequence agreement (all failures preserved). Do not substitute KL/NLL/CI success for answer correctness.
- One complete physically packaged source including core, profile deltas, internal tunnel, full-layer tunnel, slot head, router, configuration and exact physical sizes **<=0.700 bpp calculated against the immutable 134,515,008 ORIGINAL unique-parameter denominator**; excluding the tokenizer/license/runtime executable as ordinary software, but **including all learned/producible inference parameters**. No FP32 model or hidden teacher at packaged inference.
- Actual output from PyTorch-reconstructed CB59 is NOT proof of native C CPU throughput. Real native C speed (>=0.9× CB59 baseline) remains an independent required measurement.
- Note: a zero-initialized/untrained GT preserving B is engineering equivalence, NOT improvement over B; an untrained router/slot cannot be called working skills. Evaluation of trainable architecture and quality recovery will follow separately.

**Safety:** research only / DRAFT / no merge / no LIVE Jessica, Render, Supabase, APK, no ordinary Jessica chat testing. Keep existing CALIBRI 2.0 and NRA projects untouched.
