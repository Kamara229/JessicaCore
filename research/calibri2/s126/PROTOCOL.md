# CALIBRI 2.0 S126 — FULL-LAYER GREEN TUNNEL + GRADUAL 135M ROADMAP

**Draft / research only. No pretrained weights were transferred. No model has been enlarged, trained, deployed or accepted by the files in this PR. The first real S101 zero-gate CPU audit now runs separately in GitHub Actions; until its final artifact exists, zero-gate parity on the real physical model remains unverified. S125 real MPS workflow is not modified.**

## Why S126 exists

S87 original Green Tunnel owns **two trained neural residual blocks at decoder layers 3 and 9**. S111 already broadcasts those two donors to the other ten layers through 340 prompt-dependent coefficients. Thus all 12 layer outputs were touched by hooks, but information was not carried by a **distinct interlayer recurrent neural state** across all 12 layers, as requested.

S126 adds a new, explicit **sequential learned state** with a single shared narrow encoder/decoder (320 -> 8 -> 320), per-layer blending and individual residual gates on **all 12 layers**. Each layer receives the rank-8 summary derived from *preceding* layer outputs. Existing 3/9 GT and S111 donor hooks remain installed and frozen for the first stage. The new module is additive, not a replacement.

- New shared encoder + decoder: **5,120 parameters**.
- Individual learned gates and state-mixing logits: **24 parameters**.
- Total: **5,144 trainable S126-only logical parameters**; FP16 raw payload 10,288 bytes **before** safetensors headers and a counted runtime config. These are **not 5,144 new backbone/core parameters**.
- All gates start exactly at zero: 36M S101/S125 behavior must be **exactly preserved** before training.
- State is reset at each model.forward; **never persist it across requests**. Enable new tunnel during prompt prefill; disable for cached answer continuation until full 456 CPU proof.
- CPU synthetic unit tests verify *mechanism and identity*. A separate real S101 456+128 TRAIN physical CPU workflow checks the model-level extension and its actual speed/bpp; successful code compilation is not equivalent to that result.

## S126 research stage gate

1. Record final immutable physical S125 456/128 CPU evidence and S125 safe checkpoint if one exists; if S125 fails or does not gain, use immutable original S101 and label it explicitly. Do not read previously inspected S118 or other heldout QA.
2. On a **new, separate** research branch, install original GT3/GT9 and S111 hooks; install new full-layer state tunnel afterwards. On the *real* original 36,380,480-parameter CB61 CPU model with original S95 router/S98 copier, verify zero-gate **all 456 identical complete tokens including EOS** and original new S122 128 TRAIN baseline.
3. Physically serialize every new learned parameter including the new sidecar/config and original complete source, bank, router, controllers; compute **ALL-inclusive bpp <= 0.700000**.
4. Benchmark same-host alternating CPU original-S101 vs candidate complete decoding with the original S95/S98 KV path. **Speed >= 90% of original S101**. Full-depth hooks cost CPU time even at zero gain, so no speed claims before measuring.
5. Only after exact zero parity, train the new gates + shared low-rank state in bounded **1–2 MPS gradient steps** and use S125 atomic **optimizer+weights rollback**, with actual reserialized-and-reloaded full 456 identity verification after EVERY block. Protect all original S101 **353 full+EOS identities and 387 first-token identities**. Evaluate new S122 128 TRAIN full/first and both-correct 64 counterfactual pairs (baseline **14/128, 39/128, 0/64**), using S122 strictly as TRAIN, not as independent QA.
6. Fail closed on an old-skill regression, >0.7 all-in bpp, <0.9 CPU ratio, any frozen source mutation, incomplete test or exception. Training step success is NOT model acceptance. New unseen one-shot independent QA (not S118) and CPU/MPS parity/free-form speech are required before any quality claim or promotion.

## Progressive real-backbone growth, precommitted only (no automatic expansion)

**Stage A** is the real currently constructed core, not an extrapolated one. The following are **exact tied-embedding Llama geometry parameter counts**, computed analytically for the pre-existing 49,152-token vocabulary and 64-wide attention heads. No geometry after A has yet been built as a valid compressed model, so no compressed size, speed, retention or improved quality is implied by these numbers.

| Gate | Hidden | FFN | Decoder layers | GQA heads / KV | Total unique logical parameters |
|---|---:|---:|---:|---:|---:|
| A — current | 320 | 1536 | 12 | 5/1 | 36,380,480 |
| B | 320 | 1536 | 14 | 5/1 | 39,822,400 |
| C | 320 | 1536 | 16 | 5/1 | 43,264,320 |
| D | 384 | 1536 | 16 | 6/2 | 53,490,048 |
| E | 384 | 1792 | 18 | 6/2 | 63,125,376 |
| F | 384 | 2048 | 20 | 6/2 | 73,940,352 |
| G | 448 | 2048 | 20 | 7/1 | 86,263,744 |
| H | 448 | 2048 | 24 | 7/1 | 99,112,384 |
| I | 512 | 2048 | 24 | 8/2 | 116,417,024 |
| J — target | 512 | 2048 | 29 | 8/2 | 135,427,584 |

**Do not advance A→B unless S126 full-layer passes old-skill retention and TRAIN improvement.** B's first new decoder layers must be identity-initialized in a function-preserving manner *and physically repacked*, with no hidden dense FP32 fallback. The **entire 456 real physical S101 identity audit** is mandatory: expanded layer count alone neither preserves function nor teaches a skill. For widening D+, implement a separately verified shape-aware transfer; tied embedding/RMSNorm/attention and RoPE geometry change and naïve padding will not preserve old answers exactly.

**Critical throughput constraint:** increasing active decoder depth from 12 to 29 will likely fail the >0.9 S101 CPU speed requirement. Subsequent stages therefore need *measured* sparse conditional activation, shared-weight reuse, selective adapters or early-exit computation. Count only actual unique neural parameters; never inflate parameter count using unused dummy matrices or duplicate embedding aliases. Do not label a nominal 135M architecture as a qualified 135M model without both actual physical storage and quality verification. If an enlargement fails, **reject and remain at the last verified stage**.

## FP32 reference comparison (future independent lane)

Public pretrained **HuggingFaceTB/SmolLM2-135M** is a reasonable 135M *base checkpoint* candidate (Hugging Face model card: https://huggingface.co/HuggingFaceTB/SmolLM2-135M; Apache-2.0). Hugging Face lists it as pretrained on about 2T tokens; released weights are **BF16**, so loading them as FP32 creates FP32 arithmetic but **does not restore a higher-precision original checkpoint**. Be precise and record actual storage dtype, arithmetic dtype and revision SHA. The reference has ~134,515,008 parameters, within <1% of target 135,427,584.

Comparison must not leak target FP32 weights into CALIBRI, directly or through distillation unless independently authorized; retain the common 49,152 tokenizer only when verified compatible and use the SAME content questions and consistent output parsing. Independently frozen never-inspected QA, identical complete+EOS correctness, first/second/EOS failures, novel binding pairs, free-form speech, CPU throughput and serialized model size are required. Compare (a) target CALIBRI compressed, (b) unchanged FP32-cast pretrained model, and preferably (c) original available 36M FP32 initialization baseline when a genuine matching checkpoint exists. Never compare on TRAIN as evidence of generalization.

## Strict boundaries

- No changes to default/main, LIVE Jessica, Render, Supabase, APK or ordinary Jessica chat tests.
- Do not merge draft research PR into production.
- Do not run or inspect **S118**, already inspected and embargoed.
- Do not mistake successful synthetic tests, analytic parameter count or workflow green status for real CPU/GPU quality.
