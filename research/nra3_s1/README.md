# NRA2 3.0 S1 — physically compressed first, Green Tunnel compatible (RESEARCH)

This step builds on PR #706 without modifying main/LIVE Jessica/Render/Supabase/APK. It deliberately does **not** touch the S3CI reserved independent sixth EVAL128, or re-use any of the five consumed heldouts.

## Physical representation

Source: one UNTRAINED `NRA3NeuralArea` (4 grammatical causal blocks, 3 semantic causal blocks, 6 recurrent relation rounds, hidden 256). The standard Calibri `direct_codebook_v59.py` **CB59_K32V8** encoder serializes every eligible 2D operator with 5-bit per 8D codebook-vector index, FP16 32×8 centroids, FP16 per-row scale, and exact source 1D FP32 arrays, including manifest, CRC and SHA. Every parameter and every physical byte is charged. This is the same **weight codec family**, not proof that the 6M area has the same inference performance or generalization ability as its parent. Readout weights are nondegenerate; strict `output_gate=0` alone guarantees initial output identity.

The physically persisted archive is read back with `CB59(expected_keys)`, SHA and complete-key verification. The only provided reconstruction path is a **cold dense FP32 test reference**: **not native packed CPU execution**. Quantization error is reported, never hidden, and an untrained compressed checkpoint must not be promoted.

Full budget `<=0.7bpp` may hold only if the COMPLETE verified original parent binary + the complete NRA compressed binary + all separate Green Tunnel, runtime, and manifest bytes, divided by all unique trained parameters, pass. Existing GT bytes must not be counted twice; any true all-in certification requires their physical SHA. The illustrative 0.69 parent-bpp scenario is not evidence.

## Existing mechanisms & hook ordering

The *same* existing `IntegratedGreenTunnel` is installed first on its original layers (3,9), followed if used by the existing `SharedInterLayerGreenBridge` all-layer residuals; the new NRA3 layer-8 read-only hook is registered last. NRA3 gets already-modified contextual hidden320; its outputs remain shadow-only. No duplicate GT weights, no edits to original routing/cache/CB59/CALIBRI checkpoints. Test runs all available GT routes (None, S87, S91, S93) with the authentic GT and S103 module on a TOY decoder and verifies exact unchanged output and GT weight SHA. It does not demonstrate real 43M Calibri native J7 compatibility.

## Required gates for training & integration

1. Physically seal original TRAIN2048 whitelist and new Russian UD TRAIN inputs, never old spent EVAL or S3CI reserved sixth. Train codebooks/index proposals and 1D source with constrained repeatability.
2. Establish native packed CPU kernel compatibility for *all* NRA3 operators (MHA projections and GRU included), without dense hot weight materialization. Cold reference is not enough.
3. Implement correct read/write residual at post-layer8 and test causal autoregressive token caching, parent GT routes and reload, without future token leakage.
4. Zero forgetting of original protected 616 J7, separately preregistered new full-answer+EOS generalization, all physical bytes <= 0.7bpp, same CPU native speed >= 90% original CB59 on matched workload. Only then consider candidate acceptance; no promotion before every gate passes.

`candidate_accepted=false`, `output_gate=0` throughout this step. No model quality claims from fitting the codebook of **random** untrained weights.
