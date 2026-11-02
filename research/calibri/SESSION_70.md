# CALIBRI v70 — preregistered 2D GQA neuron geometry and TRAIN-only end-to-end selection

**Scope:** isolated research PR only, pinned REAL pretrained SmolLM2-135M-Instruct. No changes to Jessica live, Render, Supabase, APK, main or secrets.

## Problem / falsification
v69 demonstrated a full-model *negative* outcome for source-only diagonal norm balancing of pretrained V/O pairs in GQA layers 0 and 15. At exact full physical 11,229,303 B=0.667839413 bpw, baseline original CB59 KL=10.0799, balanced KL=12.6706 and next-token NLL=14.4304 vs 16.5799 (124 VALIDATION positions), despite one locally improved attention output. We must stop extrapolating from local weight or matrix response error.

v70 asks whether a **2-dimensional, functionally equivalent coordinate representation** of real trained value neurons, plus TRAIN-only full-model selection, can beat the intact original 5-bit vector codebook source at **identical complete model file rate** and direct native execution. A candidate may harm quality; that is a negative result, not an integrity failure.

## Frozen mathematical candidates (before touching TRAIN/TEST)
SmolLM2 real GQA: 9 query heads, 3 KV heads, 64 coordinates/head. Within each KV head, use a **fixed orthogonal 2×2 Hadamard transform H** on successive V channel pairs:

    H=(1/sqrt(2))*[[1,1],[1,-1]]
    V' = H V  (independently for each pair)
    O' = O H^T  (for all 3 Q-head copies of that KV head)

Thus original attention softmax coefficients (depending only on Q/K) are unchanged, and exact FP32 GQA attention output remains identical. This transform is NOT valid if applied indiscriminately to arbitrary Q/K with RoPE. `H^T H=I` and 2D blocks are physically absorbed into the transformed V/O source weights: zero gauge metadata/inference operations. For diagonal source-only balancing D, `V'=DV, O'=O D^-1` similarly respects replicated O channels. **All source tensor dimensions, K32V8 5-bit packed codec, original C kernel ABI and tied aliases remain unchanged.**

Eight *predeclared* complete model candidates, all at EXACT original full physical 0.667839413 bpw:
`baseline`, `l0_diag`, `l15_diag`, `l0_hadamard`, `l15_hadamard`, `l0_diag_hadamard`, `l15_diag_hadamard`, `both_diag`.
All transformed source matrices use original deterministic CB59 compress_matrix algorithm, with its identical K32 FP16 centroids, row scales, 5bit-per-8 coefficients, physical CRC and SHA. Null identity re-encoding must reproduce original CB59 byte-for-byte; the complete archive must parse, every trained tensor and embedding/output tied alias accounted.

## TRAIN-only selection — one frozen independent test
The gauge transformations are defined by original trained weights only; do NOT use TRAIN to fit their parameters. Only choose among the eight frozen candidate archives using **original FP32 teacher vs native packed whole-model mean next-token KL** on WikiText-2 TRAIN tokens `[2048:2304)`, eight 32-token windows / 248 scored next-token positions. The rule is fixed before TEST: find minimum TRAIN KL; promote it only if it improves by more than `0.10` nats versus baseline, else keep original CB59.

After selection JSON is physically saved, examine **WikiText-2 TEST tokens `[16384:16640)`** (8 disjoint 32-token windows / 248 scored next-token positions). Run only original CB59 and the TRAIN-selected variant independently as native C direct packed whole model, together with FP32 pretrained teacher in a separate process. Compare mean teacher→student KL, heldout token NLL/perplexity, teacher top-1 agreement and normalized prefill CPU seconds. Bootstrap paired 8-window KL delta with fixed 1000-resample seed 70; warn sample is tiny and bootstrap tentative. No teacher weights loaded in packed processes; no pretrained dense original 2D tensors survive in runtime. Do NOT select by TEST results or make new variants after TEST.

## Hard acceptance gates
- For ALL 8 variants: **exact 11,229,303 actual physical source bytes / 0.667839413 bpw**, checksums and aliases valid. Every unquantized transformed GQA operator differs by at most 5e-6 relative (real pretrained source).
- Independently re-encode all 4 real original pretrained V/O source matrices exactly byte-equal as null codec fidelity control; no hidden precision or special partner model.
- Original and gauged packed runtime run with unchanged native C kernel, zero pretrained 2D dense source parameters and finite complete model outputs. An external full-model teacher is allowed only in its isolated teacher process.
- Report actual positive and negative KL/NLL regardless of outcome. Full-model competence, task QA, acceptable CPU speed **cannot** be claimed from 248 tokens; no speed SLA threshold expected in v70.

## Decision
If independent TEST KL improves, require larger independent text/task checks plus multiple seeds/layers; quality that remains catastrophic is NOT solved. If TEST regresses or TRAIN rule selects baseline, mark the gauge variant and/or selection rule unsuccessful, and pivot toward genuinely optimized multi-dimensional codebook/operator representations (e.g. binary latent tensors or learned source factorization) with proper TRAIN/TEST separation. Physical compression *alone* has already succeeded in CB59 but language quality does not.

## Result
PENDING first CI at preregistration. The workflow must record run ID and full signed deltas after completion, not assume victory.
