# CALIBRI v50 — same-bytes activation-aware precision allocation

**Single research step**, following verified v49 PR #179. v49 frozen INT6 COMPLETE trained Pythia14M checkpoint 10,803,661 bytes plus exact physical balanced row recovery 610,715 B (6.49110587 bpw across all parameters). Both Ubuntu22/24 generated identical physical source AND sidecar SHA; original-vs-compressed full-model 128-token prefill +32 decode <=1.20 warm latency ratio; PPL improved on 3 first-use excerpts, but residual QNLI quality gap persists. Active runtime still has a decoded FP32 hot cache; this is INT6 AT REST, not packed-in-RAM.

## Frozen hypothesis, genuine rate-matched ablation

Keep the identical v49 SSE-balanced 12%-per-source-tensor corrected-row quota and all non-MLP row IDs. Re-rank the rows **only within the 12 pretrained Pythia MLP tensors** using the predeclared diagonal proxy for output error:

`score(row i)=Σ_j (W_ij−Q_ij)² × M_j`, `M_j=round(16×clip(E_TRAIN_A[x_j²]/median_k E_TRAIN_A[x_k²],0.25,4.0))`.

The input captures are **only WikiText TRAIN-A [0:2048]** original-teacher activations already captured offline by the fixed v46 PRE FP16 scale/code calibrator. This per-column input-importance proxy approximates output perturbation, unlike pure row SSE. M_j is quantized to fixed integer weight levels and ranking ties resolve by row ID. The 12 MLP per-tensor row quota is **IDENTICAL** to v49; no other tensor allocation changes and actual CB46 + CR47 bytes must match to the byte. No full-model gradient, validation loss, external test tokens, data tuning or labels influence the reallocation.

Physically store two alternatives from exactly the same frozen original trained full INT6 model:
- `sse_balanced_12pct`: proven v49 row IDs, 6.491bpw and per-module fixed quotas.
- `activation_matched_MLP`: identical actual physical CR47 byte size, identical per-module corrected row counts; only 12 MLP internal row choices differ if TRAIN_A activation weighting changes rank.

Require identical CB46 and both CR47 SHA, byte lengths, module quotas and chosen gate on Ubuntu22.04/24.04, with a third independent verifier. Bound true physically stored full-model checkpoint + chosen correction + **15,480 reserved POST bytes** by 6.5bpw. SHA/CRC/parent-SHA and shape checked BEFORE forward.

ONE train-only gate (same rule as v49): score model NLL on WikiText TRAIN-B [2048:4096] and newly disjoint AGNews **TRAIN[256:384]**; choose whichever contender has greatest minimum improvement over plain INT6 PRE by >=1e-5 on *both*, else choose PRE. Do not tune against test. Once chosen, measure **first-use new RECORD excerpts** from Yelp test[448:512], OpenBookQA test[256:384] and QNLI validation[256:384] (2048 Pythia token each), against teacher and both variants, show all negative domains. These are within already investigated DATASET FAMILIES, not broadly independent new benchmarks.

True whole-model warm Pythia generation: batch=1, original 128-token WikiText TRAIN_A prefill plus 32 teacher-forced KV-cache decode steps, exactly 2 warmups+7 repeats; target warm ratio <=1.20 on both CPUs. No claim about actual production Jessica, GPU, compressed 6-bit model in RAM, or larger-model scaling. Code and test in research-only draft PR; do not modify main, Render, Supabase, APK, secrets, or ordinary Jessica chat.

Improvement on new excerpts while preserving exact bytes and speed would support the sensitivity-weighted proxy. Negative or cross-host mismatch results are also decisive outcomes and must remain visible.
