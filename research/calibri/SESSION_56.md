# CALIBRI v56 — real task-informed TRAIN safeguard and independent 64-item heldout answer evaluation

One isolated research stage after CALIBRI v55, PR #185 (GitHub Actions 41491488927, 3/3 PASS, independent physical SHA and full model CPU checks). v55 selected FP16 teacher-error repair with ~6.491bpw physical full pretrained Pythia-14M checkpoint but **selected sidecar changed ZERO answer correctness on 4 disjoint 32-item test sets** compared with the plain INT6 baseline. A small text-PPL gain is not task-accuracy improvement. The new research experiment MUST NOT use previously revealed v55 test labels or sample records for candidate selection. V56 introduces **one explicit real answer-accuracy safeguard built exclusively on TRAIN records** and doubles the number of never-before-used heldout labeled records per task to 64.

## All physical checkpoint and correction rules fixed

Original EleutherAI/pythia-14m pretrained real trained FP32 weights, v46 physically stored complete CB46 signed 6-bit 2D model with full 1D vectors (10,803,661 B) and MLP PRE calibration on WikiText TRAIN-A first 2,048 tokens. Three unchanged deterministic original-vs-INT6 row SSE/physical-byte selected CR47 FP16 residual correction sidecars with per-tensor 6% / 12% / 24% allocation caps, exactly **610,751 / 610,715 / 610,875** actual stored bytes with SHA256 and CRC. All correction files physically saved, read, corruption-tested and bound to original CB46 parent SHA before any TRAIN or external task selection. **Physical full model+repair+15,480 B reserved possible POST <=6.50bpw across all trained parameters**. Original teacher used only to compare external quality, never to inflate compressed-runtime speed or hidden inference. Cold load decodes entire INT6 checkpoint into dense FP32 active RAM ONCE, applies CR47 once; not a low-RAM native INT6 GEMM.

## Fixed independent training gate #1 — v55 unchanged

Identical v55 source-only 4-family bootstrap NLL acceptance logic over WikiText TRAIN [16384:20480], AG News TRAIN[1536:1792], SciQ TRAIN[512:768] and BoolQ TRAIN[256:512], two 2048-token disjoint windows per family. 1024 paired family/bootstrap samples, fixed seed 55, 10th-percentile NLL lower-decile threshold +0.0005, >=3/4 positive TRAIN family means, minimum family loss >=−0.008 NLL. No test dataset or TEST labels accessible to this gate. v55 gate is an eligibility *prerequisite*, not an irrevocable final selection for v56.

## Fixed independent training gate #2 — NEW actual QA labels on TRAIN ONLY

After the physical model and all CR47 alternatives are verified, read four **32-record labeled TRAIN partitions** never used by earlier CALIBRI selections:
- SciQ TRAIN[768:800] — no overlap earlier SciQ TRAIN up to 768.
- ARC-Easy TRAIN[:32] — TRAIN split not used for earlier quality evaluation; distinct from TEST and VALIDATION.
- SuperGLUE BoolQ TRAIN[512:544] — later than earlier TRAIN[:512].
- HellaSwag TRAIN[:32] — TRAIN split not previously used in CALIBRI.

Compute, for plain INT6 PRE and all three physical correction candidates, **zero-shot true answer selection** by causal model **mean per-token conditional log probability**, fixed prefix <=128 real Pythia tokens and option <=32 tokens; for SciQ support+question+answer candidate, ARC-Easy question+answer candidate, BoolQ passage+question+yes/no, HellaSwag context+one candidate ending. Each candidate scored separately, correct gold label excluded from prompt. Measure actual correct/32 on each TRAIN family. This labelled TRAIN accuracy is used ONLY for final **accept/reject** of an already source-only INT6 correction, never to change correction row IDs, postprocessing coefficients, quantizer, or speed mechanism.

A candidate is eligible only if (1) it passes gate #1 unchanged, (2) across 128 TRAIN labeled tasks it achieves **at least one MORE correct answer than plain INT6 PRE**, (3) no TRAIN task family loses more than **one correct out of 32**, (4) >=3 of 4 TRAIN task families do not regress. Rank eligible by total incremental correct across 4 families, then v55 bootstrap lower decile, then smaller cap percentage. If none passes, **choose physically smaller plain INT6 PRE** with NO residual sidecar. One frozen gating decision; must agree on two independent hosts. This guard is a heuristic and can overfit a sample of 32 per family; it is NOT statistical confidence or proof of no future regression.

## New heldout true answer quality — read ONLY AFTER final training choice

Exactly **64 new labeled items each**, four disjoint heldout record ranges, no overlap from v55 32-test sample:
- SciQ **TEST[160:224]**, previous v55 SciQ TEST[128:160] already exposed.
- ARC-Easy **TEST[32:96]**, previous v55 ARC-Easy TEST[:32] exposed.
- BoolQ **VALIDATION[288:352]**, previous v55 BoolQ VALIDATION[256:288].
- HellaSwag **VALIDATION[288:352]**, previous v55 HellaSwag VALIDATION[256:288].

Exactly same candidate token scoring and prompts/normalization/answer order as frozen v55, no prompt search. Record task identities SHA256 and actual per-item correctness arrays for original pretrained FP32 teacher and **ALL FOUR INT6 models** (plain PRE, 6/12/24 repair) even if they fail the TRAIN safeguard; avoid false quality claims if no improvement. 64 cases/task still too small for strong independent-domain confidence. The base Pythia 14m is not instruction fine-tuned or equivalent to production Jessica. Do not use these newly revealed heldout results to retune the gate.

## Physical rate and speed proof

Use two GitHub independent Ubuntu22.04 and Ubuntu24.04 CPU hosts with fixed Python, pinned CPU PyTorch, AVX2 thread cap, six-bit file CRC/SHA, source hashes and golden correction sizes. Independently verify both unchanged full CB46 and all 3 CR47 patch SHA/byte counts, 1024 bootstrap source arithmetic, 4 TRAIN answer-accuracy safeguards, real 64-item heldout QA item-wise correctness and result SHA. Third comparator independently rederives selection and reports negative quality honestly.

**User speed requirement**: full original Pythia teacher vs *physically restored TRAIN-selected* INT6 model, batch1 true 128-token prefill+32 autoregressive KV cache decode, 2 warmups+7 repetitions; maximum permitted selected/original latency ratio **1.20x** (<=20% slowdown). No "packed RAM" or production deployment claims.

**Isolation:** branch and draft PR only. Do not change production Jessica, normal chat, main, Render, Supabase, APK, settings, credentials or user data.
