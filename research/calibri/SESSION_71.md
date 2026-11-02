# CALIBRI v71 — truly independent cross-corpus validation of v70 selected multidimensional representation

**Frozen hypothesis before CI.** Last verified research result v70 PR #200: among eight original pretrained full-model variants at **11,229,303 bytes = 0.667839413 bpw**, TRAIN-only model KL chose **l0_hadamard** (9 query / 3 KV heads value/output GQA-preserving paired 2D orthogonal Hadamard). v70 blind WikiText TEST tokens [16384:16640), 248 positions: KL 11.54592 (baseline) vs 11.35587 (candidate), NLL 15.26959 vs 14.90752. Positive but extremely small, no usable language-quality claim. Earlier v69 source-only norm balance made quality worse. We do NOT retune v70 after viewing its TEST.

## What is measured
- Build EXACT original v70 source variants from pinned pretrained source using v70's unmodified source script; verify full model archive and source SHA/CRC, aliases and no pretrained dense hot weights in two separate student processes. Only compare `baseline` vs `l0_hadamard`, the **already frozen v70 TRAIN winner**. No new selection; no retraining.
- Two new heldout domains: WikiText-2 `test` tokens **32768..34815**, well beyond v70 TEST 16384..16639; and AG News `test[:300]` **text only**, first 2048 tokenizer tokens. Repo and dataset model revisions from official Hugging Face metadata are recorded.
- 16 contiguous 128-token windows per corpus, 2032 next-token target positions per corpus, total 4064. Each source evaluated in identical fixed windows. Nothing in these corpora is used to choose gauges / codebooks.
- Original trained model FP32 teacher logs source logits as on-disk **FP16 for storage** and records quantization in audit. Original CB59 and v70 l0_hadamard each execute full actual 135M in independent teacher-free native packed C direct LUT/GEMV process. Compute teacher-to-student mean KL nats, observed heldout next-token NLL and teacher top1 agreement, paired bootstrap 95% interval over the **16 fixed windows**, and exploratory warm CPU prefill wall seconds. No assumption that improvement persists.
- Fixed decision: new v70 mathematical transform is **cross-corpus directionally replicated** iff both KL and NLL improve on **BOTH** corpora. CI should succeed on valid measurements even if all gains reverse. Tiny sample plus correlation of windows means bootstrap intervals are diagnostic, not formal significance.
- No full task QA, no real multi-device speed benchmark, no full weight-model competence claim. 0.7bpw physical source without quality remains a research prototype.

**No changes to production/main/Jessica/Render/Supabase/APK.** New branch research-only, PR draft. Original pretrained source artefacts downloaded temporarily in CI and *never* uploaded to GitHub source or production.

## Stage status
Pending verified CI run, report signed deltas and losses in this file after run. If cross-domain replication is negative/mixed, abandon universal gains from untrained 2D gauge; investigate TRAIN-fitted function-aware multi-dimensional factor source and mixed K32/binary codec, still native runnable and complete bit accounted. If positive, move to task benchmarks and 2nd open checkpoint rather than claiming learned intelligence.

## VERIFIED REAL execution: GitHub Actions run 41533456717 (2026-11-02)

**CI PASS for actual pinned model + true direct K32V8 archive execution** in 3 independent teacher/student processes per corpus. Original and v70-l0_hadamard source files are **both physically 11,229,303 bytes = 0.667839413 bpw**. Untouched **TEST** spans: WikiText tokens [32768,34816), AG News official TEST text [0,2048) by model tokenizer. Each domain 16×128 contexts = 2032 independently scored next-token positions; all evaluations independent from the earlier v70 TRAIN-only candidate choice. FP32 original teacher logits were temporarily stored as FP16 to avoid huge memory; metric includes this explicit measurement approximation.

| Heldout domain / measure | Unmodified original CB59 | Frozen v70 l0_hadamard | Difference (candidate − baseline) |
| --- | ---: | ---: | ---: |
| WikiText TEST teacher-to-student mean KL (nats) | 11.39294 | **10.80034** | **−0.59260** |
| WikiText TEST heldout NLL | 15.04508 | **14.33451** | **−0.71057** |
| AG News TEST teacher-to-student mean KL (nats) | 10.64398 | **10.56289** | **−0.08109** |
| AG News TEST heldout NLL | 14.35634 | **14.28288** | **−0.07346** |
| WikiText CPU full 2048-token window prefill time | 42.786s | 43.065s | **+0.279s, slightly slower** |
| AG News CPU full 2048-token window prefill time | 42.878s | 42.962s | **+0.084s, slightly slower** |

**Bootstrap uncertainty (paired 16 windows, fixed 2000 resamples):**
- WikiText KL difference 95% interval **[−1.42861, +0.12327]** nats.
- AG News KL difference 95% interval **[−0.52545, +0.37769]** nats.

**Important scientific result:** In both independent domains, the frozen source geometry shows improvements in *mean* KL **and** NLL at constant physically measured archive bytes and unchanged native packed kernel. **But both paired 95% bootstrap intervals cross zero:** there is not enough evidence here to claim a robust nonzero population quality effect. AG News improvement is small. The source compressed model still has catastrophic absolute NLL ~14+, which does **not** establish usable language competence or conversational intelligence. Native prefill is not faster; these isolated ~43s times are exploratory, not latency SLA. The benchmark did not test answer QA accuracy or autoregressive generation.

**Decision:** Keep the fixed layer0 Hadamard transform as a validated *candidate*, not a deployed feature and not a proof of model-quality rescue. Next checkpoint v72 should quantify task correctness (BoolQ/SciQ etc), more windows and a genuinely different pretrained model/architecture, with fixed selection and no retuning on these TEST records. Simultaneously investigate directly executable multi-d operator decomposition/binary factors, but compare full checkpoint bytes and throughput rather than isolated matrix RMSE. Record every regression and do not silently hide uncertainty.

Reproducible audit run: https://github.com/Kamara229/JessicaCore/actions/runs/41533456717 ; original PR #200 is a draft research reference, v71 PR #201 remains isolated draft.
