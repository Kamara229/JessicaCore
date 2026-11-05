# CALIBRI v188 — actual original FP32 ability on 192 historical DEV prompts (2026-11-05)

**GitHub Actions #42088612519 SUCCESS**; all 192 real greedy
FP32 generations completed with actual pinned original SmolLM2-135M.
This audit measures the baseline's generated response capabilities
on a *reused historical development* suite. It is NOT an independent
blind unseen TEST, and does NOT benchmark any compressed weights.
No model parameters were updated.

[Source run](https://github.com/Kamara229/JessicaCore/actions/runs/42088612519);
[complete raw output artifact #13444347281](https://github.com/Kamara229/JessicaCore/actions/runs/42088612519/artifacts/13444347281).
The artifact includes all 192 verbatim generated texts, token IDs,
conservative automatic grading, paired results and prompt hashes.

## Overall conservative auto-grade

- 96 established public DEV questions in two pre-existing prompt
  templates (direct and brief): **192 model generations**.
- Machine-parsed responses: **135/192**.
- Machine-confirmed correct: **66/192**.
- **57/192 ungraded**, not automatically incorrect.
- **69/192 parsed but not confirmed correct** (135-66).
- 36 responses classified `candidate_gold_present`: a gold
  answer token appears but grader cannot confirm semantics. These
  are **not** counted as machine-correct.
- 9/192 hit the 32-token cap without EOS.
- Average emitted-token distinctness: **0.91534**.
- The above is NOT a calibrated independent-accuracy percentage.
  Conservative strict extraction can undercount semantically
  correct paraphrases or correct content in verbose sentences.

| Source original FP32 measure | Direct (96) | Brief (96) | Combined |
|---|---:|---:|---:|
| Machine-confirmed correct | **35** | **31** | **66/192** |
| Machine parsed | **72** | **63** | **135/192** |
| Ungraded/ambiguous | **24** | **33** | **57/192** |
| 32-token cap without EOS | **5** | **4** | **9/192** |
| Unique first token IDs | **30** | **26** | not pooled |
| Mean generated-token distinct fraction | **0.9395** | **0.8912** | **0.9153** |

## Actual domains (machine-confirmed / total)

| Domain | Direct | Brief |
|---|---:|---:|
| Arithmetic | 7/16 | 3/16 |
| Fictional binding, symmetric source | 14/32 | 13/32 |
| Yes/No opposite binding | 8/16 | 9/16 |
| Factual recall | 2/16 | 2/16 |
| Fictional ordering/comparison logic | 4/16 | 4/16 |

The factual subset has **13 ungraded /16 in each style**; do not
interpret the low machine-confirmed number as proof of near-total
factual ignorance. Original FP32 is clearly not reliably
machine-parsed on many examples. Substantial style sensitivity:
exact first token identical between styles on **66/96**
questions, verbatim answers identical **31/96**.

## Opposite-pair behavior

Precommitted matched **16 fictional color binding** and **8 fictional
Yes/No** opposite pairs per prompt style:

| Pair criterion | Direct color binding | Brief color binding | Direct Yes/No | Brief Yes/No |
|---|---:|---:|---:|---:|
| Both automatic answers confirmed correct | 0/16 | 1/16 | 0/8 | 1/8 |
| First-token ID actually differs between opposites | 0/16 | 1/16 | 0/8 | 1/8 |

First-token movement is only a diagnostic; valid semantically
different answers can share beginning tokens. The opposite-pair
results show the original 135M FP32 baseline itself can exhibit
strong response-mode/copying and does NOT reliably distinguish
reversed fictional bindings under strict output constraints.
This weak teacher matters when judging compressed model v186.

## Scientific implication

Earlier v186 genuine physical CB59-format source remained
11,229,303 bytes (~0.66784 bpp), speed ~99.6% original native,
and achieved 3/12 heldout exact first-token opposite pairs
and 2/24 historical DEV machine-confirmed answers, with
arithmetic repetition. v187 genuine 4D 2x4 source had median
~0.99906 weight-reconstruction error ratio and ~71.85%
single-operator native CPU speed vs CB59; no complete 4D
model/QA exists.

**Next go/no-go:** benchmark a measurably stronger original
pretrained teacher on the SAME 96 established prompts before
trying to preserve more behavior through 0.7bpp compression.
A teacher that does not consistently answer paired tasks
correctly is a poor target for strict distillation. Compare
semantics/grade limitations *with raw answers*, not only
aggregate parser counts; later use a separate truly NEW
test split for generalization. A stronger teacher's model
size and physical 0.7bpp budget must be recalculated using
its own actual unique parameter count; one cannot reuse
135M denominator or infer trillions-scale performance.

No NLL/KL/PPL quality admission. No private v89 TEST,
no main, LIVE Jessica, Render, Supabase, APK or Green
Tunnel mutation; Jessica was not asked to run tests.
