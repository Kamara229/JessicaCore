# CALIBRI 2.0 S102 — external research: why uncompressed models also struggle (2026-11-10)

## Scope and central inference

Comparative research review only; **NO training, no test, no change to
production**. Distinguish (A) language learning from random initial weights,
(B) task adaptation/retention, (C) compression-loss recovery. Improvements
on 456 deliberately authored two-word TRAIN tasks do not establish (A).

**The correct scientific answer is YES**: full FP32 models can train
slowly, memorize tiny tasks without generalizing, or forget previous skills.
A stronger diagnosis than "0.7bpp is the only problem" is that CALIBRI 2.0
has never been established to have the broad language foundation of
a heavily pretrained full-precision model. Extreme 0.7bpp **adds** a second
independent optimization/representation constraint.

## Published research and actual research consequences

1. **Pythia 14M and 31M** — EleutherAI training records show every Pythia
   size including 14M/31M was trained on ~299,892,736,000 tokens, ~143k
   steps and ~2,097,152 tokens per step. Tiny param count does not
   imply that a few hundred question-answer pairs suffice for language.
   Official model card: https://huggingface.co/EleutherAI/pythia-14m
   Repository: https://github.com/EleutherAI/pythia
   *Data used is an observed research recipe, NOT a proven minimal data floor*.
2. **SmolLM2-135M** (original pretrained model that V CALIBRI tried
   compressing) — official model card says trained on ~2 trillion tokens
   before further instruction tuning; the 2025 SmolLM2 report describes
   extensive data curation and multi-stage pretraining. This is
   fundamentally unlike random-start 36M CALIBRI 2.0.
   Official card https://huggingface.co/HuggingFaceTB/SmolLM2-135M
   Paper https://arxiv.org/abs/2502.02737
3. **Chinchilla** (Hoffmann et al., NeurIPS 2022):
   compute-optimal model width and token exposure scale together.
   The popular **~20 tokens per parameter** would yield
   `36,380,480 * 20 = 727,609,600` tokens for a 36M model,
   as a rough compute-allocation reference. This is **NOT**
   minimum training data, is extrapolated below the paper's
   smallest ~70M model, not a recipe for fluent generation,
   and may undertrain small inference-economical models.
   https://proceedings.neurips.cc/paper_files/paper/2022/hash/c1e2faff6f588870935f114ebe04a3e5-Abstract.html
4. **TinyStories** (Eldan & Li, 2023) — sub-10M-parameter
   *full-precision* models can generate coherent multi-paragraph stories
   after being trained on a *curated narrow vocabulary and simple story
   distribution*. This demonstrates that 36M parameters need not be
   inherently unable to produce coherent language, but narrow synthetic
   stories are not generalized assistant knowledge.
   https://arxiv.org/abs/2305.07759
5. **BabyLM 2024/2025** — research shows strong dependence on
   *data composition, objective and inductive bias*. Under limited
   words, short contexts and teacher distillation may help; many
   curricula produced little or no benefit. A small model does not
   automatically generalize just because its loss falls.
   https://arxiv.org/abs/2412.05149
   https://arxiv.org/abs/2504.08165
6. **Grokking** (Power et al., 2022): on tiny mathematical/algorithmic
   tasks, delayed generalization sometimes happens long after memorization.
   **Do not** interpret it as justification to run 10x more gradient steps
   on current 456 TRAIN responses: the phenomenon is task-specific,
   not a guaranteed path to full language ability.
   https://arxiv.org/abs/2201.02177
7. **MobileLLM** (ICML 2024): for sub-billion models,
   deep-thin architecture, embedding sharing and GQA can improve
   accuracy compared with older same-size baselines. These are plausible
   ablations for *future* small FP32 from-scratch comparisons,
   not reasons to mutate frozen S101 architecture mid guard.
   https://proceedings.mlr.press/v235/liu24ce.html
8. **BitNet (JMLR 2025)**: purpose-built 1-bit/1.58-bit
   quantization-aware Transformer can be *pretrained from scratch*
   with competitive same-size quality. This is proof low precision
   can be designed into training; however **1.58 bits/weight is
   >2x** CALIBRI's inclusive <=0.7 bpp, and ternary weights with
   training-time higher precision are not proof of our current
   5-bit codebook's semantic trainability at 0.7.
   https://www.jmlr.org/beta/papers/v26/24-2050.html
   Real BitNet b1.58 2B4T trained on 4 trillion tokens, illustrating
   that extreme numerical precision still demands enormous language data:
   https://arxiv.org/abs/2504.12285

## Direct repository evidence on FP32 struggles in OUR 36M network

- `research/calibri2/s0/train.py` explicitly documents **no original
  pretrained weights**; initial bounded S0 WikiText TRAIN source only
  `48*64 = 3,072` tokens plus 24 synthetic opposing TRAIN pairs.
  Later S stages added many more synthetic TRAIN examples but a
  **cumulative distinct natural-language token exposure through S102
  has not been measured**. Do not label 3,072 as whole-lifetime training.
- `research/calibri2/s30/matched_dense_fp32_train.py` used **the same
  36,380,480-param architecture**, unpacked from actual S26 weights,
  and learned new color+noun combinations with all FP32 parameters
  trainable; *not* an independently pretrained general FP32 teacher.
- S30 PR #412 and S31 PR #414 record actual FP32 adaptation:
  new TRAIN full+EOS correct **0 -> 4/16**, but previously learned
  historical TRAIN full+EOS **6 -> 4/8**. **Uncompressed FP32 also
  suffered forgetting**, validating a separate shared training problem.
  S31 192-step old replay with global 24-step transactions rolled
  all eight updates back without acquiring safe new answers.
  See https://github.com/Kamara229/JessicaCore/pull/412 ,
  https://github.com/Kamara229/JessicaCore/pull/414 and
  https://github.com/Kamara229/JessicaCore/pull/415 .
- S101 353/456 TRAIN and prior S100 42/64 narrow novel
  word-binding prompts reflect a special task-conditioned copier,
  NOT evidence that 36M random-start core has undergone Pythia-style
  next-token language pretraining or broadly generates fluent text.
- S102 real MPS 8 steps gave 353 -> 348 full TRAIN (five losses,
  zero gained), rejected by the J7 rollback. This is **not proof
  that compressed networks cannot learn**, and not a compression-only
  cause unless matched FP32 baseline with same input/control is
  measured.

## New controlled experimental design: separate language learning from compression

Research proposal only — no evaluation or weights changed in this note.

**A. Capacity and basic syntax check, same initialization**
- Freeze a 36M fresh/from-S26 numeric-matched source.
- A1: *full FP32* trainable model, all 36M unique weights; A2:
  codebook-constrained actual CB59 model, *same initial real forward
  replies, exact prompts, gold, token count, batch and training schedule*;
  A3: A2 plus existing Green Tunnel (account bytes).
- Use small authored TRAIN arithmetic, grammatical completion, 
  invented entity-binding and whole-text next-token samples.
- Measure learning curves, GPU steps, first semantic token,
  exact full+EOS, forgotten previously correct answers.
- If FP32 fails A1 as well, investigate data, label/tokenization,
  LR, optimization, task semantics and model capacity rather than
  blaming bit-budget. If A1 succeeds but A2/A3 fail,
  investigate compressive optimization/decoder loss.

**B. Actual language foundation, separate long-running scope**
- On frozen public TRAIN-only natural English tokens, pre-register
  a small progressive schedule (e.g. 1M -> 10M -> 100M distinct
  tokens as feasibility milestones), not asserted sufficient
  for strong assistant. Hold out new untouched text splits and
  group out near-duplicates. Track *actual processed nonduplicate
  tokens*, context lengths, epochs, loss and genuine free-generation.
- If full FP32 reference never becomes fluent on a constrained
  TinyStories-like task, the exact model/data/training implementation
  needs repair before asking 0.7bpp compressive teacher parity.
- Determine whether the optimization scheme trains all required
  real source matrices or merely a tiny prompt copier.
- Do not sacrifice frozen S101 S99 copier/353 success or old
  S100 QA embargo while testing language pretraining in a
  separate branch. No teacher-computed samples from external
  VALID/TEST and no QA sample reuse.

**C. Strict model-quality decision**
- First lock FP32 and compressed candidate SHA from TRAIN only.
- Compare **same architecture, source initialization and exposure**,
  while reporting FP32 uncompressed versus compressed byte costs.
- Need (i) actual grammatical free text/EOS, (ii) exact and
  semantically reviewed content accuracy, (iii) swapped answer pairs,
  (iv) retention identities, (v) native CPU performance and
  (vi) inclusive physical bpp. Report low TRAIN loss only as a
  diagnostic, never a surrogate acceptance gate.

## Quantitative caveat

727.6M tokens (Chinchilla-style 20x heuristic) / S0's 3,072 first
WikiText tokens = ~236,852x. **This is a comparison of early S0
training stage only to a rough extrapolated compute heuristic; it is
not proof that all later CALIBRI 2.0 experiments used 236,852 times
too little data.** Count all later actual exposure before any
cumulative training-data deficit claim.

## Scientific decision

Separate the next development into:
1. **S103 forensic full-sequence audit** of the rejected 8-step S102;
   no new QA, preserve S101's 353/456 frozen specialist skills.
2. **New matched 36M FP32/CB59 training-capacity branch** on a
   constrained language pretraining task; only that controlled
   comparison can partition basic trainability versus compressive
   optimization failure. This is distinct from S30 adaptation from
   S26 unpacked weights.

No LIVE Jessica, Render, Supabase, APK, main or ordinary Jessica tests.
No auto-promote from TRAIN or numerical losses.
