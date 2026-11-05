# CALIBRI v186 — balanced opposing-answer gradient training, still 0.66784bpp

Previous v185 **real** 5-bit group reassignment produced a constant
"Yes" first token on 20/20 public development prompts. Although 6
tokens accidentally matched FP32 and one Yes/No question was graded
correct, it proved no useful semantic QA. The next hypothesis is
that **explicitly balanced contradictory behavior**, not a blind
teacher argmax pseudo-label loss, can discourage that easy collapse.

## Source and physical byte contract

Start with immutable original complete pretrained CB59
(11,229,303 bytes; 134,515,008 unique parameters, 0.667839 bpp).
Train the actual 32 FP16 codebook entries of ALL 211 matrices via
end-to-end whole-model backpropagation and change the 5-bit V8
group index codes in 96 precommitted operators spanning 30 layers.
The exact unchanged source format and per-matrix FP16 codebook
allocation are reused. Original FP32 norm vectors, stored row
scales, alias/tied embedding-output head, index bit width and
physical metadata size are frozen. Source is rejected if final
file has different physical bytes, lost original 1D values,
lost tied weights, missing CRC or bad SHA. Native CB59 K32/V8
C source inference only, no dense original 2D matrices in memory.

## Locked training corpus and independence

- Exactly **24 PUBLIC TRAIN contrasting pairs, 48 unique queries**:
  6 numerical yes/no comparisons, 6 fictional notebook possession
  yes/no, 6 arithmetic contrast pairs and 6 fictional color
  binding contrast pairs. One correct answer for each side and
  the paired expected first-token IDs must be DIFFERENT.
- Exactly **12 distinct held-out VALID opposing pairs, 24 queries**,
  all four domains, different numbers and fictional entity names
  from TRAIN. None of them is used for gradients, code proposals,
  accepting/rejecting mutations or selecting final model iterate.
- Historical 96-example development suite is disjoint from new
  TRAIN/VALID text. Existing precommitted 24 historical DEV
  questions and 20 first-token probes are used only AFTER
  the complete physical source file is sealed; none of their
  outputs/answers are passed to training.
- 96 fixed whole-model gradient steps = 4 shuffled passes through
  all 24 pairs. At each step optimize averaged full answer+EOS
  cross entropy for BOTH contradictory examples, plus explicit
  opposite-answer logit gap margin 1.5 weighted 0.20.
  A constant response cannot satisfy opposing answer gaps.
- 96 predeclared target matrices (all 30 attention O, 30 MLP
  down, 30 MLP gate and 6 attention V), max 512 proposed
  group-index swaps each step. Use earlier v185 exact projected
  code proposal and original pretrained *SOURCE weight prior*
  only; accept proposed source code swaps if the ACTUAL paired
  TRAIN objective improves. The source gradient and 211
  codebooks are joint end-to-end backpropagated.
- Preserve every negative result, including rejected code
  proposals, unchanged first-token predictions, EOS failure,
  throughput regression and unchanged model behavior.

## Two independent behavior checks after source seal

1. 12 new held-out VALID pairs: count a pair as responding
   differently only if the model actually changes its first
   token between the opposite questions. Count a pair as
   correct only if BOTH first-answer tokens are right.
   Store all first-token IDs, opposite gold answer IDs and
   eight actual generated texts (one pair in each domain).
   A constant Yes/No model fails this metric even if random
   sample-level accuracy looks 50%.
2. Frozen historical public DEVELOPMENT QA: original FP32,
   original native CB59, new native CB186 on all 24 actual
   prompts, cap 32, full verbatim text, EOS, conservative
   machine grade, 20 first-token comparisons. Eight matched
   native CPU speed pairs at exactly 32 tokens per run.

The actual model acceptance criteria remain **<=0.700 *complete
physical* bits per unique original parameter, original FP32-like
coherent/correct answers on diverse cases and >=90% of native
CB59 CPU speed**. Training cross-entropy/margin, KL, NLL and
perplexity are NOT quality gates. This tiny 135m pilot cannot
prove generalization to trillion-parameter models.

All changes isolated in research draft branch. Never touch
main, LIVE Jessica, Render, Supabase, APK, Green Tunnel or
private v89 TEST. Jessica remains **disconnected and never
executes any model test**.
