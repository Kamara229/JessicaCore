# CALIBRI session 22 — creating new geometric weights, not just repacking

## Scientific question

Past sessions measured 2-core TT decomposition and a packed 5-bit
residual. v17/v18 found geometry highly sensitive to evaluation domain.
v16's 12-step teacher KL distillation on merely 128 tokens failed.
Question: can training the **native TT core weights themselves**,
while *not* training the residual bitstream, restore functional quality
better than decomposing a pretrained weight matrix alone?

## Fixed protocol

Use exactly one real pretrained EleutherAI/pythia-14m output MLP
matrix 128×512. Pick the previously determined geo-small geometry:
(m2=16,n2=32,rank=2,INT5 residual). Physical bundle 43,832 B.
Train **only 1,280 scalar TT core parameters** using a frozen
pretrained Pythia teacher. No gradient to the original 14M parameters
or the stored 5-bit residual. Use actual physical FP16-rounded
TT components with straight-through estimation (STE) during training,
and stream packed INT5 residual in the operator.

Predeclare 2 training objectives, no tuning based on evaluation:
1. Teacher next-token KL divergence only, temperature=2.
2. Teacher KL + 0.1×normalized local block-output MSE.

For each: 16 AdamW steps (lr=0.0005), 2,048 WikiText-2 train tokens,
context=128, 16 train windows. Capture original teacher logits and
original block input/output on TRAIN exclusively. Re-serialize learned
FP16 TT cores, keep the same immutable physical INT5 residual bytes,
verify serialized-vs-training operator numerical correspondence.

Compare untouched original, fixed INT6, untrained geometry, and both
trained geometries on WikiText-2 validation and TinyStories validation
using no adaptation to either split. TinyStories is an *already*
observed corpus from v18 and WikiText validation was used for prior
selection: both are **exploratory diagnostic**, not independent
generalization proof.

## Prior art, limitations and gating

The premise of learned tensor cores and teacher distillation is known;
this is **not** a new architecture proven by one 65K-weight test.
If teacher KL improves while validation degrades again, report a
negative result and avoid claiming success. Strong baselines and
real MoE experts are still necessary.

This run does not train a whole neural network from scratch or store a
model in an "image" file. It trains a compact multidimensional
parameterization directly. The compressed residual is used in
execution without expanding whole residual W; the one-row temporary
buffer is O(cols), 2KiB for this experiment. No device latency gate
is measured here. No main merge, Render/Supabase or APK changes.

## Measured real-model results (CI run 41337899789 — SUCCESS)

The actual number of trainable TT parameters is **1,280**: G1=(8×16×2)
=256 and G2=(2×16×32)=1,024. Earlier prose said 1,536; corrected.
Both trained variants physically round-trip at exactly **43,832 bytes**,
with output difference **0** versus the in-memory FP16 STE operator
for the checked training window.

Train-only teacher KL (2,048 WikiText-2 train tokens, 16 updates):
- KL-only: 0.024438 → **0.014433**.
- KL + local MSE: 0.024438 → **0.014253**.
The local block MSE also improved slightly for both.

| One replaced pretrained Pythia 128×512 matrix | Physical bytes | WikiText-2 validation PPL | TinyStories validation PPL |
|---|---:|---:|---:|
| original FP16 reference | 131,072 | 173.3815 | 37.8896 |
| simple INT6 | 49,426 | 173.2675 | 38.0677 |
| untrained TT rank2+packed INT5 | 43,832 | **173.4953** | 37.7828 |
| trained TT, teacher KL | 43,832 | **174.8014** | **37.7446** |
| trained TT, KL+local | 43,832 | **174.8268** | **37.7498** |

**Critical interpretation**: training decreased teacher KL on its
TRAIN examples but made WikiText-2 validation perplexity WORSE than the
untrained geometry. Both trained candidates performed slightly better
than untrained on TinyStories, another *already inspected* domain.
These results do NOT demonstrate robust generalization or a universal
quality improvement. The 1% WikiText development-quality cap is still
satisfied (174.80 < 175.12), but the previously observed WikiText test
regression was not retested for the trained variant. There is no
confirmation on an untouched domain.

Next step: multiple modules and authentic routed MoE experts with
hold-out routing/task quality; then native ARM/NEON speed measurement.
