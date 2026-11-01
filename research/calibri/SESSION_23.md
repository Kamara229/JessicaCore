# CALIBRI session 23 — TRUE routed pretrained MoE expert validation

## Reason and status

v13-v22 established physical format and direct native TT+INT5 computation on
one *dense* Pythia-14m MLP projection, but did not measure behavior for
actual routed experts. v22 revealed a negative: training TT cores improved
teacher TRAIN KL yet worsened WikiText validation. v19-v21 direct packed
execution became faster with SIMD, but remained slower than dense BLAS.

This experiment is a new empirical gate, not a new compression invention.

## Pinned source

Public pretrained \`google/switch-base-8\` (Apache-2.0), 8 experts in
\`decoder.block.1.layer.2.mlp.experts\`. Real model loaded in BF16 CPU with
\`low_cpu_mem_usage\`. Download ~1.24GB from Hugging Face only in GitHub Actions;
never commit trained weights to repository or alter Jessica production.
Dataset: separate WikiText-2 train and validation splits, genuine texts
but *artificial teacher-forced* decoder context, not Switch's original span
corruption training objective.

The actual \`decoder.block.1.layer.2.mlp.router\` dispatch mask is captured
alongside the real \`expert_i.wi\` prehook activation matrix. Require exact
agreement between dispatch counts and the real invocations of all eight
experts (capacity drops must be reflected). Report per expert train/val
token sample sizes; never pretend unobserved experts have validated
accuracy.

## Paired expert scope

Each expert has BOTH \`wi\` and \`wo\` matrices; quantize the *pair*.
For each of the eight experts compare, with all headers/scales:

- Regular signed INT5+INT5 (both matrices).
- Regular signed INT6+INT6 (both matrices).
- The two most frequent **TRAIN-routed** experts also get physical
  TT rank2+INT5 residual for both \`wi\` and \`wo\`.

Using *real dispatched hidden vectors* independently gathered on
TRAIN and validation, evaluate
\`wo(activation(wi(x)))\` vs decoded candidate pair. Record relative
output MSE; do not use validation data for method selection.
For the two geometric pairs, additionally execute the actual native
TT + C packed-INT5 reader and compare to exact same-bitstream dense
restoration, NOT to the original FP16 weights.

## Full-model teacher fidelity

For INT5 and INT6 separately, temporarily load decoded candidate weights
into the eight pretrained BF16 expert pairs of one decoder layer (other
layers untouched), and compare end-to-end teacher logits on held-out
validation context. Report teacher KL and next-token top1 agreement.
This uses restored dense BF16 weights in the full model because the
previous packed CPU reader is not yet integrated into T5's BF16 runtime.
It is NOT evidence of fast compressed whole-model inference.

**Critical limitations:**

- Pretrained Switch weights are not an official QMoE quantized checkpoint.
- Simple rowwise INT5/INT6 PTQ is **not** GPTQ nor QMoE.
- Decoder text is teacher-forced WikiText; comparing teacher logits is
  not a downstream task benchmark nor human-facing response quality.
- Some experts might be routed too rarely for meaningful estimates.
- Only one decoder MoE layer, not the complete model.
- No QMoE 0.8bpw claim, no Android/ARM/SSD measurement.
- A shared geometric basis is not assumed to work: prior experiments
  showed poor inter-expert ternary correlation. Only test after routed
  functional evidence.
- No main merge, deploy, Android APK, Render, Supabase, or live data use.

**Go/no-go:** do not assert MoE-ready, quality-preserving, fast or innovative
until each expert has sufficient held-out routed activation coverage AND
the resulting full-model inference and size/latency baselines are competitive.

Reference implementation in Hugging Face SwitchTransformers v4.46.3
uses a top-1 router whose capacity-limited mask controls real dispatch:
https://huggingface.co/docs/transformers/v4.46.3/model_doc/switch_transformers
Official source checkpoint:
https://huggingface.co/google/switch-base-8

## Measured outcome — GitHub Actions 41339418910 (SUCCESS)

Pinned public pretrained `google/switch-base-8`, revision
`92fe2d22b024d9937146fe097ba3d3a7ba146e1b`. Actual Google decoder
block 1 router and each expert wi pre-hook counts **matched exactly**.
Real 256 WikiText TRAIN and 256 WikiText validation source tokens,
4×64-token encoder inputs and 4×32-token decoder inputs per split.
The actual dispatch counts were:
- Train experts 0..7: **[19,19,12,29,14,12,12,11]** (135 total).
- Validation experts 0..7: **[8,13,9,19,21,17,20,21]** (128 total).

Only this decoder block's 8 pairs of `wi`/`wo` were compressed:
- Actual FP16-equivalent reference: **75,497,472 bytes**.
- Physical rowwise signed **INT5 for BOTH wi+wo: 23,654,688 B**
  (68.67% smaller than reference; 5.0136 bits/weight including scales/headers).
- Physical rowwise signed **INT6 for BOTH: 28,373,280 B**
  (62.42% smaller; 6.0135 bits/weight).

Temporarily substituting ALL EIGHT expert pairs with restored BF16
quantized weights produced end-to-end TEACHER logit checks on 4 pseudo-
teacher-forced validation contexts (not a downstream task):
- INT5 teacher KL mean per sequence: **0.00204029**, top-1 teacher
  agreement **96.09375%**.
- INT6 teacher KL mean per sequence: **0.000756338**, top-1 teacher
  agreement **98.4375%**.

All 8 paired experts had separate genuine routed activation-local
TRAIN and validation comparisons. Reported train relative expert-output
MSE for INT5 is markedly heterogeneous, ranging from **0.0001185** on
expert 1 to **0.00755495** on expert 4; INT6 ranged
**0.00001055** to **0.00185157**. The training activity of expert 3
was 29 tokens versus expert 7's 11, but on validation expert 3 got
19 and expert 7 got 21: frequency cannot replace a rare-expert
quality gate.

Two genuinely TRAIN-frequency-selected experts [0,3] were also tested
with **BOTH** wi and wo in physically packed TT rank2+INT5 mode
(2,997,872 bytes per expert pair) against plain INT5
(2,956,836 bytes per pair). Native tensor cores + streamed INT5 C
residual executed on actual router-sent validation inputs with max
absolute output discrepancy vs exact bitstream dense restoration
**6.10e-5** (expert 0) and **1.22e-4** (expert 3). However this
TT+INT5 was **larger** than INT5 and did not materially improve
routed pair output error. It cannot be claimed advantageous.

**Main finding**: real routed MoE experts display greatly different
sensitivity to precision. A uniform 5/6-bit representation is unlikely
to be Pareto-optimal under a whole-expert-layer budget. v24 must test
globally allocated mixed precision using TRAIN-only routed errors,
a strict rare-expert threshold, and then evaluate independent validation
and real teacher logits. Baselines must include uniform INT5/INT6.
This is an experimentally supported *research direction*, not a new
algorithm claim: expert-aware mixed precision is established prior art.

Results as GitHub Actions artifact (metrics only, no model weights):
https://github.com/Kamara229/JessicaCore/actions/runs/41339418910
Dataset/checkpoint revisions and full per-expert sample/error records
are in `calibri_v23_real_routed_moe.json`; SHA256 of that exact JSON:
`ecbc5daeceeb71d1fe3ab1d78dbce6807acce61f16253d28e3db4d04786e6f27`.

No main, Render, Supabase, APK or phone modifications.
