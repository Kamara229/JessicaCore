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
