# CALIBRI v88 — physical 4D neuro-weight residual operator

**New original 135M pretrained experiment, not a design-only sketch.**
Strict isolation: research-only draft stacked on v87; never modify main, live Jessica, Render, Supabase or APK.

## Why v88 rather than more operator substitutions?
Real v87 Actions 41585113415 SUCCESS, SCIENTIFIC negative/complex:
- CB59 full model Wiki/AGNews TRAIN KL **11.30487 / 11.36656**.
- Swapping FP32 original pretrained lm_head only on compressed activations worsened KL to **14.58093 / 15.23263**, but 2-prompt real 16-token greedy distinct ratio increased **0.1875 → 0.46875**. This is a diagnostic, not recovered quality.
- Three raw FP32 MLP down projections worsen KL to **12.44628 / 11.48540**, diversity **0.09375**.
- Frozen physically paid adaptive CB85 small KL wins on two TRAIN domains but no meaningful greedy quality restored.
- Such raw original-FP32 oracles see the compressed student's different coordinate/activation distribution. Simply restoring original weights cannot be assumed to help.

## Preregistered genuinely multidimensional operator
Build from original fully packed native **CB59 11,229,303 bytes**, KEEP all original pretrained quantized tensors and original native-C packed GEMV. Replace ordinary extra dense correction idea with a **four-axis separable tensor residual** on the output projection. It learns entirely on compressed model's actual output-head activations:
\`ΔW[o1,o2,i1,i2]=sum_r A[o1,i1,r]·B[r,o2,i2]\`.
Vocab coordinate axes 192×256=49152; hidden 24×24=576; R=4. This is a 4D Kronecker-like trainable operator, **not** merely remapping a 2D weight image. Online contraction computes tensor delta; original CB59 native head remains.

Physical FP16 counts **18,432 A + 24,576 B + 49,152 bias = 92,160 actual values =184,320 bytes**. Header + CRC/SHA and entire original trained source always included in one SHA-verified CB88 file. **11,413,683 total bytes, ~0.6789 bpw**, <=0.7 cap. Original native C source untouched. Tensor extra is executed in Torch CPU, **not fused native C**; measure real tokens/s with full model. Tensor index factorization is not inherently semantically meaningful; it must win experimentally.

## New data and scientific protocol
Original **official TRAIN split only** (never TEST), freshly chosen WikiText train tokens [25000:25128) FIT, [27000:27128) VALID; AGNews train rows [10000:10040) FIT, [10300:10340) VALID, first 128 model tokens per region. Source dataset revision/token SHA recorded; AGNews labels never accessed. True original pretrained full model FP32 teacher; actual original packed CB59 native C student; cache native student's head activations + original target teacher logits. Learn only 4D A,B and paid bias with full-vocabulary teacher KL plus small NLL weight on ORIGINAL TRAIN. Predeclare checkpoints 0/8/16/24 and same-full-bytes bias-only negative control.

TRAIN-VALID guarded selection: candidate must beat original packed baseline KL by >=0.05 nats in **both corpora** and not worsen next-token NLL in either, otherwise guard selects no correction. Even if rejected, store actual physical experimental candidate and explicit negative report. Full-model physical source reloaded; compare original CB59, frozen unchanged CB85 and new CB88 under **real original TRAIN-VALID full model** with actual 16-token KV-cache greedy and CPU measured speed. Assert cached and physical actual scores agree to <1e-4. No independent unseen TEST yet; cannot claim regained QA, superior intelligence or production-speed parity.

## Facade structure
\`multidim_neuroweights_v88.py\` CLI → \`v88\` public facade → \`pipeline.py\` coordinator → \`sources.py\`, \`train.py\`, \`tensor.py\`, \`physical.py\` independent functional components. Four pure unit regressions. One dedicated CPU GitHub Actions workflow and saved physical + JSON artifacts. Do not merge or deploy.

Next step only after real TRAIN-VALID: independent unseen QA/KL/generation TEST of frozen source (on fresh original official TEST ranges) if guarded candidate passes. If negative, examine whether token-index geometry or more appropriate multidimensional coordinate construction is needed.
