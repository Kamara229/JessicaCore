# CALIBRI v182 — original checkpoint, same-architecture decoded V1, native, FP16 and group oracles

The completed [v181 high-rate diagnostic](../results/v181_20261105/README.md)
did not recover real answers at 5.0361 bpp; none of the 20 frozen
first tokens matched the pretrained FP32 baseline despite reduced
per-operator source weight error. Before inventing a new <=0.7 source
format, this isolated diagnostic locates the first hidden-state
divergence from actual FP32 and checks native source execution.

**Precommitted four diverse existing PUBLIC DEVELOPMENT probes:**
question IDs 0 (arithmetic), 16 (notebook binding), 64 (factual),
80 (logic); use the fixed 'direct' chat template, greedy first-token
logits only. No answer labels, teacher logits or data are used for
quantization fitting. Original unchanged published FP32 weights and
the **exact physically sealed v181 V1 bytes** are inputs. No private
v89 TEST, no training, no model fitting, no JSON artifact
interpreted as ground-truth answers.

For each exact same input, run forward once on:
- FP32 pretrained SmolLM2-135M.
- FP16_ROUND: same architecture, all original FP32 2D tensors explicitly
  rounded to FP16 then back to FP32 (1D unchanged), independent control
  proving the model/prompt/copy mechanics.
- V1_DENSE: same architecture, 211 2D matrices reconstructed directly
  from the **exact sealed V1** source; all original 1D tensors retained.
  Temporary 32-bit decoded weights count as **unpaid diagnostic memory**,
  never as a <=0.7 or 5bpp physical inference candidate.
- V1_NATIVE: original physical V1 source in native C 5-bit K32/V1 lookup
  without any dense trained 2D tensors.
- V1_HEAD_ONLY: V1 decoded tied embedding/output head, rest FP32.
- V1_BODY_ONLY: FP32 tied embedding/head, other 210 2D V1 matrices.

Comparisons at embedding output, all 30 decoder block outputs, final
normalization and next-token logits report per-prompt relative L2,
cosine, max absolute difference and first checkpoint crossing 5%
relative deviation. Also check native V1 versus *exact same* decoded
V1 dense weights for every checkpoint, first-token IDs and numerical
difference. The check is diagnostic, not a KL/NLL/PPL quality gate.
The head/body counterfactuals are explicitly **unpaid oracles** and
never deployable candidates. All compared model weights, prompt
SHA256, source SHA256 and coarse per-group original errors are logged.

Interpretation:
1. If V1_NATIVE vs V1_DENSE disagrees materially, fix runtime binding
   before claiming information-rate limits.
2. If they agree but both deviate from FP32 already at embedding, the
   quantized tied input/head is suspect; contrast HEAD_ONLY/BODY_ONLY.
3. If early layers agree but later blocks diverge, identify the first
   major accumulation boundary and research source operator sensitivity.
4. In no circumstance call source RMSE, FP16 control PASS, or a native
   C roundtrip an actual answer-quality PASS. Final target still
   <=0.700 physically-paid bpp, original-FP32-like coherent answers,
   native CPU >=90% CB59.

Code lives only on isolated v182 draft research branch. No main,
LIVE Jessica, production Render, Supabase, APK, Green Tunnel or
private TEST changes. **Jessica must not execute tests** and may only
relay literal chat messages; experiments are owned by ChatGPT.
