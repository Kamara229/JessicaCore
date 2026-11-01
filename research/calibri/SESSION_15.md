# CALIBRI session 15 — fair matched-byte lower-complexity controls

## v14 found a genuine negative
On real pretrained Pythia-14m, real WikiText-2 **validation**:

- Original dev perplexity: **173.38149**.
- Quality threshold +1%: **175.11530**.
- INT4, physically 33,042 B: dev PPL **179.67859**.
- Best development PPL among the considered variants fitting a <=49,563 B
  budget was 3% sparse INT4 (44,838 B) with dev PPL **177.08932**.
- INT8, 65,810 B: dev PPL **173.55925**, but violates 49,563 B budget.
- **NO_APPROVED_VARIANT**. Never label v14 fusion as a quality-preserving
  compression. Actual independent test baseline and fusion results remain
  recorded in SESSION_14.md and v14 CI.

## New hypothesis being tested

Instead of immediately inventing more elaborate stacking, check whether
standard **5-bit or 6-bit quantization** achieves better end-to-end quality at
the **same full byte budget**. This is an important anti-cherry-picking
baseline: a novel combination that loses to a simpler method is not progress.

An exact row-quantized container supports signed 4/5/6/8-bit values with
2 FP16 scale bytes per row, 18-byte header and CRC32, including all padding.
For a 128×512 pretrained weight matrix expected container sizes are:

- 4-bit: 33,042 B
- 5-bit: 41,234 B
- 6-bit: 49,426 B
- 8-bit: 65,810 B

The 6-bit representation is *inside* the v14 constraint of <=49,563 B,
whereas INT8 is not. No fused kernels or performance claims.

Protocol: same trained source/module and same training, independent
validation and untouched test splits. Evaluate fixed-bit variants plus
the previously strongest **stacked** candidates on development PPL.
Choose smallest container with dev PPL <=1.01× original and <=1.5× INT4
bytes; evaluate predeclared controls plus selected on untouched test.

Negative outcomes, especially simple bitpacking beating elaborate
stacking, must be recorded. This cannot establish cross-model quality,
MoE generalization, 0.8 bpw, or novelty.

## Next directions after this gate

If 5/6-bit already wins, build a hardware-aware mixed-bit allocator and
compare it against existing mixed-precision quantization and GPU/ARM kernels.
For aggressive low-bit retention, train/reconstruct weights with teacher
logit distillation rather than optimizing only local activation MSE.
Only then test actual Switch MoE experts with routed activations.

No production or user-app changes; this is a standalone research PR.
