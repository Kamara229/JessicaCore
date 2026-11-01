# CALIBRI — session 13: stacked transforms, real model gate

**Date:** 2026-11-01. Research branch only. No production merge, Render, Supabase,
APK, model deployment or "Update memory" change.

## Research question

Do *combinations* of quantization, low-rank correction and selective sparse
correction yield better measured byte/activation-error choices than standalone
methods, with full serialization cost included? Does selecting an option on
training activations retain quality on independent held-out language data?

This is **not an invention claim**. Sparse+low-rank+quantized compression has
published precedents (SLIM, 2025; RS-MoE, 2026; TileQ, 2026). The distinctive
hypothesis for CALIBRI is joint **format+execution+quality+shared-cost**
allocation; **v13 is a controlled, incomplete pilot of only the first part**.

Relevant prior art:
- https://deepmind.google/research/publications/148040/
- https://www.sciencedirect.com/science/article/pii/S0950705126009706
- https://proceedings.mlr.press/v306/gu26c.html
- https://aclanthology.org/2026.findings-emnlp.1853/
- https://arxiv.org/abs/2310.16795

## Physical bitstream

Header: 18 bytes. Quantized base: rowwise signed INT4 nibbles or signed INT8,
2 bytes/row FP16 scale. Optional low-rank factors: U (rows × rank) and V
(rank × cols), both actual FP16. Optional sparse corrections: sorted uint32
flat index + FP16 value (6 bytes each). Full physical container includes
header, packed numbers, scale arrays, factors, every index/value and CRC32.
Decoder rejects corrupted data.

Compared methods include:
- INT4 and INT8 baselines;
- INT4 + 1% or 3% sparse FP16 corrections;
- INT4 + rank-8 or rank-16 FP16 low-rank correction;
- INT4 + low-rank + 1% sparse corrections;
- rank-8 factorization FIRST, then INT4 quantization of its remainder,
  plus optional 1% sparse corrections.

Quantization/factorization **order is explicitly tested**, not assumed
commutative. The low-rank and sparse corrections are *mandatory*, not
per-query optional. SVD uses full weight-space residual as first pilot,
and the sparse ranking uses training activation RMS. No claimed CUDA kernel.

## Protocol

Source: pretrained EleutherAI/pythia-14m (actual model revision pinned at run),
one first sorted FFN projection. Real WikiText-2 train tokens provide
**captured** module input activations. The independent WikiText-2 test split
provides held-out **whole-model perplexity** after replacing that one module.
Forbid synthetic substitution if data/model fetch fails.

Select variant *only using training activation error*, within 150% of the
physically measured baseline INT4 bytes. Predeclare held-out evaluations:
original reference, INT4, INT8, and chosen option. Calibrated candidate
Pareto set uses byte size versus normalized **squared error** of module
outputs. Model PPL is a separate check. JSON records model and dataset
revisions, train/test token hashes, exact candidate bytes and errors.

## Validity limitations

This is **one small dense trained matrix, not an MoE**, not GPTQ and not QMoE's
0.8 bpw result. The model computes with restored dense tensors, so no
latency/speed advantage is measured. This does not establish an innovation.
One-module PPL is not whole-checkpoint quality. INT4/INT8 are simple
per-row symmetric scales, not AWQ/GPTQ. Full checkpoint packaging is absent.
All results must preserve negative cases. The 150% budget is **relative to
INT4**, not the budget for the entire model.

Future experimental gate: repeat on pretrained Switch-base-8 with actual routed
expert inputs, a genuine end-to-end held-out task and matched-size INT4,
GPTQ/AWQ as feasible, QMoE if checkpoint access restored, plus real CPU/ARM
latency and shared-base accounting.
