# CALIBRI session 25 — do not trust small MoE KL gains until replay-stable

## Why this is a scientific blocker

v23 and v24 loaded **the same public Google switch-base-8 revision**.
In two independent v24 GitHub Actions runs with identical model/data
revisions and WikiText/TinyStories **token SHA256**, the pseudo-
teacher-forced INT5 whole-model teacher-KL metrics were materially
different. Example WikiText INT5: 0.002717 in v24 first successful
run 41340130094 versus 0.002138 in corrected run 41340600560.
This is not accounted for by a changed codec or data, and the older
next-router one-hot bitwise agreement was also improperly interpreted
as token agreement. The latest v24 implementation corrects the router
metric: whole 8-bit mask equality per token, with bitwise reported
separately as a diagnostic.

These differences could stem from CPU BF16 execution, checkpoint
loading, unused parameter initialization, or some other unknown source.
We must FIND the source before asserting precise micro-Pareto gains.

## Real pretrained reproducibility audit

Load public pretrained BF16 \`google/switch-base-8\` pinned revision
\`92fe2d22b024d9937146fe097ba3d3a7ba146e1b\`.
Use Transformers 4.46.3, torch 2.5.1 CPU, 2 CPU threads, accelerate
low_cpu_mem_usage. Ask \`from_pretrained(output_loading_info=True)\`
for exact MISSING, UNEXPECTED and mismatched weight names. Store
hashes of one actual router and expert weights.

For each of two public corpus samples (WikiText validation and
TinyStories validation, 128 tokens/source), repeat exactly the same
forward under \`model.eval()\` **three times**:
1. Unmodified pretrained network.
2. Same pretrained network with router jitter set to zero explicitly.
3. All eight decoder block-1 expert pairs quantized to physical INT5,
   then decoded and copied into BF16 layers (no production bundle).
4. Exact restoration of all original BF16 expert weights.

Hash EVERY output logit and real next-router mask across all repeats.
Check whether original weights restore bitwise; check pre-quant baseline
vs restored baseline logits for the zero-jitter condition.

This experiment establishes only replay correctness. If any same-input
replay differs, cannot trust v24's per-run tiny KL differences; flag
the measured failure. If repeated forwards are stable within process
but between-run quality metrics differ, investigate checkpoint parameter
initialization and platform-specific kernels next.

No novel algorithm or task/phone quality claims; no merging or
deployment. The point is a solid gate before next steps involving
functional expert-graphs or self-optimizing weights.
