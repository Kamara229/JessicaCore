# CALIBRI session 28 — is deterministic PyTorch execution causal?

## Why a new experiment is needed

Session 26 (Actions 41350732476) demonstrated **negative** exact
cross-host replay: even unmodified Switch teacher BF16 logits differed
on Ubuntu 22.04 and 24.04 despite matching source checkpoint and
evaluation-token SHA256. Session 27 (Actions 41351926001) then achieved
**exact cross-host** full-logit and downstream expert-router hashes
on BOTH corpora for both BF16+MKLDNN and FP32+MKLDNN-off. Source
invariants passed. This CONTRAST is not yet a causal explanation:
v27 changed PyTorch's deterministic-algorithm option, forced router
jitter zero before its first forward and used a different seed; and
its FP32 control simultaneously changed dtype and MKLDNN. Do not
attribute the result to one factor without a controlled test.

The measured within-host BF16+MKLDNN versus FP32+MKLDNN-off differences
were LARGE: WikiText 24-probe maximum absolute logit delta 7.563
and TinyStories 7.208, full-vocab top1 agreement 92.19% / 95.31%,
downstream router exact per-token agreement 100% / 98.44%.
Those are deliberately DIFFERENT arithmetic backends, not evidence
that quantized CALIBRI weights are wrong or any new compression result.

## Exact v28 single-factor design

Use the same pinned pretrained `google/switch-base-8` BF16
checkpoint (SHA 92fe2d22b024d9937146fe097ba3d3a7ba146e1b),
same WikiText/TinyStories validation token samples and SHA,
same training-free model.eval, same seed 27, same two CPU threads,
the SAME BF16 parameter tensor values, MKLDNN **ON** throughout,
zero router jitter **BEFORE** first forward. On each of two
fresh GitHub runners (Ubuntu 22.04 and 24.04), execute three phases
without reloading or changing weights:

1. `torch.use_deterministic_algorithms(True)`, original BF16 forward.
2. `torch.use_deterministic_algorithms(False)`, same BF16 forward.
3. `torch.use_deterministic_algorithms(True)`, repeated control.

For each corpus and phase, capture exact full-logit and next-router
8-bit mask SHA256; 24 evenly spaced vocabulary logits per token; full
vocabulary top1 IDs; per-token full-logit norms and exact router mask.
Compare within host and across both hosts with quantitative maximum
absolute probe delta, RMSE, top1 and whole eight-expert router agreement.

## Interpretation gate

If both modes are stable across hosts, the deterministic flag cannot
by itself explain session 26's discrepancy; investigate jitter order,
seed-dependent routing and identical-container replay. If only
deterministic mode is stable, follow up with a repeated trial to
exclude other host/platform differences. If on→off→on changes
the third ON result on the same model, investigate hidden state first.
Do not call numeric differences a bug without further analysis.

The GitHub workflow **fails closed** for model loading, unavailable
corpus, invalid or nonfinite logits, source hash/shape mismatches.
Numerical disagreement is accurately logged as negative evidence,
not quietly discarded or presented as success.

No novel 0.8 bpw, Switch task PPL, compression improvement, speed
benchmark or Android deploy is claimed. This is *one targeted
diagnostic step* on real pretrained weights in an isolated branch.
No main, Render, Supabase, APK or ChatGPT ordinary Jessica test changes.
