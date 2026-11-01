# CALIBRI session 26 — independent pretrained MoE numerical replay (scientific gate)

## Why this single next step is needed
An ordinary Jessica chat cannot run a 90+ second pretrained-model experiment. The direct request to produce a new size/PPL/speed result via Jessica's normal chat timed out; this is NOT a model-quality failure. Experiments requiring pretrained weights must run under GitHub Actions in an isolated research branch.

The latest earlier evidence is not v23 alone:
- #148 / session 24: real `google/switch-base-8` precision budgeted across eight routed experts. Within-run quality/bytes measured, but two CI runs yielded different WikiText KL for uniform INT5 despite identical model and token hashes. One initially reported next-router mask metric was bitwise and was corrected to token-exact.
- #149 / session 25: loaded real pinned BF16 checkpoint, no missing/unexpected/mismatched keys, all eight expert pairs restored exactly. 3 repeats of original, zero router jitter, INT5 substitution and restoration per corpus were byte-identical **within one process on x86_64**. Artifacts: run `41341192826`. This does not establish reproducibility across independent machines or different Ubuntu images.
- #150 / session 23: a separate real Pythia-14m geometry gave 43,832 bytes and mixed diagnostic quality. It did not meet 0.8 bpw and no ARM benchmark was proven.

## Preregistered v26 check
Two **separate fresh GitHub hosted runners** (ubuntu-22.04 and ubuntu-24.04) load the *same immutable source checkpoint and held-out tokens* with identical package versions. They run the exact unchanged `moe_replay_v25.py` script with 3 within-process repeats and evaluate:
- identical pretrained checkpoint source model, revision, token SHA256 and source router/experts SHA256;
- no missing/unexpected/mismatched model weights, unchanged BF16-expert restoration and identical original/restored logits;
- within-run bitwise identical forward logits and actual router masks for two corpora under four phases;
- **between-run** bitwise identical output-logit hashes, router hashes and sample logit checksums for all phases.

Cross-run comparator writes a JSON verdict including differences; exits nonzero when a gate fails. Negative results are not hidden. Only small JSON reports uploaded; no model weights or secrets; no modifications to main, Render, Supabase or APK. Cross-run exact replay is a **reproducibility diagnostic**, not a claim that variation in numeric kernels is necessarily a correctness defect.

## Next decision after evidence
If the hashes differ, narrow down CPU/kernel BF16 differences using deterministic FP32 reference and separate weights restoration; don't compare tiny KL improvements between runs or claim Pareto novelty. If stable, run a separate REAL-WEIGHTS compression/size/WikiText PPL/native throughput study with new isolated train/test split, stronger same-byte controls, and explicit hardware details.

This session does NOT yet claim new compression, WikiText perplexity, mobile speed, QMoE 0.8bpw, real Switch downstream-task quality or trained geometric improvement.
