# CALIBRI v64 — Same-rate binary latent geometry pilot on REAL pretrained weights

**Protocol committed prior to CI results. Research-only, no live system changes.** Parent v63 net lossless saving was only 59,971 bytes (0.534% of CB59), with 4.9749 marginal entropy per 5-bit index. Concentrate next on source representation and task-aware functional preservation instead of assuming stronger entropy coding creates large recoverable space.

## Primary scientific motivation
NanoQuant (ICML 2026) and LittleBit / LittleBit-2 (NeurIPS 2025 / ICML 2026) have already shown **sub-1-bit LLM compression** is possible using low-rank binary factorization + compensation, including much larger models. We cannot call our 0.7bpw packing a new absolute frontier; both competing methods are also empirically more relevant to quality preservation than conventional packet FEC. The research question is whether such latent binary geometry transfers to **this particular 135M dense trained model at fixed physical local matrix budget**, without model rebuilding or test leakage. This is not a reproduction of their full QAT/ADMM training and is NOT a claim to outperform published numbers.

## Preregistered real-matrix comparison
Original pretrained SmolLM2-135M-Instruct revision 12fd25f77366fa6b3b4b768ec3050bf629380bac; verified CB59 physical source from v59 Actions 41507201766. Four modules, selected **before** seeing outcomes: early attention q, early MLP down, middle MLP gate, last MLP up. Original trained teacher WikiText-2 TRAIN tokens 0–127 provide source activation inputs; WikiText-2 VALIDATION tokens 0–127 are disjoint and used **only** for heldout measurement. Fixed source/dataset hashes and REAL matrix tensor names go to JSON. This is a small pilot, not statistically significant task evaluation.

The original CB59 tensor remains identical as baseline. Reconstruct the chosen CB59 matrix only inside this diagnostic, measure:
- source weight relative Frobenius RMSE;
- observed teacher TRAIN input/output relative error, X_train(W-Q)^T;
- observed teacher VALID input/output relative error, X_valid(W-Q)^T.

## Candidate architecture
Given each actual matrix W (m,n), use CB59's existing FP16 row scales s (physically 2m bytes). Source-only randomized truncated SVD W/s ≈ U diag(singular_values) V^T at capped rank r≤192; select the maximum rank multiple-of-8 whose physical **candidate payload** cannot exceed CB59 local matrix payload (packed 5-bit codes + FP16 original row scales + FP16 K32x8 dictionary). This predetermined rank is never adjusted from validation error.

Convert left and right factors to **literal 1-bit sign matrices** P(m,r), Q(n,r). Two binarization geometries:
- baseline sign(SVD-balanced factors);
- source-only fixed-seed orthogonal latent rotation before sign, algebraically preserving unbinarized factor product.

For each geometry, compute 2 sets of per-rank gains g in float16, using (a) source-weight Frobenius least squares or (b) actual TRAIN activation/output least squares. Model W_hat = diag(s) P diag(g) Q^T. Every candidate counts two actually bitpacked sign arrays, r FP16 gains, m FP16 scales and 64-byte local header. The real decompressor/direct kernel is not implemented; this is a fair **local** bit-budget feasibility test, not a complete single physical checkpoint.

Choose a single best variant for each matrix strictly on TRAIN signal response relative RMSE, then report **heldout VALIDATION** response error relative to unchanged CB59 at the same local payload upper bound. Source-only singular vectors cannot encode hidden TEST information. Zero extra trained correction bytes exist; no secret teacher sidecar.

## Falsification and meaningful next stage
- Positive for a **pilot**: several of four heldout activation errors below CB59 with candidate payload≤baseline per tensor, while TRAIN selection and heldout remain disjoint. This is not an end-to-end model improvement and cannot be called viable 0.7bpw LLM inference until complete 135M physical packing, direct binary-factor GEMV, runtime and independent perplexity/task checks.
- Negative: if binary SVD sign factor at this rate fails on heldout, naive PTQ binary projection is insufficient. Next test should focus on iterative factor optimization (as in NanoQuant), latent-rotation initialization (LittleBit-2), or error-shaped task-aware source assignment, not simply use more bits or report lower isolated weight loss.
- Severe baseline condition persists: v61 achieved 0/8 next-token agreement with the original. An improvement of 5% on a few matrix responses alone will not solve original model quality.
- Licensing: published algorithmic ideas may be studied; code reuse must respect original project license. LittleBit repository currently lists CC BY-NC 4.0; this new script is original study code and does not copy their repository.

**Workflow status in this preregistration note is pending**, so no quantitative outcome is anticipated here. CI collects true real trained source + true teacher signals and full local comparison table in results/calibri_v64_binary_geometry.json. Source comes from an expiring Actions artifact; failed download must fail loudly, never silently switch to synthetic data.

Sources: NanoQuant https://proceedings.mlr.press/v306/chong26a.html ; LittleBit https://proceedings.nips.cc/paper_files/paper/2025/hash/a917c1d57088897beba47f96b4fd7f3c-Abstract-Conference.html ; LittleBit-2 https://proceedings.mlr.press/v306/lee26y.html ; official LittleBit source/license https://github.com/SamsungLabs/LittleBit ; previous v62 roadmap research/calibri/CODING_THEORY_ROADMAP_V62.md.


## VERIFIED GitHub Actions result — 2026-11-02, run 41525887887

A real public trained 135M teacher and physically pinned v59 CB59 source were used. Real WikiText-2 TRAIN/VALID provenance: dataset SHA b08601e04326c79dfdd32d625aee71d232d685c3. Four tensors, 4× combinations of binary geometry × scale-fitting; all local factor payloads below CB59 payload. The result is **mixed**, not a validated model-quality breakthrough:

| Original real matrix | CB59 VALID rel output error | Best TRAIN-chosen binary factor VALID error | Factor/CB59 | Factor rank |
| --- | ---: | ---: | ---: | ---: |
| Layer 0 attention q [576,576] | 0.4661 | **0.3198** | **0.6860** | 176 |
| Layer 0 MLP down [576,1536] | **0.6257** | 0.8018 | 1.2815 | 192 |
| Layer 15 MLP gate [1536,576] | **0.6624** | 0.6960 | 1.0507 | 192 |
| Layer 29 MLP up [1536,576] | 0.7978 | **0.7175** | **0.8993** | 192 |

Mean ratio **0.9794**, two tensors better and two worse, and individual errors still very large. All TRAIN-selected binary winners used a random orthogonal latent rotation plus TRAIN-functional least-squares gains, rather than the identity geometry. **Important confound:** the computational limit r≤192 left significant physically unused local source bits on 3 MLP matrices: candidate 52,288 B vs CB59 70,784 B (down) and 54,208 B vs 72,704 B (gate/up). Therefore the next controlled test should saturate the original per-tensor byte budgets at rank≈256 and compare *source-trained iterative geometry alignment* to both identity and random orthogonal rotation, while freezing heldout validation. No end-to-end LLM quality, full-archive sub0.70 rate, native binary runtime or speed is established by v64.
