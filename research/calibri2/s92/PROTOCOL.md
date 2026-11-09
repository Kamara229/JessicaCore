# CALIBRI S92 — gradient-protected joint compressed-core / internal Green Tunnel repair

**Research branch only. Never merge or deploy to LIVE Jessica, Render, Supabase, APK or main.**
Predecessor S91: GitHub Actions run 42674413574; MPS GPU succeeds, 456-TRAIN CPU full/EOS guard fails.

## Measured S91, not a generalization claim

- S87 frozen 456-TRAIN correct full+EOS: **196**.
- S89 full+EOS: **225**; 63 newly correct, 34 lost relative to S87.
- S91 full+EOS: **257**; 86 newly correct vs S87 and **25 S87 full regressions**, **1 S87 first-token regression**. In addition to S89: +32 net full successes.
- Ten of twelve NEW S91 target full/EOS answers passed. Other S91 gains include 64 previously untrained rows **within the same TRAIN cohort**, not unseen QA.
- S91 physical all-inclusive **0.6847170790 bpp**, measured CPU speed ratio **1.00858065**; both budgets passed. No release: `candidate_accepted=false`.
- Complete 456 report: CI artifact `calibri-s91-full-456-strict-EOS-quality-no-forgetting-and-CPU` (artifact ID 13699884365). Trained weights: `calibri-s91-unaccepted-MPS-joint-quality-TRAIN-prototypes` (13699189487). Prior frozen S87 and S89 reports are required.
- Problem: averaging replay cross-entropy/margins over a few examples per step cannot ensure exact full-answer preservation on the 196 protected S87 identities, even when the training mini-guard improves.

## Proposed S92 falsifiable training protocol

1. Obtain the **frozen** S87 and S91 full 456-TRAIN audits, verify identity IDs, question and target hashes, complete paired contrasts, model weights hashes. Never open independent QA/TEST; do not reuse it.
2. Define protected **union** of all S87 and S91 previously correct complete answers plus their previously correct first tokens: 196 + 257 - (196 - 25) = **282 distinct protected full-answer identities**. This union is stricter than S91's baseline-only guard and must not be interpreted as 282 successes by a single model.
3. Start S92 solely from actual *physical* S91 compressed source and internal Green Tunnel; update only two physically paid core codebook dictionaries and internal rank-8 neural blocks; frozen first/later/pointer and output head, zero-gate CPU path unchanged.
4. Implement gradients on genuine **full-policy, whole-answer and EOS** teacher forcing. Separate candidate improvement gradient from sampled or prioritized protected-example gradients. Use first-order half-space protection: projected update `g` must satisfy `dot(g, grad(L_protected_i)) >= 0` for every sampled protected constraint, where SGD update is `-lr*g`. Include exact first-token and full-policy later/EOS margins. Return/skip any update failing constraints or nonfinite.
5. Prioritize the 25 measured S91 regressions and one lost-first identity, while rehearsing all S87 old-correct, S91 new-gained, newly targeted wrong paired contrasts. Do not train on or select by independent QA.
6. Serialize physically paid FP16 centroids and Green Tunnel **before** strict CPU inference and actual all-inclusive bpp accounting. Run *full* 456 TRAIN after GPU, comparing per-identity exact full/EOS and first-token output against the union. A new accepted checkpoint must have **zero lost protected full and first** and **at least one new strictly correct answer beyond union** (>=283/456), <=0.7 bpp, >=0.9 CPU/S82. If none, report honest failure; do not cosmetically accept a checkpoint. Any later independent generalization check needs a fresh preregistered sealed cohort.
7. Maintain research-only branch / draft PR; **no Jessica tests** or LIVE changes.

## First isolated code change
`gradient_safety.py` implements intersection-of-halfspaces first-order projection with a zero-gradient fallback when no verified feasible improvement is found. It is a *helper*, not proof of exact no-forgetting; strict physical CPU 456 guard remains mandatory.
