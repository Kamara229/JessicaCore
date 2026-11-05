# CALIBRI v185 — completed real mutable 5-bit index training (2026-11-05)

**Scientific outcome: FAIL actual coherent answer recovery.**
**Technical outcome: PASS complete 5-bit source invariants/native speed, and
TRUE code-index updates for the first time in this experiment series.**

- [GitHub Actions run #42076205631](https://github.com/Kamara229/JessicaCore/actions/runs/42076205631) completed SUCCESS.
- [Full raw artifact #13437829470](https://github.com/Kamara229/JessicaCore/actions/runs/42076205631/artifacts/13437829470) contains physically complete candidate source, unfiltered full
  generated answers and machine extraction results, original public
  frozen TRAIN protocol, 24-step gradient/swap audits, matched CPU timings.
- Draft research [PR #323](https://github.com/Kamara229/JessicaCore/pull/323).

## Physically authenticated source

Original pretrained checkpoint unique parameters **134,515,008**;
original CB59: **11,229,303 bytes**, SHA256
`50eac24ba7d7a452c495f8adced2cdef3f9cc44e024217fc21d8f2cebb87413c`.
v185 rewrites K32/V8 physical **5-bit group indices** and actual
FP16 codebooks inside the unchanged original CB59 file, with
every tensor, norm, row scale and alias fully accounted.

- Complete final source **11,229,303 bytes =
  0.6678394131307638 bpp**, below 0.700 limit.
- Sealed source SHA256:
  `df6f146fcc365b638737ddc569a9dcef8f5fbe23b88c66eb486d31431d6d71a7`.
- 24 precommitted whole-model TRAIN gradient steps.
- 12,288 original K32/V8 5-bit index swaps proposed; **10,240
  accepted after checking ACTUAL supervised TRAIN forward loss**.
  The final actual packed source has **10,240 changed 5-bit indices**.
- **53,842 FP16 codebook cells changed**. No model sidecar, raw
  1D tensor change, row-scale change, duplicate tied head or
  physically unpaid field.
- Same original native compressed CPU inference; no 2D original
  FP32 source resident at inference.
- TRAIN surrogate vs actually persisted native source logit
  relative error **5.0517e-7**, top token identical on original
  public TRAIN input. This verifies execution fidelity, **not QA**.

## Actual real answers, not likelihood or local weight error

| Fixed public DEVELOPMENT measure | Original pretrained FP32 | CB59 native | v185 native |
|---|---:|---:|---:|
| Physical total source bpp | 32 | 0.667839 | **0.667839** |
| First tokens matching FP32, 20 probes | 20/20 | 0/20 | **6/20** |
| DISTINCT first-token IDs on same 20 | 10 | 3 | **1** |
| Machine-confirmed correct answers /24 | 6/24 | 0/24 | **1/24** |
| Machine parseable answers /24 | 15/24 | 0/24 | **2/24** |
| Ungraded/ambiguous answers /24 | 9/24 | 24/24 | **22/24** |
| Hit 32-token cap without EOS /24 | 4/24 | 24/24 | **5/24** |
| Token-distinctness fraction | .895 | .051 | **.253** |
| Median native CPU speed vs CB59, 8 equal-work runs | n/a | 100% | **100.79%** |

**Critical manual semantic inspection:** v185 collapses to **exactly
one first token, `Yes`, for EVERY first-token probe**. 6/20
first-token "matches" are merely the original FP32 sometimes
starting with `Yes`; they **do not show recovered prompt
understanding**. All arithmetic, factual and notebook-binding
prompts produce repetitions of `YesYesYes...` until occasional
EOS. The sole machine-correct 1/24 is the trivial fixed-polarity
question **"Is 13 greater than 10?"**, answered `Yes`.
The paired opposite question **"Is 13 greater than 16?"** also
receives `Yes` and is marked wrong. This is a degenerate
constant-output predictor, NOT successful reasoning or useful
compression quality. Although EOS occurrence improved, EOS
alone is also NOT evidence of coherent answer quality.

The strictly measured speed/bytes and genuine index mutation
establish the feasibility of an operator-level source editing
pipeline, but the full compressed model is still unusable.
No claims of recovered QA, 6/20 genuine content matches or
generalization. The source cannot be promoted/deployed.

## Implication for the next experiment

Blindly training a small fraction of K32/V8 codes under
teacher-top-token CE can **increase a universal answer mode
instead of information recovery**. Future v186 must preregister
a genuinely balanced, diverse **behavioral TRAIN** target with
positive/negative pairs, require a *nondegenerate diversity gate*
on distinct public TRAIN/VALID prompts, and measure actual answer
quality against original FP32 on historical DEV. A useful
controlled study would contrast teacher-top-token pseudo-labels
with balanced supervised token targets and paired contradictory
answers, while using the physically costed index/codebook path
now validated by v185. Avoid reusing the exact 24 DEV prompts
for optimization or choosing checkpoint by their scores.
The index update fraction (10,240 of approximately 16.8 million
5-bit group codes) is tiny and also limits this pilot.

No KL/NLL/PPL acceptance criteria. All work on isolated
research branch, PR unmerged. No main, LIVE Jessica, Render,
Supabase, APK, Green Tunnel or independent private v89 TEST change.
Jessica relays only; never runs tests.
