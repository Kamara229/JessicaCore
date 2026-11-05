# CALIBRI v197 — Green Tunnel universal core + actual original-135M parity audit (5 Nov 2026)

**Three-job GitHub Actions #42107876507 SUCCESS, but source model correctness remains FAIL.** The primary experimental lesson is that we must improve actual restored source, not the answer grader or the size of the original model.

[Successful all-source schedule + parity CI #42107876507](https://github.com/Kamara229/JessicaCore/actions/runs/42107876507) produced:
- [Authentic trained v186 135M original 211-matrix recovery schedule artifact #13451573773](https://github.com/Kamara229/JessicaCore/actions/runs/42107876507/artifacts/13451573773);
- [Signed REAL v196 full model and original-FP32 16-case strict answer-parity audit artifact #13452598066](https://github.com/Kamara229/JessicaCore/actions/runs/42107876507/artifacts/13452598066).

## Measured physically paid status / strict answer outcome

| Quantity | Value |
|---|---|
| Actual original pretrained 135M unique parameters | **134,515,008** |
| Original previously trained real v186 archived source size | **11,229,303 bytes** |
| Actual physical paid source bpp | **0.66783941313** |
| Original 135M FP32 correctness (v193 16 public DEV questions) | **6/16** |
| Previous physically trained 135M v186 correctness | **2/16** |
| New v196 2-cycle Green Tunnel correctness | **0/16** |
| New v196 median native CPU speed vs original CB59 | **100.17%** |
| v196 heldout v186 first-token contrasting pairs both correct | **0/12**, versus prior v186 **3/12** |
| v196 complete physical source self SHA | **8e284672755c46c9cb874d9b5273aa5ad97bd7b9b589de414527a92711249b74** |

A native 135M QA answer such as 'red', 'redred' or 'redredred' to a math question is degenerate, not an intelligent response even if EOS is emitted. All 16 new v196 real DEV replies were not machine-confirmed; 4 were machine-recognized incorrect, 12 ungraded. Previous v186 had 2 machine-confirmed. The original FP32 baseline has 6/16; this is a small public development comparison, NOT an independently verified equivalence or comprehensive semantic accuracy.

## What new engine code actually accomplished

1. **Signed physical source and no hidden inference weights**: green_tunnel/contract.py rejects missing original unique parameter count, extra bytes, source SHA drift, altered tying or any source above a COMPLETE 0.700 bpp file budget. A future neural Green Tunnel sidecar must be physically counted if used; v196 used no sidecar.
2. **Generalized codec matrix adapter and deterministic schedule**: green_tunnel/adapter_cb59.py enumerated ALL real 211 original trained matrix entries from the v186 physical archive, validated the shared embedding/head alias, and successfully planned **211 unique physical index source update turns EACH cycle (422 proposed turns for 2 cycles)**. This schedule is a PLANNING result, not the actual v196 work: v196 only targeted 96 source operators per cycle, while applying global continuous gradients to all 211.
3. **Hard strictly frozen FULL-answer original-FP32 parity gate**: green_tunnel/parity.py was exercised on synthetic positive and negative controls, including all-YES, all-ungraded collapse, lost original correct case IDs, incomplete 16-case DEV, speed below threshold, unpaid FP32 runtime and failed EOS/diversity. The real v196 and v186 artifacts were then audited with the same scorer; **v196 fails** the genuine original-FP32 answer requirement.
4. **Explicit freeze of scope**: NO 360M or larger original-model experiments are authorized before original 135M achieves parity. Both historical 24 DEV and v193 16 DEV can inform source failure diagnosis but cannot justify scalability or final quality claims.

## Next source-level recovery design, without making unsupported claims

We have a proven failure mode: all-layer continuous teacher-guided source optimization can regress from 2/16 to 0/16 and collapse to one arbitrary output token despite decreasing TRAIN losses. Therefore merely increasing Green Tunnel cycles, swapping the original network or reducing file bytes is insufficient.

The next controlled source fit needs:
- **Continual-restoration trust region**: keep previous v186 as immutable signed source; protect already verified PUBLIC TRAIN response contrasts and EOS/first-token diversity during every cycle, using only TRAIN probes for intermediate guards. Revert a cycle instead of accepting source mutations that introduce a constant-token collapse; do NOT tune to historical DEV or independent TEST.
- **All-211 source coverage plus per-family sensitivity**: the actual generic schedule covers every input/output embedding, attention and MLP source weight, not only 96 previously selected matrices. Train with a model/codec adapter using data-efficient gradient samples, but do not assume full-parameter FP32 fidelity from a low training loss alone.
- **More diverse actual language supervision and possible paid capacity allocation**: full-vocabulary teacher distribution at a handful of positions is not the same as learning all real language. Increase public TRAIN text contexts and genuinely balanced answer examples. Independently audit layer sensitivities and evaluate physically paid improvements to the CB59 source representation using no more than the remaining **540,760 bytes** of full-source budget.
- **Real source-native QA parity before larger models**: preserve authentic original FP32 controls, full rendered replies, contradictory 2/2 pairs, extra ungraded, EOS and matched CPU speed, and reserve a NEW sufficiently large untouched precommitted independent QA set for final comparison. The current frozen v192 machine parser misses some correct full-sentence facts; require manual semantic review as well.

This v197 code implements and verifies physical source invariants, complete operator scheduling and strict no-scale checks. It does not yet train or create an improved 135M source. Its useful output is an actionable, experimentally grounded recovery-engine foundation with explicit negative results.

No main/LIVE Jessica, Render, Supabase, APK changes and no private v89 TEST access. Draft PR #336 remains unmerged; user has disabled Jessica and requested direct ChatGPT GitHub work.
