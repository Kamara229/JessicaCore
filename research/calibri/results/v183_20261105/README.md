# CALIBRI v183: first-block operator counterfactuals (5 Nov 2026)

**Successful execution, but NO answer-quality improvement.** The seven
single-operator and three group restore interventions are UNPAID
diagnostic source copies, NOT compact 0.7-bpp candidates.
[Successful GitHub Actions run #42070471102](https://github.com/Kamara229/JessicaCore/actions/runs/42070471102).
[Complete raw outputs #13435826317](https://github.com/Kamara229/JessicaCore/actions/runs/42070471102/artifacts/13435826317).
3 independent unit tests and all 11 actual pretrained model arms passed.

Original model SmolLM2-135M; pinned V1 source 84,679,302 bytes
(5.036125 bits / original parameter), source SHA256
`1832c303f2655ee52d335ea8a3cbdda76756abcacc598573f4462cf7703451d8`.
Four frozen PUBLIC DEVELOPMENT prompts (0,16,64,80).
Original FP32 tied embedding/output head throughout; 210 other
body matrices V1 except restored first-block matrices.

| Unpaid temporary FP32 restoration | First-block hidden relative L2 | Change vs V1 body | Final logits relative L2 | FP32 first-token matches |
|---|---:|---:|---:|---:|
| none: V1_BODY_ONLY | 0.404643 | baseline | 0.394559 | 0/4 |
| q_proj(0) | 0.407636 | **-0.74%** | 0.365084 | 0/4 |
| k_proj(0) | 0.401856 | +0.69% | 0.309469 | 0/4 |
| v_proj(0) | 0.391849 | +3.16% | 0.306647 | 0/4 |
| o_proj(0) | **0.354884** | **+12.30%** | **0.632735** | 0/4 |
| mlp.gate_proj(0) | 0.432801 | **-6.96%** | 0.407499 | 0/4 |
| mlp.up_proj(0) | 0.419121 | **-3.58%** | 0.512674 | 0/4 |
| mlp.down_proj(0) | 0.390283 | +3.55% | 0.328205 | 0/4 |
| all 4 attention projections of layer 0 | **0.354053** | **+12.50%** | **0.682330** | 0/4 |
| all 3 MLP projections of layer 0 | **0.699049** | **-72.76%** | **0.903021** | 0/4 |
| ALL 7 projections of layer 0 | **0.000000** | **+100%** | **0.601920** | 0/4 |

Positive 'change' indicates smaller **local** hidden-state error,
NOT improved language-model answers. These are last-prompt-position
metrics; no full answer was generated in v183. Original v181 tested
24 DEV answers per compressed source and found 0/24 machine-correct.

**Central result:** restoring full FP32 first block exactly eliminates
the local block-0 error, but **does not restore a single next-token
prediction** (0/4), and increases final logit error from 0.395
to 0.602. Single O-projection repair improves local block output
by 12.3% yet worsens final logits by 60%; repairing the entire
first-block MLP alone worsens local error substantially.

This **strongly argues against independent first-layer 'fix the
largest error' heuristics** for the present global K32 quantizer.
The measured subsequent V1 layers are affected by the distribution
of earlier hidden states, and local fidelity is insufficient.
The cause is pretrained operator representation/coordinated behavior,
not the native runtime (v182 matched native and exact decoded PyTorch
V1 at every layer within 2.93e-5).

**Next research direction:** joint, whole-network adaptive
compression/training under the *fully paid* <=0.700 bpp envelope,
evaluated on real diverse QA and native CPU. Not more unpaid
first-layer FP32 patches or uncalibrated per-matrix codebooks.
Record all operator codebooks/indices, residual/adapters and
any Green Tunnel weights explicitly; do not silently omit costs.
NLL/KL/PPL are not final quality gates.

Only isolated draft research PR; no main, LIVE Jessica, Render,
Supabase, APK, Green Tunnel, private v89 TEST or Jessica model tests.
