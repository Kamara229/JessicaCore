# CALIBRI v178 — completed paid column-scaled source experiment, 5 Nov 2026

**Decision: FAIL on real answer quality. Reject CB178 as an FP32-quality
candidate.** The physical source format and native executor pass their
engineering constraints, but the recovered model does not answer
questions coherently. Do not merge or deploy.

## Auditable completed measurements

- Draft research PR #315, original implementation commit
  `33b7c84b187b8ad92c604c64d4ba3d47529d10a7`.
- Completed targeted GitHub Actions
  [run 42026853526](https://github.com/Kamara229/JessicaCore/actions/runs/42026853526),
  job `127243098393`; native tests, full real-source fitting, 72 actual
  generations (24 each), matched-speed benchmark and artifact upload
  **succeeded**. This is not evidence of scientific success.
- Complete physical weights and raw results in
  [artifact 13414841437](https://github.com/Kamara229/JessicaCore/actions/runs/42026853526/artifacts/13414841437)
  (expires 2026-12-05).
- Original pinned CB59 SHA256
  `50eac24ba7d7a452c495f8adced2cdef3f9cc44e024217fc21d8f2cebb87413c`.
- Final complete CB178 SHA256
  `682a2fddc2c2363eb648c1d28e9b593426ff02d2ac2e87a427da4371a523d233`.

| Measured property | Original FP32 | Native CB59 | CB178 |
| --- | ---: | ---: | ---: |
| Physical complete archive bytes | Dense FP32 | 11,229,303 | **11,540,059** |
| Physical bits per 134,515,008 unique original parameters | 32-bit source | 0.667839 | **0.686321** |
| Matched 32-new-token native speed / CB59 | -- | 100% | **106.21%** |
| Unique first token IDs on 20 development prompts | 10 | 3 | **5** |
| First token matches original FP32 on 20 prompts | 20 | 0 | **0** |
| Confirmed machine-correct historical development answers | 6/24 | 0/24 | **0/24** |
| Outputs reaching 32-new-token cap without EOS | 4/24 | 24/24 | **24/24** |

**Do not report 0/96:** 24 development prompts per arm were generated
under the precommitted early-stop rule, because no candidate first-token
match to FP32 was found. The original FP32 machine 6/24 is not
exhaustive independent semantic quality (ungraded narrative cases exist).

CB178 generated different token patterns than CB59, but the texts remain
irrelevant repetitions such as `ellum`, `canoes`, `fname` and
`overty`. Increased token variety is **not** restored intelligence.

The full physical format includes 5-bit K32/V8 indices, FP16
per-matrix 32x8 dictionaries, FP16 row scales, **300,672 physical bytes**
of new FP16 column factors, FP32 original 1D vectors, shared alias
metadata, manifest, CRC and SHA256. The C-native GEMV scales activations
inside the LUT, and embedding scales only the decoded selected rows;
no dense pretrained weight remains in the compressed runtime.
Tied embedding/head identity and native-vs-literal numerical checks passed.
The ~106% speed is specific to eight warmed matched pairs on this CPU,
not a global benchmark.

## Full source reconstruction diagnostics (NOT quality gates)

Across all **211** source matrices, the unweighted mean original CB59
relative source-weight RMSE was approximately **0.727568**; the
column-conditioned CB178 mean is **0.726812**, a tiny difference.
Averaging by family:

| Family | Count | CB59 mean matrix relative RMSE | CB178 mean |
| --- | ---: | ---: | ---: |
| Tied embedding/output head | 1 | **0.6625** | **0.6769 (worse)** |
| Attention | 120 | 0.7276 | 0.7263 |
| MLP gate and up | 60 | 0.7279 | 0.7276 |
| MLP down | 30 | 0.7289 | 0.7290 |

These are approximate unweighted per-matrix means over original FP32
weights, not activation-aware distortion, and should not be interpreted
as accuracy measures. They show that a purely column-RMS source
preconditioner does not make the underlying K32/V8 representation
meaningfully more faithful, and makes the tied source *less* faithful.

The output-collapse problem is not the native numerical ABI or file
format: the actual **quantized source operators** have large deviations
from the source, and successful low-bit execution merely executes
those deviations faithfully. This does not mathematically prove the
global <=0.7-bpp objective is impossible.

## Stop rule and next research boundary

Stop treating small K32/V8 dictionary refits, single per-column source
scaling, or minor 4/5/6 bit reallocations as promising recovery steps
without stronger evidence. Both native C and byte-rate work.

A genuinely different source strategy is needed, such as **structured
operator factorization, cross-group predictive/delta coding,
low-rank-plus-sparse explicit paid residuals, or joint training of a
model designed to be representable in the code**, rather than merely
postcompressing an FP32 model. Before committing to a large QA run,
analyze the actual physical savings, numerical operator distortion,
and *correct first-token decisions* on held-out public training
examples; then evaluate full real answers against semantically
reviewed FP32 behavior with source <=0.7 and native speed >=0.9.

Do not use KL/NLL/PPL as release criteria. Do not open private v89 TEST
or change main, LIVE Jessica, Render, Supabase, APK, Green Tunnel or the
production app. Keep negative results and original source SHA pinned.
