# CALIBRI v176 — completed 5 November 2026

**Verdict: FAIL on answer quality; PASS on compact bytes, native numerical
roundtrip and measured speed.** Joint training of the existing paid source
fields did not restore usable answers. This is a completed negative experiment,
not a quality breakthrough. Do not merge or deploy.

## Reproducible evidence

- Draft PR: #313; commit `c5281827efa4c25116ea9964f6031c92f1d687c9`.
- GitHub Actions run: https://github.com/Kamara229/JessicaCore/actions/runs/42018971696
  job `127216597913`, **completed successfully** on 2026-11-05 11:42 UTC;
  all 5 focused unit tests passed. Green CI certifies the protocol executed,
  **not** that the model regained quality.
- Complete raw artifact `13410619756`:
  https://github.com/Kamara229/JessicaCore/actions/runs/42018971696/artifacts/13410619756
  (30-day retention; expected expiry 2026-12-05).
  Contains the final `.cb59`, four progress checkpoints, exact questions and
  288 real answers, candidate seal, per-step TRAIN curve, matched speed,
  source/native accuracy, and complete JSON report.
- Source SHA256:
  `50eac24ba7d7a452c495f8adced2cdef3f9cc44e024217fc21d8f2cebb87413c`.
- Final candidate SHA256:
  `2feebfdb09733e676340b666745be1ce705c3b3045588f0954b606d0e6a9b348`.

## Measured outcome

| Measurement | Original FP32 | Original packed CB59 | Joint-trained CB176 |
| --- | ---: | ---: | ---: |
| Historical development prompts | 96 | 96 | 96 |
| Machine-confirmed correct | 36 | 0 | **0** |
| EOS within 64 new tokens | 94 | 0 | **0** |
| Cap hit without EOS | 2 | 96 | **96** |
| Unique token IDs across all generated responses | 187 / 1002 tokens | 9 / 6144 | **1 / 6144** |
| All-64-token constant repetitions | 0 | 18 | **96** |

All 6,144 CB176 generated tokens were token ID `442` (decoded as repeated
` N`); responses did not depend meaningfully on the prompt. The original
packed CB59 also failed all questions, but had 9 generated token IDs. These
are not narrowly wrong answers: the candidate's output distribution
**collapsed to one repeated token**. FP32's 36/96 is the deliberately
conservative frozen machine metric; the previous v174/v175 posthoc semantic
review of the same 96 FP32 texts found 46 clearly correct, with additional
ambiguous cases. Do not equate 36 machine-confirmed with true FP32 ability.

The complete archive remained **11,229,303 bytes =
0.6678394131307638 bpp**. All 211 centroid tables and row scales changed,
as did 61 already-paid raw vectors. The physical 5-bit assignments and
format were unchanged; no Green Tunnel/adapters/sidecar were added.
TRAIN included 128 updates of 293,824 parameters on 64 samples (two passes).
The mean logged per-step TRAIN objective fell from 2.5244 on pass 1 to
1.7161 on pass 2 when matching the 64 samples; this is an in-sample
training observation, **not** a held-out QA improvement. The hidden-error
component fell 1.0501 -> 0.9555; the hard-margin component 7.3715 ->
3.8027, also on the same samples.

The 8 matched 32-token native speed pairs give a median
**CB176/CB59 token-throughput ratio 1.00312244 (100.31%)** on this CI CPU;
not a universal hardware guarantee. The saved native model agreed with the
trained dense surrogate to relative logit error
**3.0274e-7** on the audited TRAIN probe. Native C versus decoded source
errors were around 2e-7 to 4e-7 on four sample matrices. Serializing and
executing the model is therefore not the observed cause of the 96 failures.

## What the experiment establishes, and does not

- **Measured:** the chosen optimization objective can decrease on TRAIN
  while visible answer quality remains at zero and autoregressive decoding
  becomes a constant-token loop. Therefore the current training signal is
  unsuitable as a quality gate.
- **Measured:** retraining only the existing continuous K32x8 fields and
  norms, with indices fixed under this tiny 128-update protocol, failed.
- **Not established:** that joint training with a different allocation,
  larger training set, moving assignments, new source geometry or different
  quantization is impossible at <=0.7 bpp.
- **Hypothesis, not measurement:** the fixed shared 32x8 codebook geometry
  and uniform 5-bit-per-group assignments may insufficiently preserve
  information in critical embedding/head and block operators. The
  constant-token failure does not identify a unique causal layer.

## Research decision

Reject CB176 as an answer-quality candidate. Do not promote the lower
TRAIN objective or successful CI as progress toward FP32-level responses.
Do not use KL/NLL/PPL as release criteria.

A meaningful next experiment must change the *physical representation*, not
repeat the same fixed-codebook training or a K32/V8 index refit. Freeze a
new budget allocator in advance, explicitly account for every byte including
special high-precision regions, and run inexpensive predeclared
first-token/early-EOS and per-family ablations **before** 96-response
generation. Compare real full responses to a semantically reviewed FP32
reference, with >=90% native speed and <=0.7 bpp. Keep independent TEST
sealed until candidate selection; historical questions are development only.

## Isolation

No main merge, LIVE Jessica, Render, Supabase, APK or Green Tunnel changes.
The v174/v175 stage report remains the historical source for those runs:
`research/calibri/results/v174_v175_20261105/README.md` (commit
`fe1df13b502d6c2d652dfb65d1a63cf4ae7c266c`).
