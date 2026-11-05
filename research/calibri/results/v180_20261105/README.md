# CALIBRI v180 — completed signed orthogonal source experiment (2026-11-05)

**Scientific result: FAIL on actual answer quality.** This is a complete
source+native implementation and a definitive negative result for the fixed
32/64-dimensional Hadamard transformations in this 5-bit K32/V8 format.
The research does not establish impossibility of *learned* transform
BTC-LLM-like representations or trained sub-bit architectures.

## Verified measurements

- Draft PR [#317](https://github.com/Kamara229/JessicaCore/pull/317),
  implementation source commit
  `c40018872fe6a41eded7025bbf59d62644f8c19c`.
- [GitHub Actions run 42062163519](https://github.com/Kamara229/JessicaCore/actions/runs/42062163519),
  job `127366541906`: **SUCCESS**, focused compiled C/numeric/orthogonal
  tests passed, both actual 211-matrix physical sources built, full native
  checks passed, 96 actual generations (24 each FP32, CB59, H32, H64)
  completed.
- Complete physical sources, all actual responses, native C roundtrip
  and byte seals: [artifact 13432333964](https://github.com/Kamara229/JessicaCore/actions/runs/42062163519/artifacts/13432333964),
  expires 2026-12-05.
- Original pinned source CB59 SHA256
  `50eac24ba7d7a452c495f8adced2cdef3f9cc44e024217fc21d8f2cebb87413c`.
- Native source H32 SHA256
  `825453c2a6aa7c2944f192d82f2e445e2998a97b93908534e00b3f4360172584`.
- Native source H64 SHA256
  `07dabbcfa325191f5e65ba97142ea0c181a739f74b92384745348f780759067d`.

| Check | FP32 | CB59 | H32 | H64 |
| --- | ---: | ---: | ---: | ---: |
| Complete bytes | Original FP32 | 11,229,303 | **11,229,439** | **11,229,439** |
| Bits per unique original parameter | 32 | .667839 | **.6678475** | **.6678475** |
| Median native CPU speed vs CB59, eight matched 32-token pairs | -- | 100% | **99.00%** | **98.90%** |
| First-token matches original FP32 on frozen 20 dev prompts | 20 | 0 | **0** | **0** |
| Unique first token IDs on 20 dev prompts | 10 | 3 | **1** | **1** |
| Machine-confirmed correct answers / 24 | **6/24** | **0/24** | **0/24** | **0/24** |
| Did not emit EOS before 32 tokens | 4/24 | 24/24 | **24/24** | **24/24** |

The predeclared gate was not met. Therefore only **24 questions per arm**
were actually generated: DO NOT CLAIM 0/96. FP32 grade 6/24 is
conservative machine extraction on historical DEVELOPMENT questions;
semantic accuracy was assessed separately on previous 96-sample runs.

Native C versus exact reconstructed compressed source relative errors
were ~2.77e-7 to 3.54e-7 on four matrices per arm. The native embedding
lookup correctly applies inverse Hadamard, and embedding/output head use
one identical physically stored tensor. All original trained 2D
operators remain physically packed. Runtime calculations and storage
are **not** the reason actual answers are nonsense.

Observed output text:
- H32 repeats `hehehehe...`, with mostly token ID 255.
- H64 repeats `isationsisations...`, mostly token ID 11562.
- Native original CB59 repeats unrelated `speci`.
No meaningful coherent answer recovery.

Weighted operator reconstruction error across all matrices:

| Family | H32 relative weight RMSE | H64 relative weight RMSE |
| --- | ---: | ---: |
| Tied embedding/output | 0.67896 | 0.68046 |
| 120 attention matrices | 0.72863 | 0.72813 |
| 90 MLP matrices | 0.72715 | 0.72688 |

The tied embedding/output distortions are higher than the v177/v178
original CB59 around 0.6625. These weights have not gained fidelity.
Therefore any small amount of numerical CPU speed improvement is not
a meaningful quality gain.

## Research implication

Do not promote H32 or H64, do not merge or deploy. The original full
source 0.66784 bpp is physically compact and executable, but still loses
too much model information. A fixed basis without learned compensating
operators has not changed this.

A separately isolated **v181 deliberately higher-rate** genuine
pretrained-recovery diagnostic tests whether the SAME full pretrained
model regains answers when groups have two or one original weights
per five-bit code. It intentionally **does not meet <=0.7 bpp** and
must never be described as a final CALIBRI candidate. A positive
higher-rate control would establish an empirical bitrate cliff,
instead of pretending negative low-bit quantization experiments are
quality progress.

Research isolation: no main, LIVE Jessica, Render, Supabase, APK,
Green Tunnel or private v89 TEST changes. Jessica is NOT assigned
tests; all GitHub tasks run directly under assistant control.
