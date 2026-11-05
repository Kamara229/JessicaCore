# CALIBRI v177 — completed physical mixed-bit A/B experiment (5 Nov 2026)

**Verdict: FAIL on answer quality; PASS on physical size, native correctness
and 90%-of-CB59 speed threshold on the measured CPU.**
Neither physical source qualifies for answer-quality promotion.

## Verified completed first full run

- GitHub Actions run [42024514856](https://github.com/Kamara229/JessicaCore/actions/runs/42024514856),
  job `127235246352`, **success**; 4/4 unit tests; 96 real
  generations in total (24 per FP32, native CB59, native ADAPT_A,
  native ADAPT_B), exactly as the **early-stop** rule declared before QA.
- Reproducible artifact [13413447458](https://github.com/Kamara229/JessicaCore/actions/runs/42024514856/artifacts/13413447458)
  (archive expires 2026-12-05): both actual complete sources, all
  response texts/IDs, full protocol, native C vs decoded matrix checks,
  first-token triage, matched CPU speed, and per-tensor source diagnostics.
- Source ref for this run: branch head `5120e73f745af3bca6be354b2063e23d7ac30e0d`.
  Later commits changed the **development-question triage IDs and protocol
  metadata only**; do not conflate run IDs or claim they executed here.

| Actual measurement | FP32 | Original CB59 | ADAPT_A | ADAPT_B |
| --- | ---: | ---: | ---: | ---: |
| Physical complete model bytes | Original FP32 reference | 11,229,303 | **11,311,173** | **11,264,937** |
| Complete bits / 134,515,008 unique params | FP32 | 0.667839 | **0.672708** | **0.669959** |
| Native CPU speed, relative to CB59 | -- | 100% | **92.12%** | **92.06%** |
| First-token matches to FP32, 20 dev prompts | 20 | 0 | **0** | **0** |
| Distinct first token IDs in 20 dev prompts | 12 | 3 | **2** | **1** |
| Machine-confirmed correct, 24 actual answers | 5 | 0 | **0** | **0** |
| Outputs hitting 32-token cap without EOS | 4 | 24 | **24** | **24** |

A complete CB177 archive with new source fields, actual variable-width
16/32/64-entry dictionaries, metadata, tied-source aliases, physical CRC,
SHA256, row scales and 1D FP32 vectors was saved for *both* arms.

- A source SHA256: `90c7a4cb62ba737daaa426df275cf6a8acc88648878400d54e7470a2472539d1`.
- B source SHA256: `3f148fcd8c76801ba83fdb79f6886030f4353b13f6abe9cd9458937a49e2d411`.

The complete measured sizes do satisfy <=0.700 bpp; native 32-token
fixed-work pairwise median speed ratios are ~0.92 on this CI CPU.
No hidden inference adapter or dense source weight contributes to these
figures. The decoder matches exactly decoded compressed source with
~2e-7 to 5e-7 relative matrix error for selected embedding, attention
and MLP matrices. Numerical fidelity of the packed implementation is
not the problem.

**Answer behavior is still unusable.** Original CB59 repeats “speci”;
ADAPT_A repeats “enormously” and “stream”; ADAPT_B repeats “sust” and
“delect”. Exact input prompts vary, but the model almost always emits
unrelated repeating tokens, with no EOS at token 32 and no confirmed
answers. First-token agreement with actual FP32 is zero for both arms.
Do **not** present increased output diversity as restored reasoning.

A run with 24 gated questions is **not** a full 96-question failure
count: record it precisely as 0/24, not 0/96. FP32 5/24 is an
imperfect frozen machine-extraction value, not semantic ground truth.
The same historical prompts have previously undergone assistant semantic
review; they are not independent blind TEST.

## Scientific conclusion and research boundary

Matrix-wise selection of 4/5/6-bit K16/K32/K64 V8 source,
even with extra precision on the tied head and alternate
attention/MLP-down allocation, **did not recover behavioral quality**.
This narrows, but does not exclude, other source geometries or
methods that jointly shape operators around their real activations.

**Do not** repeat small reallocations of K16/K32/K64 alone.
Next investigation should measure where correct original token decisions
are lost, against independent PUBLIC TRAIN prompts, and test a materially
different coding representation (e.g., structured transform plus
explicitly paid correction) *before* costly full QA. Do not treat
matrix RMSE, KL/NLL/PPL or first-token diversity as release gates.
The criterion remains real coherent answers, physical <=0.7 bpp,
measured CPU ratio >=0.9, and no regressions versus a correctly reviewed
original FP32 reference.

This is isolated research; do not merge to main, change LIVE Jessica,
Render, Supabase, APK, Green Tunnel, or open private v89 TEST.
