# CALIBRI v83 — hierarchical facade architecture and genuine TRAIN-only distortion atlas

**3 November 2026. Draft, research-only, stacked on v82.** No changes to `main`, Jessica, Render, Supabase, APK or running model. Not a new trained codec nor a claim of restored answer quality. Baseline real original `HuggingFaceTB/SmolLM2-135M-Instruct` at pinned revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`.

## Why this experiment
Completed [v82 run 41559245311](https://github.com/Kamara229/JessicaCore/actions/runs/41559245311) had fully successful **execution**, but **failed scientific go/no-go**. On 80 new SciQ questions teacher 65, CB59 25, v76 20, v81-style correction 16; on 80 BoolQ teacher 42, CB59 42, v76 33, v81-style correction 44. KL regressions for v81-style vs v76 were +0.22503 WikiText and +0.08955 AG News; greedy distinct ratio: teacher 0.7812, CB59 0.1042, v76 0.0729, corrected 0.0833. Source byte length of CB76 **11,725,487** = **0.697348923 bpw**, fully counted. **Cross-host/source-reproduction SHA mismatch**: v82 regenerated `31b1de60f62feb249d3a7c07b13edc2453568f33a028e9cf7be4ca71c6279fd0` versus earlier v81 `94282c492713b053ccabc76510bad5f45791142866aa81abad62b35081e752c0`, although same-host repeated TRAIN produced matching binary. v83 does **not** conceal or repair that independent defect.

Before attempting adaptive bit allocation or another output-only rank-4 correction, localize exactly where quantization errors originate. Crucially distinguish the error from a single packed **operator with identical FP32 teacher inputs** from the cumulative hidden-state drift of the full student.

## Explicit nested FACADE tree

```text
layer_atlas_v83.py                   # CLI facade (only dispatch)
└── v83/__init__.py                  # public facade: run
    └── v83/pipeline.py              # orchestration facade (no numeric kernels)
        ├── v83/measure/__init__.py   # measurement facade
        │   ├── hooks.py             # actual paired trained operator probes
        │   └── metrics.py           # pure streaming FP64 tensor numerics
        └── v83/study/__init__.py     # study facade
            ├── sources.py           # strictly frozen training inputs / SHA
            └── scoring.py           # descriptive ranking; no weight edits
test_v83_facades.py                  # pure regression tests
```

No 20k-line monolith. Each small file owns a single responsibility; further modules may be grouped under another facade if the domain expands. Existing legacy scripts remain untouched in this branch, so future migration can be incremental.

## Actual data and exact measurement

- Download the previously audited full original CB59 source (11,229,303 B, 0.667839413 bpw) from original real pretrained 135M v59 workflow artifact. Verify original pretrained parameter count, exact SHA before and after, and that the compressed runtime retains no dense pretrained 2D source.
- Use exactly two **original TRAIN** text domains `wiki_fit`, `news_fit`, first two consecutive **64-token windows per domain** after the preexisting pinned TRAIN slicing: four *genuine*, replayable and separately SHA-checked inputs. Not TEST, not VALID.
- Build original FP32 pretrained teacher and original CB59 native C packed fully executable model. Use the actual 30-layer network (dynamic layer count, assert student/teacher match).
- For each of 7 `q,k,v,o,gate,up,down` projections in **every** transformer layer, intercept the original teacher's activation `x` and teacher output `W_fp32 x`. On the **same x**, call the original CB59 native C packed operator `W_packed x`, not a float32 re-quantization reconstruction. Accumulate reference/output/error squared energies, pooled relative L2, absolute RMSE, cosine, signed bias over genuine tokens and both TRAIN domains.
- In a **separate** diagnostic, compare outputs of complete teacher and actually packed native student layers on the *same original token window*: these are propagated/cumulative drifts, not the local operator-only error.
- Rank operators by true error energy share and aggregate per layer. **Magnitude does not equal causal importance.** No task-conditioned trained bit allocation, no claims that reassigning bits improves QA, and no use of v82 TEST for selecting new weights.
- Strictly metadata/experiment output; no modifications or new generation source; this is the required quantitative groundwork for v84 mixed-precision and interlayer correction proposals. v84 must use new disjoint heldouts.

## Quality/safety acceptance

Source complete and pinned; source SHA unchanged; every original projection verified present as `PackedLinear`; every teacher hook hit once per TRAIN window; no NaN/Inf; all layers paired; no Test source; no source or TRAIN optimizer writes. Pure regression unit tests pass before slow actual pretrained CPU run. CI may pass even when distortion measurements reveal severe losses — that is an important negative result, not proof of recovery.

## Follow-up after genuine results

Use error atlas to choose **hypotheses** for sensitivity interventions, then test with actual causal perturbations *on TRAIN+VALID only*, include all headers/factors in physical bits/param, and independently validate on a **new**, untouched TEST. To address v82 cross-run mismatch, investigate determinism of teacher activations, dataset revisions, CPU BLAS and checkpoint creation using controlled reproducibility diagnostics before trusting byte-identical reconstructions across hosts.
