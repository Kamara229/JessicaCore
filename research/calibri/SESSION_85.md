# CALIBRI v85 — input-conditioned compressed neural operators, entirely paid, real native C

**Research-only. Draft PR stacked on v84 (#214), no deployed Jessica/main/Render/Supabase/APK changes.**

## Why static operator rewriting is not sufficient

Real prior v84 CI #41577360464 successfully rewrote exactly three physical original CB59 FP16 codebooks at no extra bytes. The direct-local tensor discrepancy reduced (model.layers.11.mlp.down_proj 0.83877→0.73457, layer28 0.99616→0.91962, layer29 0.74785→0.62581), but there was **no valid two-domain full-model language quality gain**. Original packed baseline Wiki/News VALID KL **14.12105 / 14.96862**; v84 static layer11 **14.28695 / 14.59994** (Wiki worse), static layer29 **14.05281 / 15.17067** (News worse), all three **14.26578 / 14.84420** (Wiki worse), so **research_go=false**. Improved local operator approximation is NOT equal to improved global language model.

User hypothesis: to improve CALIBRI we must replace the behavior of error-prone operators, not only compress original old weights.

## What v85 actually changes
For each of 3 actual pretrained high-error MLP down projections, define a new input-conditioned operator:
`y = C_original(x) + sigmoid((x/RMS(x))·g + b) × (C_v84(x) − C_original(x))`.
- Both `C_original` and `C_v84` call **real native C packed 5bit/8D operators**, with identical 5bit codes and per-row FP16 scales but different full physical FP16 codebooks.
- The **gate is token-activation-dependent**, not dataset/oracle-label-dependent. Fit `g,b` on the actual **native packed student's input activations**, targeting the corresponding original pretrained teacher output at that same original token index. This allows attempting to compensate upstream drift too.
- The gate vector `g` and scalar `b` plus the three alternative 32×8 FP16 centroids are **actually stored in new integrity-protected CB85 full source**. Original CB59 source retained in full, source manifest/CRC/SHA counted. No unaccounted sidecar, no original FP32 dense pretrained source resident.
- **This is more functionally adaptive than v84**, but not trained-from-scratch LLM architecture, cannot claim improvement without real independent TEST. Current prototype requires *two* native C calls and one Torch CPU gate per affected operator, **not yet fused/optimized**. It may be slower even if quality improves.

## Facade structure
```text
conditional_native_v85.py         thin CLI facade
└── v85/                          public facade
    ├── __init__.py
    ├── pipeline.py               orchestration only
    ├── physical.py               full CB85 physical source + integrity
    ├── native_ops.py             actual activation-gated original C operators
    ├── sources.py                new strictly TRAIN and new VALID inputs
    ├── gate_train.py             small FP16 trainable adaptive router
    └── validation.py             REAL full model KL/NLL and strict selection
test_v85_conditional_native.py    pure+actual small native C regression tests
```
No code inserted in Jessica runtime, no monolithic replacement of v59/v76/v81 scripts.

## Preregistered train/valid data (disjoint from previous v84 portions)
- Pinned original teacher `HuggingFaceTB/SmolLM2-135M-Instruct`, complete original CB59 downloaded from original v59 run, alternative native codebooks from v84 audited run.
- Original WikiText `wiki_fit` original TRAIN **windows 2 and 3**, AG News `news_fit` original TRAIN **windows 2 and 3**. This is 256 TRAIN tokens. v84 used windows 0 and 1.
- Original WikiText `wiki_valid` **window 1** and original AG News `news_valid` **window 1**. This is 128 original independent TRAIN-split validation tokens. v84 used window 0. These remain tiny samples, not official independent TEST, and previous v76 experiments used portions of these corpora.
- Freeze source actual FP16 gate after **84** predetermined Adam optimization steps using only TRAIN local teacher-output targets and actual original C student inputs. Do not change the gate on any validation or TEST evidence. Keep complete physical CB85 size <=0.700 bpw.
- On first opening the fresh VALID, score original native CB59 baseline, 3 single gated operators, 3 pairs and all 3 simultaneously. Score original teacher→native *full-vocabulary* language KL and real next-token NLL on both VALID corpora. Predeclare promotion guard: both VALID corpus KL <= baseline and mean improves >=0.0001. Negative gate retains baseline as research winner; experimental fully physical CB85 artifact remains unpromoted.
- Evidence is truthful even if v85 research fails. A positive VALID does not establish broader cross-domain assistant competence; only after this design do we preregister entirely unseen TEST.
- Strictly observe original CB59 checksum and original 134,515,008 trained unique parameter count; all additional gate/codebook/header/CRC/SHA bytes included. No original dense 2D trained matrices in the student hot runtime.

## Follow-up if positive or negative
If positive, **separate independent unseen test** of real task scores, Wiki/AG News KL/NLL and greedy distinct token ratio, plus speed comparison and a fused direct C implementation. If negative, investigate cross-layer error cancellation and task-aware training vs local distillation, possible training-from-scratch extremely-low-bit architecture inspired by BitNet; do not declare a new universal compression method on limited data.
