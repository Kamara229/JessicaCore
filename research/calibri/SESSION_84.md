# CALIBRI v84 — native operator rewriting rather than only retrofitting pretrained weights

**Proposed 3 November 2026. Research-only, draft PR stacked on v83; not Jessica.**

## Hypothesis
Instead of blindly compressing original trained FP32 matrices then attaching an output-only residual, re-synthesize the **internal compressed operator itself** using ORIGINAL pretrained teacher activations on TRAIN samples. This prototype keeps fixed 5-bit index codes and per-row scales, **relearns the real operator's 32×8 floating codebook values** and snaps them into the actual physical FP16 source. That is an operator-level parameter redesign and real source rewrite, **not yet a new layer topology, nonlinear gating system, or fresh pretrained model**.

This distinguishes three possible future stages:
1. **Operator data rewiring at equal bits** (this v84): regenerate the codebook of each high-error `mlp.down_proj` from actual original FP32 teacher `x -> Wx` behavior on genuine TRAIN. No sidecar, no additional bits, no output-only correction.
2. **Causal validated adaptive operators** (next): decide which layers deserve more representational capacity using TEST-unseen TRAIN/VALID, rather than trusting v83 error energy. Reassign finite bits and potentially redesign mappings/routing while maintaining real <=0.700 bpw.
3. **Native-from-training operators** (much larger project): train transformer architecture with its own quantization-aware low-bit operators from the start (BitNet-like approach), using enough pretraining data and real compute. This is a new training methodology, not a free lossless transformation of an existing pretrained weight source.

## Connection to literature, avoid novelty overclaims
- AdaRound (Nagel et al., ICML 2020): learns data-aware quantization rounding of pretrained weights. https://proceedings.mlr.press/v119/nagel20a.html
- LoftQ (Li et al., ICLR 2024): joint quantization and low-rank initialization; compressed pretrained models. https://proceedings.iclr.cc/paper_files/paper/2024/hash/39ec972afab01e0d8ddc6834a9d12ac1-Abstract-Conference.html
- LoQT (Loeschcke et al., NeurIPS 2024): low-rank quantized weight training and merging. https://papers.neurips.cc/paper_files/paper/2024/file/d0c6bc641a56bebee9d985b937307367-Paper-Conference.pdf
- BitNet (Wang et al., JMLR 2025): training from scratch with low-bit operators; not applicable through simple PTQ alone. https://www.jmlr.org/papers/v26/24-2050.html

Original engineering claim is **not** that codebook adjustment itself has never been published. CALIBRI's hypothesis is that physical <=0.700 bpw native operator rewriting can improve a globally compressed network enough to justify future more adaptive and functionally differentiated learned operators. Require measured evidence.

## v83 genuine evidence informing targets
Real original pretrained SmolLM2-135M-Instruct v83 CI #41575653490: 30 transformer layers × 7 physical C operators = 210. On 4×64 original TRAIN tokens (not TEST), local same-original-activation squared error ranked: layer11 `mlp.down_proj` **58.70%**, layer28 `mlp.down_proj` **34.45%**, layer29 `mlp.down_proj` **4.97%** of aggregate local error energy. These shares are non-normalized energy; alone they cannot predict task importance.

## Nested facade architecture, one responsibility per file

```text
operator_rewrite_v84.py          thin CLI
└── v84/                        top-level package facade
    ├── __init__.py             exports run
    ├── pipeline.py             experiment orchestration only
    ├── sources.py              pinned original model and TRAIN/VALID activations
    ├── fit.py                  differentiable 32x8 trainable physical centroids
    ├── archive.py              exact physical source rewrite and CRC/SHA
    └── valid.py                true native-C whole-model VALID KL/NLL and guard
test_v84_operator_rewrite.py     pure regression tests + actual small native C
```

The legacy v76/v78/v81 and v83 source are deliberately not modified; future large modules will be grouped by next-level facades. This stage is for *codebook recompilation of existing operators*, not replacing every operator implementation yet.

## Exactly what is trained and evaluated
- Original pinned pretrained `HuggingFaceTB/SmolLM2-135M-Instruct` and real original CB59 trained source from CI #41507201766.
- Complete original CB59 archive **11,229,303 B = 0.667839413 bpw** versus original 134,515,008 unique trained parameters. Zero increase in bytes allowed.
- Original WikiText `wiki_fit` and AG News `news_fit`: first **two 64-token TRAIN windows each**, total **256 TRAIN tokens**. Record source revision and SHA. Capture each of 3 target operator INPUT activations, paired FP32 teacher outputs on those exact positions.
- Original `wiki_valid` and `news_valid`: first **one 64-token window each**, total **128 independent TRAIN VALID tokens**, not unseen TEST.
- For each target operator, keep exactly the original 5-bit group index stream, same exact per-row FP16 scales, same structural source manifest and all other original tensors. Optimize its 256 physical codebook scalars in FP32 using streaming minibatch MSE and anchor regularization, checkpoint at steps 12/36/72, snap every candidate back to physical FP16. Select checkpoint by local TRAIN-VALID teacher-conditioned relative error only; keep unmodified option.
- Before any optimization, verify original differentiable source construction matches the actual original native C packed kernel output on identical inputs.
- At physical serialization stage patch only the 32x8×FP16 codebook segments in original CB59 byte stream. Regenerate source CRC32 and SHA256, reparse with CB59, assert **unchanged full physical byte count** and identical all 5-bit indexes.
- Independently evaluate native compiled C full model on original TRAIN-VALID with baseline, each of 3 single rewritten operators, and all 3 simultaneously. Compute true FP32 teacher→compressed full-vocabulary KL and original-token NLL for each VALID corpus, no teacher forced local output pretending to be global competence.
- Scientific selection guard: **both VALID corpora must show KL no worse than baseline and global mean KL must improve by >=0.0001 nats**. If no variant passes, retain baseline as conceptual winner, record physical experimental rewrite artifact **without a go claim**. Even if passed, only source-level *TRAIN-VALID proof of concept*; no independent unseen TEST, not production.
- No source TEST tuning, no editing Jessica/main/Render/Supabase/APK, no claim of full-native optimized inference speed, no proof of improved general QA or reasoning. Keep alternate v81 cross-host nonreproducible issue open.

## Acceptance / falsification
1. Pure tests pass; physical candidate full source bytes identical; rewritten codebook FP16 roundtrip and CRC/SHA valid.
2. 3 actual ORIGINAL pretrained target operators and valid full-model native runtime physically run, with local and global error diagnostics.
3. Publish negative or positive actual TRAIN VALID before designing independent new TEST (new corpus and QA indexes, not reused).
4. If trainable codebooks fail to improve both VALID KL, move to **operator architecture** research (cross-layer scale-equilibrating transformations / input-dependent routing with fully counted physical parameters). No hidden gains from choosing checkpoints on previous TEST.
