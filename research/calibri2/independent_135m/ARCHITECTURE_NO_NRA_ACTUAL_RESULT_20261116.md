# Measured technical outcome — pretrained FP32 135M → CALIBRI2-like architecture (NO NRA)

CI **#43946485860 SUCCESS** (2026-11-16); artifact **#14265473081**. This is a physically audited research prototype, **not yet a trained Green Tunnel or FP32-quality model**.

## Proven architecture geometry

- Original FP32 architecture: `HuggingFaceTB/SmolLM2-135M-Instruct`, revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`; **134,515,008** unique original trained parameters; native **30 decoder layers × width 576** preserved.
- CALIBRI-style real hooks: S87-inspired trainable low-rank internal GT at layers **7 and 22**, S126-inspired recurrent shared low-rank GT through **all 30 layers** with sequential-coverage assertion; two-slot source-span parallel shared-prefill head; compact trained-S1 delta profile bank and prompt-only neural route. **No NRA or S3 linguistic adapters**.
- S87/S126 GT residuals **zero-initialized**, S95 route and S159 slot head **untrained and disabled for answer override**. These *structures physically exist and execute*, but are **not learned functional replacements**. The generated C0 output MUST equal unmodified direct compressed B.
- All-in physical source **11,545,390 bytes** (11,229,303 original core + 132,280 compact S1 profile delta + 182,498 GT/router/parallel FP16 sidecar + 1,309 manifest). **0.6866380292673365 bpp**, cap 0.700, all parameter payloads counted. 90,341 new trainable auxiliary parameters. No alternative full S1 11MB archive hidden inside the measured package.
- Training source S1 physically carries 128 changed 5-bit code groups and altered centroid tensors for all 211 source matrices. Original S0 source unchanged; zero-init C0 produced exactly B outputs, restored profile0 after S1 also produced exactly B outputs.
- 3-way parallel candidates were **genuinely calculated in batched forward calls**, but no positive consensus or correct generated answer was achieved. The source-bound S159-inspired head's proposals were explicitly shadow-only.

## Actual 12 new benchmark prompts and matched generated answers

| Variant | Literal-only strict exact answers (primary frozen parser) | EOS seen | Same full token sequence as original FP32 |
|---|---:|---:|---:|
| A: original pretrained FP32 | **0/12** | 9/12 | 12/12 by definition |
| B: direct pretrained FP32→CB59 source | **0/12** | 0/12 | 0/12 |
| C0: new CALIBRI2 architecture NO NRA, GT residuals untrained/identity | **0/12** | 0/12 | 0/12 |
| C1: C0 with S1 four-cycle centroids/codes, shadow profile | **0/12** | 0/12 | 0/12 |

**Critical scoring caveat:** The frozen strict parser incorrectly treats full-sentence FP32 responses as incorrect because it demanded only the literal answer word. This is a *format* rather than factual benchmark. Real raw FP32 output demonstrates meaningful correct answers, unlike all compressed versions which emitted repetitive fragments (`speci` or `uesday`). Do not interpret the primary **FP32 0/12** as no language ability.

**Post-hoc human-readable review of the frozen unedited outputs** (NOT the precommitted primary metric and not for checkpoint selection): The FP32 model substantially answered items 0,1,2,4,10,11 correctly (= **6/12** directly identifiable factual answers); wrong/incomplete items: 3 (December instead of October), 5 (6 instead of 9), 6 (incomplete), 7 (12 instead of 9), 8 (incomplete), 9 (gone instead of went). The three compressed/architecture variants answered **0/12** meaningfully and lacked EOS. This conservative review is diagnostic; no statistical noninferiority claim from 12 manually authored questions.

Example original FP32: `The capital city of Japan is Tokyo.` vs compressed C0: `compon speci speci...`. Example `6 + 7 = 13` in FP32 vs repetitive source token fragments in C0.

## What the experiment establishes and what it cannot

- **Established:** a full physically payable pretrained-135M CALIBRI2-style architecture transplant with *actual* S87, all-layer S126, S94/95 compact profile/router, S159 shadow slots, 3-way real branch generation, original pretrained FP32 reference, exact frozen S0 B/C0 functional parity, and no NRA.
- **Not established:** that the untrained GT/slot/router provides quality recovery; true learned physical GT; FP32 parity; multi-domain statistical significance; native C >=90% throughput; CALIBRI 43M learned weights directly usable in original 135M.
- **Recommended next research step:** freeze *new* blind EVAL, then TRAIN-only joint training of selected original paid 211-codebooks + S87 GT + 30-layer S126 tunnel plus protected replay and rollback, train S95 route and S159 source-bound slots on eligible TRAIN examples only, physically repack within <=0.7 bpp, and compare real full/EOS answers to the pinned FP32 on new untouched EVAL. Avoid reusing these 12 disclosed items or prior v231/v232 as checkpoint selection.
- Do not merge into main or alter LIVE Jessica, Render, Supabase, APK. Ordinary Jessica chat tests are not involved.
