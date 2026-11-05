# CALIBRI v193 — NEW 96 public original FP32 135M vs 360M QAs, frozen v192 scorer (5 November 2026)

**Result: both original uncompressed FP32 models were actually evaluated on 96 identical new publicly constructed QA inputs each; the newer 360M shows a stronger arithmetic baseline but also serious repeated-answer defects. Absolutely NO 360M compression was evaluated.**

[GitHub Actions successful run #42095901870](https://github.com/Kamara229/JessicaCore/actions/runs/42095901870) saved [actual 192 full original generated answers, all prompt and tokenizer IDs, code provenance and precommitted golds, artifact #13446083971](https://github.com/Kamara229/JessicaCore/actions/runs/42095901870/artifacts/13446083971).
First attempted CI #42095586864 failed with a **Python SyntaxError before any model was loaded**; fixed the logger's closing parenthesis, then independently reran full experiment successfully. The frozen 96 QA golds and original v192 SHA remained unchanged by the correction.

## Precommitted reproducibility

- **Exactly 96 NEW public developmental questions**, 24 arithmetic, 24 opposing greater-than comparisons (12 pairs), 24 reversed fictional ownership colors (12 pairs), 24 short factual.
- No exact reuse from historical public v169 96 questions or v190 64 questions. The complete gold/prompt input set was physically written before loading either original FP32 model.
- Same 96 question texts for both checkpoints; each checkpoint uses its own native chat tokenizer/template.
- Both original FP32 HuggingFace checkpoints pinned to their original SHA:
  - 135M original `12fd25f77366fa6b3b4b768ec3050bf629380bac`; 134,515,008 unique parameters.
  - 360M original `a10cc1512eabd3dde888204e902eca88bddb4951`; 361,821,120 unique parameters.
- **Exactly SHA-sealed v192 scorer** `e230e480c02c680ae02d17b0825b451cbf76b138ceb743f181cc6fb62ad4ac1f`, frozen in prior independent successful v192 CI #42094376648 **before v193 existed**. No post-answer grader change is allowed for primary scores.
- Greedy 32 emitted-token cap, full actual answer preserved, all correct, wrong and ungraded cases retained. No tuning, compression or access to private v89 TEST.

## Precommitted strict machine scoring (new actual public DEV inputs)

| Grade | 135M original FP32 | 360M original FP32 |
|---|---:|---:|
| **Machine-confirmed correct / 96** | **30** | **46** |
| Recognized machine-wrong / 96 | 37 | 38 |
| **UN​GRADED, not presumed wrong / 96** | **29** | **12** |
| Answer conservation | 30 + 37 + 29 = 96 | 46 + 38 + 12 = 96 |

| Domain | 135M confirmed /24 | 360M confirmed /24 |
|---|---:|---:|
| Three-digit arithmetic incl. multiplication | **6** | **18** |
| Reversed numeric Yes/No | 12 | 12 |
| Reversed fictional folder binding | **12** | 11 |
| Short factual | 0 | 5 |

**Paired test (requires BOTH opposing answers correct)**: reversed numeric Yes/No **0/12 both models**; reversed fictional ownership **2/12 original 135M**, **0/12 original 360M**.

## Some representative verbatim evidence from original FP32 responses

- Arithmetic: input `Compute 356 plus 55. Reply with the result as one number.`, original **135M emitted `356 + 55 = 391` (incorrect)**, original **360M emitted `356 + 55 = 411` (correct)**. More math examples and all raw token IDs archived.
- Reversed numerical comparison pairs: originals answer `Yes` to the greater-than statement AND `Yes` to its deliberately false reversal. The 12/24 marginal correctness score is **misleading**; **zero** verified opposite pairs.
- Binding pair with the same fictional owner: original 135M or 360M may answer the earlier color irrespective of which owner-color binding was reversed, causing zero complete opposites for 360M despite 11/24 marginal correct.
- Factual grader limitation: original 135M emitted `The largest planet in the Solar System is Jupiter.`, `The capital city of Norway is Oslo.`, `The chemical symbol for silver is Ag.` — **correct content but UNGRADED by the frozen generic v192 grammar**. Original 360M emitted `A typical adult spider has 8 legs.`, also UNGRADED. Therefore **0/24 confirmed factual for 135M is NOT evidence 135M knows zero facts**, and the factual per-domain comparison is NOT comparable to human-judged accuracy.
- These original models have genuine wrong factual replies as well: original 135M `A typical adult spider has 10 legs.`, `The smallest unit of a chemical element is a proton.`; original 360M `Mercury` when asked for the largest planet, and `Oxygen` when asked for atmosphere's dominant gas.

## Decision: stop cycling only source checkpoints; conduct a genuine 360M physically paid compressed-model experiment

Repeated QAs from v188 through v193 provide sufficient evidence that 360M original is stronger than 135M mainly in arithmetic, but original source is still weak on paired Yes/No and ownership consistency. **Do not claim a compressed model can perform better than its original just by a parser improvement.** More source-only QA alone will not address CALIBRI's main goal.

Next attempt: create a **new real physical <=0.7 bpp archive from the 360M ORIGINAL 361,821,120 unique parameters** using the existing CB59 compact per-8-weight 5-bit-group, per-row scale and FP16 codebook baseline as an auditable rate-first control, including metadata, tied aliases and SHA/CRC. Evaluate physically packed C native CPU execution, memory and actual answer degradation on pre-frozen public v193 cases; do not allow uncounted FP32 source or teacher sidecar in packed inference. Only if such a compact 360M source is actually created and decoded can it be compared to a better actual compression method, e.g. independently trained 4D / per-layer adaptive quantization, **with measured bytes, QA and native throughput**, not only RMSE. 360M complete 0.700 bpp ceiling = **31,659,348 bytes**; this is a ceiling not yet an achieved result.

No NLL/KL/PPL acceptance. No source-training on any public DEV golds. No main, LIVE Jessica, Render, Supabase, APK, Green Tunnel or private v89 TEST modified or accessed; research draft PR unmerged; Jessica no model tests.
