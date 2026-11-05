# CALIBRI v191 — actual post-hoc read-only v190 answer-format audit (5 Nov 2026)

**Scientific status: successful source-answer annotation diagnostic, NOT an independent QA experiment.**

Completed [GitHub Actions #42093390298](https://github.com/Kamara229/JessicaCore/actions/runs/42093390298) on the *unchanged* [v190 128 actual uncompressed original-model answers #13446560270](https://github.com/Kamara229/JessicaCore/actions/runs/42092017343/artifacts/13446560270). [Complete original+supplemental raw audit artifact #13444789944](https://github.com/Kamara229/JessicaCore/actions/runs/42093390298/artifacts/13444789944) preserves all 128 answers, model provenance, precommitted v190 scores, original exact text and each new specific structural parse rationale.

Initial CI #42093137307 FAILED due missing torch/numpy in a read-only audit environment. Corrected this by designing the v191 evaluator to use ONLY the sealed v190 JSON source: no model, tokenization, training libraries or additional original source inference. Final CI #42093390298 **SUCCESS** with adversarial tests proving an incidental correct answer mention or an incorrect arithmetic calculation is not silently scored correct.

## Why the precommitted v190 scorer under-counted

The **precommitted** original v190 strict grader accepts only plain one-word/integer Yes/No/color/fact answers. It legitimately scored an original source completion `200 + 37 = 237` as UNGRADED even though it unambiguously contains the mathematically correct asserted equation. The original 135M and 360M often reply with full owner-qualified text, e.g. `NeriDalo0's folder is purple.` rather than `purple`. It does not follow that every such response is correct: a very similar `217 + 40 = 267` is demonstrably wrong (answer 257) despite a valid grammatical equation.

v191 retrospectively recognizes only an explicit single full utterance where the equation's given operands match the original question, or the statement's grammatical subject matches the **specified fictional owner**, or one of the complete bounded single-claim factual statements. Negated/multivalued/extra statements and arbitrary gold-word mentions remain UNGRADED. Existing v190 precommitted grades are never overwritten.

## Full measured tables, historical public 64 questions per model

| Criterion | Original 135M FP32 | Original 360M FP32 |
| --- | ---: | ---: |
| **Original precommitted v190 confirmed** | **9/64** | **18/64** |
| Original v190 machine_wrong | 8/64 | 8/64 |
| Original v190 UNGRADED | 47/64 | 38/64 |
| **Exploratory POST-HOC v191 grammar-recognized correct** | **34/64** | **43/64** |
| Exploratory POST-HOC recognizably wrong | 26/64 | 19/64 |
| Exploratory POST-HOC UNGRADED | 4/64 | 2/64 |
| Prior ungraded responses reclassified by narrow patterns | 43 | 36 |

### Retrospective per-domain exploratory correctness

| Domain (16 original questions each) | Original 135M FP32 | Original 360M FP32 |
| --- | ---: | ---: |
| Three-digit addition/subtraction | **6/16** | **14/16** |
| Opposite greater-than Yes/No | **8/16** | **8/16** |
| Reversed fictional folder owner binding | **9/16** | **7/16** |
| Short factual questions | **11/16** | **14/16** |
| **Total** | **34/64** | **43/64** |

**Negative and important:** Both original models answer **Yes to all sixteen** of the greater-than comparisons. Since the opposite pair labels alternate Yes/No, each original answers only 8/16 and **zero of eight** contradictory pairs correctly in both directions. A larger original 360M is not an automatic universal teacher improvement: on reversed fictional folder binding this exploratory sample has **360M worse than 135M** (7/16 versus 9/16). The observed arithmetic improvement (14/16 vs 6/16) is considerably stronger.

A red flag in the v190 precommitted strict metric is now diagnosed: **85/128 original answers were initially ungraded**, primarily because short valid full assertions were not bare answer strings. v191 reclassified **79/85** (43+36) into recognized correct or recognized wrong according to narrowly anchored syntactic patterns. This is *format recognition*, **not new model training or compression**.

## Honest interpretation / next gate

This v191 evidence is **POST-HOC**: the grammar rules were implemented after inspecting raw v190 responses. These newly recomputed counts are exploratory and MUST NOT be labeled precommitted, truly independent source accuracy or QA parity for any compressed CALIBRI model. A new independent public question set with the revised grader **frozen before answers** is required to confirm them, especially to distinguish domain-level capability vs prompt familiarity. Prefer answer-level semantics and opposite-pair correctness to first-token matches.

Both source models here are **original FP32**, not compressed. No physical 360M file, actual 360M bpp or compressed-native speed has been measured or can be claimed. Existing genuine v186 physical **0.667839 bpp** model is derived from the **different 135M original**, lacks competent broad answers, and cannot be extrapolated to a 360M score. For a hypothetical complete 360M representation, the <=0.700 bpp ceiling remains **31,659,348 bytes for 361,821,120 original unique parameters**; source bytes, accurate generation and >=90% baseline native CPU would all still need to be verified.

All activity confined to isolated GitHub research draft branch; no main, LIVE Jessica, Render, Supabase, APK, Green Tunnel or sealed v89 private TEST modified or read. Jessica never ran any model test.
