# CALIBRI v189 — actual 360M vs 135M uncompressed teacher ability (5 Nov 2026)

**Research result: larger ORIGINAL uncompressed FP32 model improves many answers, but not all question types. No compressed 360M source was created.**

The complete actual model comparison passed [GitHub Actions run #42089843075](https://github.com/Kamara229/JessicaCore/actions/runs/42089843075) and archived [all 192 original 360M responses, conservative grades and the question-paired 135M comparison, artifact #13443883690](https://github.com/Kamara229/JessicaCore/actions/runs/42089843075/artifacts/13443883690). 192 old public historical DEVELOPMENT questions (96 questions in both original direct/brief prompt styles), not a blind TEST or novel source-training benchmark. The preserved raw answers include cases the strict grader could not recognize.

## Source provenance and budget

| Provenance | Original FP32 135M | Original FP32 360M |
| --- | ---: | ---: |
| HuggingFace checkpoint | SmolLM2-135M-Instruct (original pinned revision in v188) | HuggingFaceTB/SmolLM2-360M-Instruct |
| 360M resolved checkpoint Git SHA | — | `a10cc1512eabd3dde888204e902eca88bddb4951` |
| Original unique trained tied parameters | 134,515,008 | **361,821,120** |
| Max physical original model file at 0.700 bpp | 11,770,063 bytes | **31,659,348 bytes** |

No CALIBRI 360M compression or native CPU throughput benchmark exists in this run; 0.700 bpp above is a future candidate *limit*, not a measured result.

## Actual frozen public DEV question outcomes

| Machine grading, all 192 historical public prompts | 135M FP32 | 360M FP32 |
| --- | ---: | ---: |
| Machine-confirmed correct | **66** | **105** |
| Parsed, whether correct or incorrect | **135** | **148** |
| Ungraded / ambiguous (not known wrong) | **57** | **44** |
| Parsed but not confirmed correct | 69 | 43 |
| Hit 32-token generation cap without EOS | 9 | **0** |
| Mean emitted token distinctness | 0.91534 | **0.96433** |
| Source-trained on public DEV for this comparison | No | No |

Machine-confirmed per prompt style: direct **35/96 vs 51/96**;
brief **31/96 vs 54/96**. Over 192 prompts there are **39 more
automatically confirmed answers** for 360M than 135M, and **13 fewer
ungraded**. An automatic grade is NOT a human-rated semantic accuracy
measure, and ungraded responses must not be silently treated as errors.

### Source-capability by domain (machine-confirmed / number of prompts)

| Historical question domain | 135M direct | 360M direct | 135M brief | 360M brief |
| --- | ---: | ---: | ---: | ---: |
| Arithmetic | 7/16 | **11/16** | 3/16 | **11/16** |
| Fictional notebook-binding/color | 14/32 | **26/32** | 13/32 | **27/32** |
| Opposite Yes/No | **8/16** | 8/16 | **9/16** | 8/16 |
| Factual recall | 2/16 | 2/16 | 2/16 | 4/16 |
| Ordering/logic | 4/16 | 4/16 | 4/16 | 4/16 |

**Measured important negative result:** the newer 360M original
demonstrably improves arithmetic and fictional color binding under
the grader; **no improvement at all in direct Yes/No or logic**, and
some original 360M answers are wrong, such as a fictional false claim
also receiving `Yes`. Factual recall is parser-limited:
the 360M original generates `Rome` for the capital of Italy, but
this response is marked UNGRADED by the strict extractor. Thus its
machine-confirmed 2/16 factual result is an unsafe proxy for
semantic knowledge.

Two fully machine-correct opposing color-binding questions:
- 135M direct 0/16; 360M direct **10/16**.
- 135M brief 1/16; 360M brief **11/16**.
- Opposite Yes/No both correct: original 360M **0/8** in either
  style, despite answering `Yes` to the true half.
These paired counts test both responses, not marginal response accuracy.
For many correct color-binding pairs the **FIRST token stays identical**,
likely due to common answer introductions, so v188's first-token-change
diagnostic is not a sound generic behavioral correctness gate for
free-form language. Check entire generated answer and conservative
pair grade rather than mistaking token differences for semantics.

Paired confirm-only direct: 19 questions correct in new original
but not old, 3 correct in old but not new; brief: 27 newly confirmed,
4 lost confirmations. These counts still include parser ambiguity.

## Decision for next CALIBRI research

The 360M original source is demonstrably more capable on the
existing historical DEV arithmetic/binding tasks than 135M. The
relative benefit is highly domain-specific and **does NOT establish
general reasoning parity** or meaningfully solve opposite Yes/No.
Before committing to a new 0.7 bpp physical model architecture,
a well-controlled next step is original-source behavioral auditing
on a **new, independently constructed, balanced public development
set** whose question types distinguish valid answer content, and
a corrected answer evaluator that recognizes concise correct
paraphrases (`Rome`) without accepting partial gold words inside
wrong statements. Freeze gold labels, prompt hashes and graders
before model generation; do not use any private v89 TEST.

If this evidence supports 360M as a stronger preservation target,
recalculate 360M physical source from its own 361,821,120 unique
parameters. Under <=0.700 bpp the complete byte ceiling is
31,659,348 bytes. Actual 360M quantization, paid codecs, native
CPU speed and real QA must all be separately measured; there
is NO such PASS at present.

No NLL/KL/PPL acceptance criteria; no LIVE Jessica, Render,
Supabase, APK, working main, Green Tunnel or private TEST changed;
all work in isolated draft research PR.
