# v128: critical retrospective audit — previous token rank was sometimes a space

v127a actual 135M run #41838848382 reported spuriously optimistic gold-first-token top1 on three arithmetic prompts. The model outputs " 22", while correct gold "42"/"37"/"56" were encoded as bare blank token ID 216 followed by numeric content tokens. Therefore a rank of 1 for token 216 does NOT signify the correct answer is available. The old v120 code had the same underlying mistake: among first candidates gold and " "+gold it selected minimum rank, potentially selecting nonsemantic bare whitespace for some tasks. Old v120 topK 24-question findings must be qualified, recomputed and never used to claim intelligence.

v128 read-only audit against THE EXACT SAME v120 diagnostic 24 factual examples, test original full native CB59 and paid correction v119 using the complete original packed 135M pretrained archive (0.697353 core-only bpp with one shared Green excluded). For each task compute (A) old wrong "first token" potentially whitespace; (B) first NON-whitespace lexical subtoken rank from original unprompted context; (C) first actual lexical subtoken rank conditioned on forced leading whitespace prefix — NOT autonomous; (D) actual full autoregressive answer quality + EOS. Never give expected answer to generator, only use it for token diagnostic and external correctness check; no new training or altered weights.

The old v120 metric error is not merely a rounding or cosmetic issue; the distinction between blank token and lexical content materially changes any claim that an 8MB Green can recover the correct first word. If core content tokens are not accessible, a tiny neural gate/critic cannot restore lost knowledge without different operator allocations or outside verified facts. Formal independent v89 TEST unopened. Production Jessica/Render/Supabase/APK/main unchanged.

## Completed real audit #41841524270 — SUCCESS, discovered meaningful measurement error

The original full CB59 source remained frozen; v119 complete packed source 11,725,552 bytes = **0.697352789 bpp**. Actual twenty-four v120 VALID QA were regenerated from both original CB59 and v119. Six of the 24 truth labels contain a leading standalone *whitespace* token in the spaced-tokenized reference; the old `gold` or `" "+gold` min-rank objective could count token 216 = bare whitespace, which is not a correct number or word.

| Real full SmolLM2-135M candidate | Historical potentially invalid top32 | **Corrected lexical-content top32 without forced prefix** | Corrected lexical-content top512 without forced prefix | TRUE autonomous correct answers w/ EOS |
|---|---:|---:|---:|---:|
| Original CB59 | 0/24 | **0/24** | **2/24** | 0/24 |
| v119 compressed + trained corrections | 10/24 | **4/24** | **8/24** | **1/24** |

For completeness: new content rank after artificially *forcing* any whitespace prefix gives v119 10/24 top32 and 14/24 top512; those values MUST NOT be called autonomous token accessibility, since the model may only generate a bare blank then repeated '22'. For real free text, v119 only 1/24 answers exact+EOS, and original CB59 0/24. E.g. truth "27": prior v119 old rank 1 (a blank), corrected *unprompted lexical* rank 34485, actual output " 2222222"; truth "37": old rank 1, corrected lexical rank **45775**, actual output " 2222222". The original v120 '10/24 gold top32' overstates the lexical evidence by more than twofold, and must NEVER be reused without corrected context.

This does NOT invalidate physical 0.697352789 bpp or correct native decoding: it invalidates some earlier claims of how readily correct words are retrievable under original logits. It makes hypothesis of solving the language issue with only a text reranker or six learned gate scalars substantially less credible. Prior scientific FAIL remains unchanged and stronger.

Source and full per-prompt token IDs in [GitHub Actions #41841524270](https://github.com/Kamara229/JessicaCore/actions/runs/41841524270), artifact #13329241850 expiring 2026-12-04. The fully frozen independent v89 TEST and Jessica LIVE/Render/Supabase/APK/main were not touched.
