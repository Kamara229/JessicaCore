# CALIBRI v179 — measurable feasibility of entropy/predictive coding

**Source-only diagnostic, not a compact model-quality candidate.**

User-suggested signal compression and redundancy ideas must be tested
against **actual source data**, not assumed to save enough bits.
Prior CB59/v174–v178 physical source stores 16,809,984 independent
logical code indices in 211 actual matrices, at five bits per eight
original FP32 weights, plus FP16 dictionaries/row scales and full
FP32 original vectors. Native decoding works; answer quality collapses.

Before investing in Huffman/arithmetic, differential, predictive,
or error-reconstruction coding, this experiment measures whether the
**current 5-bit index stream contains removable entropy**:

1. Reopen the exact pinned original CB59 from validated CB161 source
   SHA256 `50eac24ba7d7a452c495f8adced2cdef3f9cc44e024217fc21d8f2cebb87413c`.
2. Measure exact 32-symbol Shannon entropy per matrix and group
   frequency, plus H(code_t | code_{t-1}) within each original source
   weight row. Keep all original matrices and family breakdowns.
3. Concatenate actual per-matrix packed byte streams in canonical
   sorted source-key order; measure **byte-exact** zlib level 9 and
   LZMA preset 0 compression, and verify exact decompression.
4. Report theoretical per-tensor entropy savings separately from
   real compressed byte savings. Make no false claim of runtime
   speed, native random access or lower complete bpp without
   implementing and paying for the actual decoder and format.
5. No model generation, no QA answer tests, no original FP32 teacher,
   no NLL/KL/PPL or private TEST. This only rules IN or OUT
   entropy coding of the **existing indices**, not all future
   predictive source designs.

If code entropy is nearly 5.0 per index, fixed-index arithmetic/Huffman
coding cannot release a meaningful byte budget. The appropriate next
research is to change the **underlying representational variables**
(cross-layer predictor, structured operator, shared low-rank basis
plus paid residual), not repeatedly compress statistically near-random
K32 assignments.

Research-only PR; never merge/deploy into LIVE Jessica, main, Render,
Supabase, APK or Green Tunnel.
