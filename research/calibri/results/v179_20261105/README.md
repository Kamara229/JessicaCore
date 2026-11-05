# CALIBRI v179 — completed original 5-bit index entropy audit, 5 Nov 2026

**Result: Generic lossless compression of existing CB59 K32/V8 code IDs
does not free enough physical bits to solve the answer-quality problem.**
This is an information/coding feasibility result, NOT a new model or
answer-quality improvement.

## Reproducible actual-source measurement

- Completed targeted GitHub Actions
  [run 42028506816](https://github.com/Kamara229/JessicaCore/actions/runs/42028506816),
  job `127248658033`: **3/3 tests passed** and exact original
  CB59 source entropy/coder benchmark completed successfully.
- Original physical source SHA256
  `50eac24ba7d7a452c495f8adced2cdef3f9cc44e024217fc21d8f2cebb87413c`,
  fully validated through CB161 wrapper.
- Raw per-matrix results, code-index SHA256s and provenance:
  [artifact 13414917316](https://github.com/Kamara229/JessicaCore/actions/runs/42028506816/artifacts/13414917316)
  (expires 2026-12-05).
- All **211** original 2D trained source matrices scanned, all
  **16,809,984** group indices, **10,506,240** physical original
  five-bit packed index bytes; complete CB59 physical source
  **11,229,303 bytes**, source parameter denominator **134,515,008**.

## Actual marginal entropy

| Group | Count matrices | Mean code entropy (bits / code) |
| --- | ---: | ---: |
| Tied embedding/output head | 1 | **4.944282** |
| Attention q/k/v/o | 120 | **4.970602** |
| MLP gate/up | 60 | **4.987960** |
| MLP down | 30 | **4.985566** |
| **All per-matrix group-weighted** | **211** | **4.974866** |

Maximum for each 5-bit code is **5 bits**. The 32 symbol IDs all occur;
a single global pooled histogram yields **4.997848 bits**, even closer
to uniform. Conditional on the preceding index **within the same
matrix row**, the group-weighted conditional entropy is **4.914853
bits/code**. This is weaker local predictability than needed for large
lossless savings; it is not a formal global information lower bound on
other representations of pretrained weights.

## Real bit-exact packed-byte compression

Original concatenated packed five-bit streams (matrix-sorted, same
actual index bytes, no teacher state, exact decompression verified):

| Coder | Compressed index bytes | Saved index bytes | Saved index fraction |
| --- | ---: | ---: | ---: |
| zlib level 9 | **10,433,685** | **72,555** | **0.69%** |
| LZMA preset 0 | **10,363,024** | **143,216** | **1.36%** |

Largest demonstrated literal savings are **143,216 bytes**, or about
**0.00852 bits per original model parameter**. These are savings on the
index stream **before paying** for a new decoder, any block boundaries,
random-access capability and CPU decoding overhead. No new native
compressed model exists. It would be misleading to claim the entire
CB59 checkpoint is already 11,086,087 bytes: this is only a
hypothetical raw-byte subtraction, **not an implemented archive**.

Per-matrix independent ideal Shannon entropy suggests around 52.8 kB
of index stream savings before coding overhead; it is less than the
measured whole-stream LZMA benefit because LZMA can also exploit some
short-range cross-matrix/byte structure. It is not a contradiction.

## Scientific decision / next source family

1. **Reject generic lossless recoding of current K32/V8 indices as
   the main quality-improvement strategy.** It cannot supply the
   several bits per sensitive weight that the corrupted source operators
   appear to require, and random-access native execution may become
   more costly. Keep these exact measurements.
2. This does **not** rule out coding *different underlying variables*:
   cross-block predictive weights, structured tensor sharing,
   jointly-trained low-rank-plus-sparse paid corrections, or
   architecture built to learn within sub-bit storage. Those require
   testing real source structure, not assuming code-index redundancy.
3. Future go/no-go stages should predeclare physical byte budgets and
   measure correct original-token decisions plus full answer text,
   not just local reconstruction error, entropy, KL, NLL or PPL.
4. The original compressed model remains unusable for answer quality;
   this entropy experiment did not modify any model or restore answers.

All experiments confined to research. No modifications to
main/LIVE Jessica/Render/Supabase/APK/Green Tunnel and private v89
TEST remains unopened in this research line.
