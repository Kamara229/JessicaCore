# v178 — source-only column conditioning, <=0.7 bpp, real native answers

## Motivation

CALIBRI v174, v175, v176, v177 show that modifying 5-bit K32x8 indices,
joint codebook training, or 4/5/6-bit matrix-level reallocation does not
reproduce coherent answers on the historical 96-question development suite.
v177 physically valid A/B measured 0/24 correct with 24/24 EOS cap hits
and 0/20 original FP32 first-token matches per arm. Better source RMSE
alone is not a useful answer-quality release criterion.

## New source representation

For each two-dimensional original FP32 matrix, compute **one shared FP16
positive multiplier per input/output feature column**. Column RMS of the
original weight, after division by paid per-row CB59 scale, is normalized
by its column median and clamped to [0.25, 4.0]. This is precommitted and
not selected by QA or TEST. A K32x8 dictionary and exact 5-bit group
indices are refitted in the whitened column space. Full reconstructed
original weight is:

`W[i,j] ~= row_scale[i] * codebook[index[i,j//8], j%8] * col_scale[j]`.

All original 1D FP32 vectors and tied input/output embedding source are
preserved. The exact scales are now part of the **new physical source**,
not an external adapter. Native C linear multiplies activation columns by
the paid scales *once per eight-feature input group* before LUT; embedding
rescales only requested decoded rows. It never expands complete pretrained
2D weights in deployment.

The complete CB178 archive includes 5-bit indices, 32x8 FP16 codebooks,
FP16 row and **new column scales**, all FP32 1D vectors, alias metadata,
manifest, physical CRC32 and SHA256. Any physical size above 0.700 bits
per 134,515,008 unique original parameters fails; no uncounted sidecar,
teacher, or Green Tunnel. The added column data are paid for from the
remaining source byte budget.

## Precommitted comparisons and controls

- Original pinned FP32 SmolLM2-135M.
- Original pinned native CB59, SHA256
  `50eac24ba7d7a452c495f8adced2cdef3f9cc44e024217fc21d8f2cebb87413c`.
- Native CB178 with source-only column conditioning, candidate SHA-sealed
  **before** any behavioral questions.

All research uses previously defined historical **development** prompts,
not private TEST. In the same spirit as v177, a fixed twenty-question
first-token triage checks >=3 distinct candidate first tokens **and**
>=2 matches to the original FP32 first tokens. If both pass, run
all 96 questions per arm with up to 64 newly generated tokens; otherwise,
run only fixed 24 questions per arm up to 32 tokens. Store full decoded
text, token IDs, EOS and conservative machine grades. First-token
agreement and distinctness are diagnostic triage, not intelligence or
answer quality.

Native 32-token CPU speed on eight warmed matched alternating pairs:
target **>=90% of CB59**; complete packed source **<=0.7 bpp**.
The decisive criterion is actual answer accuracy and coherence
relative to the *actual* FP32 model, after conservative semantic review.
Do not use KL/NLL/PPL as a quality criterion.

Separate tests cover physical bit packing, column factorization and
compiled native C linear and embedding versus literal decoded source.
Actual original FP32 weights are used for source fitting only, and are
not kept in the compressed runtime. The quality suite does not train or
select scales/dictionaries. Results recorded whether positive or negative.

## Isolation

This is research only, a draft PR. Never merge with main or modify
live Jessica, Render, Supabase, APK, Green Tunnel or private v89 TEST.
