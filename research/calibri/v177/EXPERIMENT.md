# CALIBRI v177 — adaptive matrix bit allocation, actual native test

## Motivation and scientific limitation

v174 showed severe degradation even when individually compressing source families;
v175 (all-matrix covariance-aware 5-bit index updates) gave 0/96 correct;
v176 (joint optimization of K32x8 dictionaries/row scales/norms at unchanged
11,229,303 bytes) gave **0/96** and **all 6,144 generated tokens identical**.
The saved C decoder agreed with its intended values. Optimizing the same
underpowered representation again is not evidence of quality recovery.

v177 changes actual *physical source geometry*, not a training objective:
four-, five- and six-bit indices are used for individual full matrices,
with native codebooks of 16/32/64 vectors of length eight, all physically
stored. No post-inference dense matrix and no uncharged teacher, sidecar,
new memory, Green Tunnel, or quality-guided fitting. Source bit rates and
dictionary sizes are paid for by the complete one-file archive.

## Frozen allocations (both evaluated, neither cherry-picked)

| Matrix family | A | B |
| --- | ---: | ---: |
| Tied embedding and output head | 6 | 6 |
| Attention q/k/v/o (120 matrices) | 6 | 5 |
| MLP gate + up (60 matrices) | 4 | 4 |
| MLP down (30 matrices) | 5 | 6 |

Original 5-bit matrix segments are copied byte-for-byte. Four- and six-bit
matrices are independently fitted from original frozen FP32 weights with
deterministic kmeans-style 8D clustering and nearest-int32 source indices.
The paired allocation is source-only, not chosen using any QA responses.
The global rate cap is **<= 0.7 bits per 134,515,008 unique parameters**,
counting code indices, all FP16 dictionaries and row scales, FP32 raw vectors,
manifest, alias metadata and CRC+SHA. The 5-bit CB59 complete checkpoint is
the fixed reference for size, execution speed and downstream answers.
Any candidate above the actual byte budget is invalid and cannot be promoted.

## Native execution and controls

A new C lookup-GEMV reads 4/5/6 bits per 8 weights for each matrix with
actual 16/32/64-entry codebooks. Embedding reconstructs only requested rows.
The code makes no full dense 2D matrix allocation at inference. Unit tests
check bit-boundary/end-of-buffer behavior, physical codebook size, toy C
matvecs and requested-row embedding versus literal decoded FP32; full
candidate requires per-family native-vs-exact-decoded checks, tied source
identity and no dense trained matrices. Physical archive is SHA-sealed
before any actual question generation.

The precommitted suite is the 96 historical DEVELOPMENT prompts; this is
not independent TEST and does not claim universal quality. First-token IDs
on 20 fixed development questions are merely a cost triage: run all 96
questions / 64 new tokens per arm if either candidate yields >=3 unique
first tokens and >=2 matches to original FP32. Otherwise generate only 24
frozen balanced development questions / 32 tokens for each of original FP32,
native CB59, native A and native B. Keep full verbatim responses, token IDs,
EOS, conservative machine grading, and ungraded semantic-review caveat.
This triage rule is declared BEFORE running any candidate.

Measure native A / original CB59 and B / original CB59 with eight warmed,
alternating same-prompt / same-32-generated-token CPU pairs. A successful
candidate would require **<=0.7 bpp**, native CPU ratio **>=0.90** and
credible non-collapsed answers approaching **actual FP32 answer ability**,
not just token diversity or first-token overlap. A green GitHub Actions
build is technical success only. No KL/NLL/PPL quality criterion.

## Scope / reproducibility

Commit-pinned SmolLM2 original weights from
`binary_geometry_v64.MODEL_SHA`; immutable original CB59 inside verified
CB161 source from workflow run `41920207168`. All source inputs and frozen
questions are recorded in artifact JSON. Training/test leakage is avoided
for the **source encoder** because it reads only FP32 weight tensors, not
questions or gold answers; development question suite is still previously
seen and cannot replace an independent, sealed final TEST. Target independent
TEST stays unopened until candidate selection.

All work is confined to this research-only draft PR. Do **not** merge into
main, LIVE Jessica, Render, Supabase, APK, or the Green Tunnel.

Outcome will be recorded when GitHub Actions and all artifacts complete.
