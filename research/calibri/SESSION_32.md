# CALIBRI session 32 — physical multidimensional representation, actual quality and speed (isolated)

## Why this experiment

v31 GitHub Actions **41355127113** established a stable compute control on
independent Intel and AMD Linux runners for public pretrained
`google/switch-base-8`: with fixed ATen CPU capability `AVX2` and
oneDNN cap AVX2, the original BF16 Switch forward produced byte-identical
logits and router traces between hosts on two small held-out corpus
samples. Switching selected ATen dispatcher `NO AVX` ↔ `AVX2`
changed actual logits by as much as 9.5 on the 24 fixed vocabulary probes.
This is a **numerical reproducibility control**, not compression proof.

Next CALIBRI objective is **actually serialize multidimensional weights,
measure byte length, language quality, and a physically byte-backed compute
path**. This pilot deliberately returns to a small **different model**
with a well-defined causal-language-model perplexity task:
real pretrained `EleutherAI/pythia-14m` (one trained MLP projection,
128 × 512) and WikiText-2. This is NOT a comparable QMoE compression
ratio on SwitchTransformer experts and NOT a full-model compression.

## Preregistered candidate: tensor cores + train-weighted mixed-bit 2D tiles

1. Reshape one dense MLP projection into a **4D tensor**
   with topology `(m1,m2,n1,n2)=(16,8,16,32)`. Fit a
   **2-core tensor train** by existing TT-SVD, rank **4**,
   store the actual rounded FP16 TT factors physically with CRC.
2. Subtract the **reconstructed FP16-roundtripped tensor core** from
   original weights. Partition the residual into **16 × 32 tiles**
   (a *different* 2D grouping from the 4D tensor-core topology).
3. Write each residual tile as a physical **signed INT3 or INT4 bitstream**
   plus one FP16 scale and one byte specifying bitwidth. Candidate
   quality selector uses **only the real train-split layer input
   second moment** and the positive reduction of weighted squared
   reconstruction error from INT3→INT4.
4. The **hard physical budget equals the measured serialized
   rowwise INT4 control bytes**, including its framing and scales.
   Start with all INT3 tiles; upgrade as many highest-positive
   training error-reduction tiles to INT4 as fit. No validation-set
   selection. Include all TT factors, headers, metadata, CRC, scales
   and packed symbols in counted physical bytes; write an actual
   `.bin` artifact, read it back, verify SHA256/CRC.
5. **WikiText validation PPL** on a real pretrained Pythia-14m with
   exactly one MLP projection replaced by the **DENSE decoded**
   candidate. Compare original source weights and physical INT4
   rowwise control at exactly same model, corpus, CPU dispatch and
   evaluation tokens. 256 train calibration tokens, 1,024
   WikiText validation tokens in non-overlapping 128-token windows.
6. **Actual byte-streamed operator:** separately run a CPU
   **no-full-reconstructed-W** reference using a TT core contraction
   and decoding one residual tile into small scratch at a time,
   on 32 genuine train-activation input vectors. Verify output
   matches a dense matmul using the exact same decoded candidate
   within 1e-4 max absolute. Benchmark its median CPU latency
   versus the same decoded dense W. This is a Python prototype,
   not a fused/native optimized kernel and almost certainly slower.
   Capture negative speedups honestly; decode timing separately.

This combines known TT-SVD and quantization with an activation-aware
**mixed-precision tile allocation under an actual byte cap**. These
individual methods exist; the particular combination is a tested
engineering hypothesis, NOT a proven novel mathematical compression
method, granted patent, or state-of-the-art result.

## Control, reproducibility and failure criteria

Two isolated GitHub-hosted CPUs run the same code independently.
Set `ATEN_CPU_CAPABILITY=avx2`, cap oneDNN to AVX2 before importing
PyTorch 2.5.1+CPU, seed 32, deterministic option ON, two threads.
Record actual model revision, corpus revision, **train and validation
token SHA256**, source matrix SHA256, projection name, physical blob
SHA256, per-block bit allocation, **PPL**, native dense matmul
microbenchmark and Python streaming median latency per runner.

Fail closed on invalid tensor dimensions, impossible budget, nonfinite
weights/evaluation, incorrect bitstream length, corrupted CRC, failure to
restore source parameters, actual packed/no-full-W operator mismatch,
or source SHA mismatch between hosts.

The quality result is an **exploratory WikiText validation subset**:
this project has used WikiText splits in earlier sessions, so do not
claim blind independent holdout or benchmark-wide PPL generalization.
The selected dense model matrix is only one of 12 MLP projections;
no full checkpoint physical compression, no mobile throughput,
no switch MoE QMoE 0.8 bits/weight, and no production deployment.

Only create a separate research branch and draft PR. Do not alter
main, Render, Supabase, APK, working Jessica or secrets; do not
request a chat-side benchmark. Finish with the actual metrics, whether
positive OR negative, when the GitHub Actions evidence is available.


## Measured experimental result — Actions 41356201302 (all three CI jobs PASS)

Two independently hosted CPU jobs successfully loaded REAL trained public
Pythia-14m weights and evaluated identical WikiText-2 token windows, with
the **same source model/matrix/token SHA** across both. CRC-protected
physical binary `calibri_v32_multidim_bytes.bin` was actually written,
read back and checked against reconstructed stream. Its SHA256 was also
**identical across both independent runners**. The subsequent
source/metrics comparator completed successfully. PASS is an artifact /
reproducibility pass, **NOT a compression-quality win**.

Both hosts reported the following **same measured metrics**:

| Actual property | Original FP16 source | Rowwise packed INT4 control | Physical multidimensional TT+adaptive INT3/INT4 |
|---|---:|---:|---:|
| **Physical projection bytes** | 131,072 | **33,042** | **33,006** |
| **Effective bits per projection weight** | 16 | 4.0334 | 4.0291 |
| **WikiText validation PPL, full model with one changed MLP** | **173.38149** | **179.67859** | **219.21966** |

The proposed 4D+tile codec used 128 physical 16×32 residual tiles:
**67 INT3** and **61 INT4**. Its actual byte stream was only **36 bytes**
smaller than the physically packed plain INT4 comparator (0.109%) and
its language-model PPL was **22.0% worse** than INT4 and **26.44% worse**
than the original dense teacher. Therefore at virtually equal storage
budget it is **dominated by the much simpler INT4 baseline in quality**.

The no-full-weight byte-streaming prototype produced the mathematically
same decoded tensor result (**maximum absolute difference 2.31×10⁻⁷**),
but was much slower than native dense restored-matrix multiplication:
- Ubuntu 22.04: byte streaming **4.8982 ms** vs dense **0.03553 ms**,
  about **138× slower**.
- Ubuntu 24.04: byte streaming **4.4137 ms** vs dense **0.03475 ms**,
  about **127× slower**.

These are genuine *per-projection microbenchmarks*, NOT model-wide
inference speed, not ARM speed and not native fused packed throughput.
PPL was computed with **decoded dense weights** in the full model; the
byte-backed prototype was benchmarked independently.

**Conclusion:** scientific reproducibility and binary serialization
succeeded; the proposed rank-4 TT+mixed 3/4-bit layout **failed** a fair
matched-size rate–distortion comparison, and the unfused Python
byte-streamed implementation **failed** a speed comparison by two
orders of magnitude. Do not describe this as a CALIBRI breakthrough
or deploy any part to production.

## Next scoped research hypothesis (not yet proven)

Remove/ablate the costly TT factorization on this projection and use
train-activation-weighted *per-row or fine-grained block allocation*
within a **fixed 33,042-byte cap**, comparing raw INT3+INT4 tiles to
rowwise INT4 and TT+residual. Independently predefine the storage
budget and compare PPL and a compiled/vectorized packed operator in a
separate research session. Do not simply tune ranks against previously
observed WikiText validation or call Python loops a native engine.
