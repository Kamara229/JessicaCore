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
