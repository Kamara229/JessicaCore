# CALIBRI session 16 — can weights be natively multidimensional?

Research status: isolated draft PR, no merge to main, no changes to Jessica,
Render, Supabase, APK or the phone. The experiment is **not QMoE reproduction**.

## Fundamental observation

A 4D reshape alone does not save bytes. A compressed **TT-matrix**
representation can save bytes by using two small core tensors, but only
to the degree that the trained layer tolerates an imposed low tensor rank.

For the real Pythia-14m projection 128×512, map the output indices to
(i1=16, i2=8) and input indices to (j1=16, j2=32). Two TT cores are

G1[16,16,r], G2[r,8,32].
W[i1*8+i2, j1*32+j2] ≈ sum_r G1[i1,j1,r] * G2[r,i2,j2].

This is not a mere 4D image. The native forward contraction computes
t[b,r,i2,j1] = sum_j2 x[b,j1,j2] * G2[r,i2,j2] and then
y[b,i1,i2] = sum_{r,j1} t[b,r,i2,j1] * G1[i1,j1,r].
It does **not** construct a dense W matrix for pure TT execution.

Core coefficients are FP16 in the physical container, with 22-byte
header, full physical count and CRC32. Hybrid mode additionally stores
rowwise INT4 quantization of **the residual**, with FP16 scales.
The residual is currently dequantized as a dense buffer during evaluation;
this is not a full compressed fused kernel.

## Predeclared comparisons

Pure TT rank 8/16/32, TT rank 8/16 plus INT4 residual, and ordinary
packed INT4/INT5/INT6/INT8; plus TT rank 8 + INT4 residual with **12
train-only teacher-logit optimization steps**. Freeze all other model
parameters. The residual values remain fixed during student training,
ensuring the physical stream exactly matches the trained forward.

Compare physically accounted bytes, development PPL, and predeclared
independent test PPL on one real pretrained Pythia-14m MLP output
projection. Quality limit: development PPL no more than 1% above
original while bytes ≤ 1.5× INT4 baseline. Select on **development**;
never use test PPL in selection. No unapproved candidate is deployed.

Additionally test mathematical equality of direct two-core TT
contraction against a dense reconstructed matrix, physical roundtrip,
corruption detection, and illustrative CPU-only native TT vs dense FP32
microtiming. Timing is not a claim about Android/ARM performance.

## Prior art guard

Tensor trains for neural nets are established (Novikov et al. 2015).
2026 papers extend TT compression for small language models and
joint tensor compression of MoE expert weights:
- https://arxiv.org/abs/1509.06569
- https://link.springer.com/article/10.1007/s11633-026-1653-x
- https://proceedings.iclr.cc/paper_files/paper/2026/hash/fc8ee7c7ab5b5f6b1615045dfb617ed6-Abstract-Conference.html

Therefore simple TT, TT+quantization and teacher distillation **cannot**
be asserted to be a new invention. CALIBRI's unresolved hypothesis is
a joint rank/topology, quantization, functional healing, and hardware
layout optimizer with demonstrated Pareto gains against strong baselines.
One dense layer cannot establish this.

## Practical go/no-go

Go only if development quality holds at a worthwhile physical size
AND runtime characteristics are competitive. The independent test must
not degrade materially. A failure or a simple INT6 win is a useful
negative, not evidence of success.

Special caveat: WikiText-2 test has been observed in prior CALIBRI
sessions; treat these comparisons as exploratory. Use a separate,
previously unseen task suite for any later confirmatory claims.
