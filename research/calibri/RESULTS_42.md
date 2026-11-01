# CALIBRI v42 — measured result, three first-use corpora (GitHub Actions 41364043512)

**ACTUAL, not predicted.** Research-only PR #172, two independent GitHub-hosted
CPU real-Pythia14m jobs and one independent physical integrity comparator
completed **ALL THREE JOBS SUCCESS**. Model original checkpoint SHA, twelve
trained Pythia source MLP tensor SHA, WikiText TRAIN-A/B and DEV SHA,
each of THREE first-use evaluation dataset revisions/token SHA,
and every physical CB36 archive SHA are **bit-for-bit identical across
the two fresh CPU hosts**. Source/data/physical integrity PASS,
errors empty.

**Nuance: full-model PPL is NOT bitwise identical across hosts in v42.**
The independent comparison correctly flagged
`PPL_bitwise_identical_across_hosts: false`; original teacher WikiText
DEV was 173.3814885 on ubuntu22 versus 173.3811320 on ubuntu24.
The largest approximately 0.0113 raw PPL difference appeared
in the much worse uncalibrated WikiText baseline (1620.4023 versus
1620.3910). This does **not** invalidate exact physical bytes and SHA,
but prohibits describing v42 CPU outputs as bitwise numerically equal.
All THREE new-corpus **directions** for the primary matched-rate
joint-vs-local comparison did agree across both CPUs.

## Locked architecture + storage

Source: real **EleutherAI/pythia-14m**, exactly 12 source MLP projections
= 786,432 original trained coefficients. Original v41 same binary codec,
unchanged row bits/fp16 calibration/optimization limits and 343,840-byte
global physical cap (~3.49772 bits/MLP coefficient).
All four LOCAL un/calibrated, ungated JOINED and final B-gated
CB36 binary files are **ACTUAL 343,840 bytes**, including scales,
row symbols, CRC and outer SHA. Normal all-12 physical INT4 anchor
is **401,224 bytes** and thus above budget.

Original 2048 WikiText TRAIN-A local NLL **6.3493035436** versus
joint selected **6.3424172103** (ubuntu22), with separate 2048-token
TRAIN-B one-shot local NLL **6.3833079040** versus joint
**6.3773140013**. Predeclared B gate **ACCEPT** on both CPUs,
with consistent candidate binary SHA.

Local calibrated CB36 SHA256
`cd0209af68cf07a8dc81b0c9565e3e9971454e5f80840c7afaa2a4018ba1f64c`;
joint/B-accepted CB36 SHA256
`4f56f4ba7af027225ded8ce3fa8ab5fe9c9143a54999c096aa5ffe35caca2878`.
Both exact cross-host matches and identical to v41's frozen files.

## TRUE full-model pretrained PPL on 2048 tokens per never-before-used corpus

All 12 physical compressed MLP matrices were *restored as dense*
coefficients and substituted together in original Pythia-14m;
attention, embedding, norm weights unchanged. The following
measurements are from ubuntu22; ubuntu24 values are provided in the
independent 41364043512 comparator artifact and have small
numeric differences but the same verdict:

| Corpus (first-use) | ORIGINAL source PPL | LOCAL calibrated PPL, 343840B | JOINT accepted PPL, SAME 343840B | Joint minus local PPL | HIGHER-rate INT4 PPL, 401224B |
|---|---:|---:|---:|---:|---:|
| **BoolQ validation**, passage+question | 118.23845 | 300.58184 | **300.53376** | **-0.04808 (tiny improvement)** | 249.27317 |
| **ARC-Easy validation**, question+alternatives | 62.69741 | 158.83660 | **158.38078** | **-0.45582 (small improvement)** | 111.56901 |
| **Yelp Review Full TEST**, review text | 120.73456 | **285.77912** | 286.48576 | **+0.70664 (negative/worse)** | 245.15997 |

Known previously used WikiText DEV (1024 tokens) also measured:
original **173.38149**; local calibrated **440.06428**;
joint accepted **437.00548**; higher-rate INT4 **372.07758**.
Do NOT count this repeated corpus as fresh confirmation.

**HONEST OUTCOME: first-use domain directional improvement is 2/3,
not 3/3.** Tiny BoolQ (about 0.048 PPL) and ARC-Easy
(about 0.456 PPL) gains coexist with an explicit
Yelp regression (**+0.707 PPL**) under exactly the same
343,840 physical bytes. Even the better compressed candidates
are far worse than original pretrained teacher and physically
larger 4-bit anchor on ALL THREE new domains. A frozen TRAIN-B
gate accepted a model that degraded Yelp, showing it does
**not** guarantee positive out-of-domain quality.

This is exploratory 2048-token PPL per corpus, NOT BoolQ/SciQA
answer accuracy, Yelp classification quality, full-benchmark
statistical significance or broad cross-domain dominance.
No change of bits, gates, number of rounds, PPL sampling
or dataset list after evaluation.

## Decision for the next research step

Do **NOT** tune on BoolQ, ARC-Easy or Yelp now that all three
PPLs are visible. Global findings point to a fundamental
rate/quality gap even when calibration and allocation are improved.
Next honest step is a **frozen rate–distortion Pareto curve**
of the all-twelve-MPL original codec across multiple
*pre-registered* physical byte budgets (e.g., 3.5, 4, 5, 6 bpw),
using new validation texts, source-aligned strong INT4 anchors,
or else a genuine byte-backed blockwise native decode/compute
benchmark with end-to-end timing and memory. Keep compression,
quality and execution-speed hypotheses separate. No QMoE
0.8-bpw trained SwitchTransformer replication, new mathematical
multidimensional learned weight representation, deployable whole
model checkpoint or packed inference speed improvement established.

No changes to main, production Jessica, Render, Supabase, APK,
ordinary Jessica text tests or user data.
