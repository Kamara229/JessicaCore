# CALIBRI v34 — physical rate–quality study at a materially tighter 3.5 bits/weight

## Clarifying the stage already completed

PR #160 / v31 was the **CPU determinism and effective dispatch** experiment,
not the storage/PPL experiment. PR #161 / v32 and PR #162 / v33 have **already**
measured genuine physically serialized real Pythia-14m weights and WikiText
perplexity using one changed pretrained 128×512 MLP projection.

Previously confirmed (both independently run GitHub-hosted CPUs):
- Original unchanged-model WikiText validation PPL: **173.38149**.
- Physically stored rowwise INT4: **33,042 B**, **179.67859 PPL**.
- TT+adaptive residual v32: **33,006 B**, **219.21966 PPL**,
  Python nonfused stream 127–138× slower than decoded dense.
- TT-free adaptive 2D INT3/4 v33: **32,980 B**, preselected
  8×64 tiling **180.93612 PPL**, still **0.70% worse PPL**
  than physically stored INT4 at only 0.188% less bytes.
  Its Python stream was 95–125× slower. Both v32 and v33 are
  **negative quality/latency findings** versus simple INT4.

## New v34 test hypothesis and preregistered measurement

At **exact 3.5 bits per coefficient including ALL metadata** the
strict maximum stored bytes for this one 65,536-coefficient matrix is
**28,672 B**. This is **13.2% below** the earlier 33,042B
INT4 comparison, rather than a cosmetic 36–62-byte reduction.

Use real pretrained `EleutherAI/pythia-14m`, source checkpoint
revision recorded and compared across independent runners, same
`Salesforce/wikitext` WikiText-2 train/validation tokenization and
fixed evaluation subset (256 train, 1,024 validation, context 128).
Select the alphabetically first 128×512 MLP projection out of 12.
No other weights may change. Calibration uses genuine **train-only
MLP activations** and second moments.

From identical 3- or 4-bit physical block codec (CRC32, per-tile FP16
scale, bitwidth byte, packed symbols and 20-byte header) predeclare:
- rowwise 1×512 per-row mixed INT3/4 under ≤28,672B;
- 2D 16×32 adaptive blocks under the SAME ≤28,672B;
- 2D 8×64 adaptive blocks under the SAME ≤28,672B;
- a **uniform 3-bit rowwise low-rate anchor** (expected ~24,980 B);
- old physical **INT4 above-budget reference** (33,042 B) as a
  higher-rate quality anchor; DO NOT present it as matched-rate.

Start from all INT3; upgrade tiles to INT4 using the positive
train activation-weighted MSE decrease, ranked by gain under the
**actual byte ceiling**. Hold each topology fixed and choose a
single winner by the **training proxy** BEFORE seeing validation
PPL. Record all three full-model validation PPLs for exploratory
comparison, alongside original, rowwise pure INT3 and over-budget
INT4, after actually writing/reading CRC-protected physical files.

Two independent Linux GitHub Actions hosts pinned to PyTorch 2.5.1
CPU, Transformers 4.46.3, selected ATen `AVX2`, requested
oneDNN AVX2, deterministic ON and 2 threads. Verify source
model/data/token/weight SHA256, file SHA256/real bytes and
cross-host results. Retain negative results. No speed benchmark this
step: v32/v33 already established the Python reference cannot compete
with optimized dense GEMM, while this v34 aims solely to measure
rate/quality before optimizing native packed streaming.

### Decision gate

Do NOT call a variant superior merely because its PPL is lower at a
higher byte count. Compare the **three fixed-cap candidates** at
their true bytes; rowwise fixed-cap is the **rate-matched** control.
The lower-rate uniform INT3 and over-budget INT4 are boundaries, not
rate-matched baselines. If no 2D candidate improves on rowwise
fixed-cap mixed bits, record a failed multidimensional quality
hypothesis and do not deploy a native kernel. No state-of-the-art,
0.8-bpw/QMoE or full-model checkpoint-storage claim.

The small repeated WikiText validation subset is exploratory: prior
CALIBRI versions have seen it. Full-model PPL measures a dense-restored
physical representation for **ONE** projection; nothing proves speed
or memory benefits for the full model. Other methods (per-row quant,
adaptive bits, tensor cores) are existing prior art, not established
original mathematical weights.

All work in research branch and draft PR only. Do NOT modify main,
Render, Supabase, APK, production Jessica, data, or secrets. A
user-visible regular Jessica chat test is NOT requested.
