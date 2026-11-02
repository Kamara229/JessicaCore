# CALIBRI v66 — Functional noise shaping at exactly ZERO additional source-code bits

**Preregistered genuine-source experiment.** V63 lossless index entropy coding saved only 59,971 physical bytes; v65 physical locally rate-matched binary factors improved 3 of 4 genuine tensor output errors but worsened early MLP down. Next distinct idea transferred from signal coding: analysis-by-synthesis / error feedback, where we control the **direction** of unavoidable quantization error instead of using extra FEC parity bits.

## CALIBRI-FNS functional noise-shaping hypothesis

Freeze original full trained SmolLM2-135M-Instruct checkpoint, pinned revision 12fd25f77366fa6b3b4b768ec3050bf629380bac, and the literal full original CB59 archive from verified v59 Actions run 41507201766. The original packed format has one 5-bit index per 8 pretrained weight coefficients, K32 tensor-local FP16 codebooks, FP16 row scales, complete non-matrix FP32 tensors and tied-alias mapping, all CRC32/SHA256 protected. Full 134,515,008-unique-parameter checkpoint is exactly **11,229,303 actual bytes / 0.667839bpw**, and new modified archive must be EXACTLY the same file length. No centroids/scales/sidecar/task data added.

Given a row and a group g of eight source trained weights, use teacher TRAIN activation sequences X_train and original teacher output Y_train to measure row residual e=Y - QX. Hold all other groups fixed; examine all 32 existing centroid choices for this group, pick the one minimizing the remaining whole-row TRAIN response squared error **plus a fixed source-fidelity penalty**, and propagate the updated residual to the next group. Two deterministic sweeps through all groups (forward then backward). THIS is an analogy to analysis-by-synthesis and noise shaping, **not** literal classical sigma-delta modulation or channel FEC, and no signal recovery is claimed without quantitative heldout tests. A 5-bit code still consumes exactly 5 bits; the physical inference kernel is not modified.

## Preregistered actual training and measurement

- Model: original pretrained source (not synthetic), freeze public revision above and exact physical CB59 artifact. Dataset WikiText-2, genuine TRAIN tokens 0–127, independent VALIDATION tokens 0–127; record SHA for both. Only TRAIN teacher activations used to select codes, NEVER validation.
- Four preregistered matrices: early attention q, early MLP down, middle MLP gate and last MLP up. Exactly 64 uniformly spaced output rows per matrix; rows chosen by position, NOT by a validation or source-sensitivity oracle. Only **256 rows of four tensors** are touched, not whole pretrained source weight arrays.
- Quantizer: fixed K32 centroids, two forward/back source-code index sweeps, original row scales, source-correlation penalty lambda=0.10 of local source squared error weighted by TRAIN group activity. Also report lambda=0 diagnostic-only variant, but **do not** use validation to select candidate lambda, codebook, ranks or rows.
- Full physical archive: repack original 5bit code segments with exact byte lengths, patch only those segments, keep rest of archive identical, rebuild CRC/SHA, re-open/validate entire complete CB59 source including 1D weights and tied aliases. All bytes including metadata and checksums remain counted; all source trained 2D matrices remain physically compact and original native C packed lookup operator remains ABI-compatible.
- Measure TRAIN and VALID *on exactly the selected 64 output rows* before and after, for all four matrices. Report every source-code edit, full-file SHA, exact bpw. If validation errors worsen, report failure.
- No claim of a successful original-model language PPL/QA, no deployed binary runtime, no verified full-model output KL or prefill/decode speed. A full physical trained source file existing at 0.667839bpw is a **necessary, not sufficient** criterion for competent model execution.

## Why this is worth testing

Conventional independently nearest-centroid vector quantization minimizes a local source-weight error, often poorly aligned with output behavior: our v59-v61 results showed catastrophic model output drift even with a very small checkpoint. The transfer is to **distribute the lost information into less functionally important directions using only the original fixed-rate codes**. If the same rate yields lower validation layer-output error, stronger repeated cross-domain calibration and full model QA are warranted. If it fails, the approach overfits calibration and must be regularized or abandoned rather than claim magical reconstruction.

After genuine evaluation, compare v66 against unchanged CB59 and v65 binary source pilot. Future mixed-codec selection needs TRAIN-only global rate allocation and a single full source archive; live deployment remains explicitly prohibited.

Background references: DeepCABAC https://arxiv.org/abs/1907.11900 ; Frame Quantization of Neural Networks (2025) https://doi.org/10.1007/s00041-025-10182-7 ; CALIBRI coding catalogue in research/calibri/CODING_THEORY_ROADMAP_V62.md ; CALIBRI verified v65 in research/calibri/SESSION_65.md.
