# CALIBRI v43 — preregistered physical MLP rate–distortion curve, NOT full-model disk size

## Ground truth from preceding v42

v42 research-only PR #172, Actions 41364043512: two fresh GitHub-hosted CPU runs plus integrity comparator PASS. Real pretrained EleutherAI/pythia-14m, 12 source MLP matrices, 786,432 source MLP parameters. Frozen TRAIN-A 2048 WikiText tokens, TRAIN-B next disjoint 2048 as one-shot internal gate; rate-matched locally calibrated and jointly optimized CB36 MLP binaries each **343,840 actual disk bytes = 3.49772 effective bits per MLP parameter** (including per-row FP16 scales/CRC, bitstreams and CB36 container SHA). First-use BoolQ PPL delta JOINT–LOCAL **−0.048**, ARC-Easy **−0.456**, Yelp **+0.707 (WORSE)**. These small gains were not universal, and all compressed all-12 models had substantially worse absolute PPL than original and a physically larger INT4 anchor. Source and packed physical SHA were identical on both hosts; PPL numerics differed slightly between hosted CPUs, so do NOT claim bitwise identical CPU model inference.

## v43 measurement hypothesis — freeze rate and training before new evaluation

Measure actual rate-vs-quality over **physically serialized** row-quantized original source pretrained MLP weights, instead of extrapolating a claimed 0.8-bpw result or comparing rates with guessed metadata.

**Do not tune any method to held-out perplexity.** Reuse v42 model/checkpoint, codebook and train-only control exactly: 2048 WikiText TRAIN-A activations for original local 3/4 bit selection and v35 FP16 train-output row scale calibration; identical v39 two-round, up-to-20 proposals/round jointly measured TRAIN-A NLL zero-byte row swaps subject to 75%–125% per-MPL INT4 row precision bounds; one disjoint 2048-token TRAIN-B acceptance gate, strict improvement >1e-6. Preserve both local rate-matched calibrated and final B-gated joint calibrated 343,840-byte CB36 records and their SHA. Do NOT reopen SQuAD, IMDb, BoolQ, ARC, Yelp or SciQ to decide model settings.

**Additional pre-registered physical quantizers**: keep established v15 CF15 signed per-row INT4, INT5 and INT6 physical bitstreams, original row maxima scales stored as FP16 with CRC32. For EACH fixed rate, record both true v15 uncalibrated file and a SAME-EXACT-BYTE calibrated copy: adapt only existing per-row FP16 scale and integer symbols by searching the frozen v35 scale factor grid and minimizing the real source MLP TRAIN-A feature-second-moment-weighted row error. This is a diagonal approximation, not the v35 full train output covariance and not a fair same-codec calibration superiority claim. The FP16 scale and physical format have the same bytes before/after each high-rate calibration. Training-A only; no TRAIN-B, WikiText DEV or new corpora influences the additional high-rate model parameters. Only the fixed v41 3.5-bpw mixed-width model uses TRAIN-B for the already-frozen one-shot gate; high-rate baselines do not run test-selected gating.

**Physical rate** is defined as TOTAL ACTUAL 12-MLP CB36 bundle bytes read from disk divided by 786,432 source MLP coefficient positions, including all CF15/CT33 per-module headers, FP16 scales, CRCs, CB36 layer descriptors and outer SHA. Planned actual points, subject to CI verification:
- v41 rate 3.49772bpw: **343,840 bytes**.
- CF15 INT4 actual **401,224 bytes ≈ 4.08146bpw** (NOT within strict ≤4bpw).
- CF15 INT5 actual **499,528 bytes ≈ 5.08146bpw** (NOT ≤5bpw).
- CF15 INT6 actual **597,832 bytes ≈ 6.08146bpw** (NOT ≤6bpw).

The raw code symbol widths 4/5/6 bits are NOT the physical all-in rate. These rates correspond to distinct existing families (CT33 mixed3/4 with joint cal vs CF15 fixed-per-row 4/5/6, raw and train-moment-scaled); therefore this is an **observed rate–distortion tradeoff curve for named methods**, not proof of a globally optimal achievable Pareto frontier or mathematically novel tensor encoding.

**Frozen first-use evaluation**: after ALL 10 physical files (4 mixed 3/4 and 6 CF15 high-rate) are written, reread and SHA verified, load exactly two different, never previously examined public domains, 2048 original Pythia tokens each:
1. `allenai/openbookqa`, `main`, `validation[:128]`: `fact1`, `question_stem`, all choice texts; exclude correct-answer labels.
2. `stanfordnlp/snli`, `validation[:128]`: concatenated premise and hypothesis, exclude entailment labels.
And WikiText2 original 1024-token DEV for history ONLY (not fresh corroboration).
Pin each dataset revision/token SHA, source Pythia-14m checkpoint revision and all 12 original tensor SHA. Compute original full-model PPL, all physical decoded candidates’ full-model PPL (12 original MLPs replaced; attention, embedding/norms unchanged); full numerical inference operates on reconstructed **dense** tensors; measure no native packed serving time in this study. Distinguish calibration changes at IDENTICAL high-bit file bytes and quality changes from buying EXTRA stored bits.

Repeat independently on hosted Ubuntu 22.04 and 24.04, Python 3.11, CPU PyTorch 2.5.1, Transformers 4.46.3, selected ATen AVX2 and oneDNN AVX2 request, deterministic and two threads. Independent third CI step MUST check strict source/token identity, physically persisted binary SHA, exact all-in lengths (343,840; 401,224; 499,528; 597,832), per-rate CRC decode, correct training B gate and all measured PPLs. Report PPL host numerical differences honestly instead of assuming bitwise consistency; distinct outcomes of rate/calibration are scientific outcomes even if PPL gets WORSE.

## Success / negative gates

A positive monotonic quality-vs-rate trend would provide evidence that this particular rowwise all-12 3.5bpw method operates far below the useful-quality part of its rate curve; it would NOT validate QMoE 0.8bpw or a new graph/multidimensional neural weight primitive. Calibrated fixed-bit may improve or worsen true full-model PPL despite TRAIN moment error decreasing. Do not confuse local code-symbol bits with physical total. Compare original and CF15 same bitwidth before/after as separate tests; compare actual sizes in the same table. If any first-use external PPL fails, preserve it, do not drop a corpus post-hoc.

These bundles store ONLY the 12 MLP matrices and are not complete model checkpoints. There is NO byte-backed GEMM, ARM/mobile performance, full-model memory saving, successful end-to-end deployment, SwitchTransformer/QMoE reproduced 0.8bpw, or newly invented learned multidimensional neural architecture.

Research only: isolated branch and draft PR, NEVER change main, working Jessica, Render, Supabase, APK, data or ordinary Jessica chat tests.
