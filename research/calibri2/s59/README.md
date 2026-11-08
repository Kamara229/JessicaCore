# CALIBRI 2.0 — S59: compressed-weight restoration that survives GPU exhaustion

## Evidence / separation

* The Kaggle API authorization is real, and the account reports **30.00 GPU hours remaining as of 2026-11-08**, resetting 2026-11-14T00:00Z.
* Actual private Kaggle GPU T4 submissions **did not receive a physical NVIDIA device** even though the saved kernel metadata was `enable_gpu=True`, `machine_shape=NvidiaTeslaT4`. Direct hardware probe found zero `/dev/nvidia*`, no working `nvidia-smi`, `torch==2.11.0+cpu`, and `torch.cuda.is_available()==False`. GPU quota stayed 30.00h. This is not an acceptable GPU training environment; **do not consume more quota on repeated submissions**. Kaggle phone verification/account GPU permission is a possible cause, not a confirmed diagnosis.
* All research remains under draft PRs #444–#446; no merge into LIVE/main, no Jessica chat tests, no Render, Supabase or APK changes.
* **Do not use** S26 inspected QA, sealed S29 QA, previously inspected S53/S55/S57 QA for loss, candidate selection, or model optimization.

## The substantive training change

We are moving beyond fitting just 64+64 contrasting answer strings. S59 contains a **genuine differentiable all-codebook TRAIN candidate**: the 36.38M compressed host's paid K32×8 FP16 dictionaries (all layers) and the already paid rank-12 Green Tunnel after-first adapter train jointly on public WikiText TRAIN text plus guarded original/augmented/historical TRAIN answer continuations. Its frozen rank-2 first-token adapter stays intact.

The same Python file works with `--device auto`, `--device cuda` when actual hardware exists, or `--device cpu` without changing the training task. No dense FP32 model or optimizer state is exported into inference. Physical inference weights are repacked to the **same CB59 5-bit group indices and FP16 dictionary slots**; the neural sidecar remains inside the original 0.7-bpp all-inclusive bound. Intermediate optimizer checkpoints are separate TRAIN-only continuation material, not extra deployed weights.

### Strict result gates

1. Physical reload proves the new packed CB59 can run, and full **native greedy + EOS**, not a teacher-forced token score, supplies correctness.
2. Preserve all 42 original and 27 augmented previously correct TRAIN answers, all 7 correct historical answers, and all 58+46 previously correct first-color tokens. Preserve the previously correct paired binding identities; improve two-sided complete pair correctness.
3. Do not call a TRAIN pass *generalization*. The first genuinely fresh blind QA created **after** candidate freeze, without reuse for selection, is required to prove a quality jump. An explicit baseline FP32 must use exactly the same unseen prompts and scoring.
4. The CPU-only benchmark ratio must still reach the user's target and be measured independently; four TRAIN timings are insufficient.
5. All complete bytes (repacked source, Green Tunnel, manifest) must be ≤0.700 bpp.

### CPU continuity

The candidate generator `paid_dictionary_and_green_tunnel_cpu_gpu.py` writes physically repacked candidates and `S59_STEP_####_TRAIN_RESUME.pt` at fixed boundaries. The latter contains optimizer state plus TRAIN-only provenance and SHA-pinned model identity. If Kaggle GPU is unavailable or quota ends, resume from the stored checkpoint with the **same script and tokenizer on GitHub Actions CPU**; numerical equality across hardware is not assumed and is checked with physical reload and complete free-answer response guards.

Example continuation (the three immutable physical inputs must exist):
```bash
PYTHONPATH=research:research/calibri python research/calibri2/s59/paid_dictionary_and_green_tunnel_cpu_gpu.py \
  inputs/s45/CALIBRI2_S45_FIRST_COLOR_GUARDED_NOUN_PAID.cb59 \
  inputs/s56/S56_DUAL_ACTIVE_PAID_GREEN_TUNNEL.safetensors \
  inputs/s56_report/S56_DUAL_ACTIVE_PAID_GREEN_TUNNEL_MANIFEST.json \
  results/s59 --device cpu --steps 8 --resume results/s59/S59_STEP_0008_TRAIN_RESUME.pt
```

### Research limits

S59 is a **hypothesis to test**, not an established generalization or FP32-parity breakthrough. Because the original CALIBRI 2.0 small host started from random weights, limited paired-task memorization may be an intrinsic obstacle. Broader actual-language training can help, but only unseen results decide. Until then S56 remains the accepted baseline. 
