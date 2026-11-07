# CALIBRI 2.0 S0 — next TRAIN-only anti-collapse pilot (precommitted 2026-11-07)

## Verified baseline (no inference beyond TRAIN)
- Random SmolLM2-compatible 320x6, 22,368,320 unique parameters; physically 1,932,195 bytes / 0.6910469807298895 bpp. No pretrained FP32 neural weights.
- CI #42366424415 SUCCESS, 64-step mixed WikiText + 24-pair TRAIN: 0/16 full exact + EOS, 2/16 correct first tokens, one distinct greedy first token, repetition excess 112; exported SHA256 3a87686b629f2fd3d19a3ac4ee028a970969dfb88e5b7870b53fec5b1edc49d2.
- CI #42368066171 SUCCESS: one frozen opposing TRAIN pair becomes first-token-correct at 32 steps, but full + EOS not both correct at 64.
- CI #42368307719 SUCCESS: same opposing TRAIN pair, two independently initialized-identical arms. Centroids-only achieves both full answers + EOS at 128 steps; centroids+5bit achieves it at 96 and 128. Both keep 0.6910469807298895 bpp. This proves TRAIN-pair memorization, not broad instruction following or FP32 parity.

## Next experiment: paired retention under a second contradiction (no automatic model growth)
Hypothesis: loss interference, not inability to express opposite first tokens, drives mixed-TRAIN collapse.

Precommit *before execution*:
1. Use exactly two frozen existing synthetic TRAIN pairs from `v186.pairs.build_pairs(tok,"TRAIN")`: pair IDs 0 (yesno_numeric) and 6 (yesno_binding), preserving original prompt/answer/EOS token IDs and immutable source SHA. No new data, public DEV/VALID/QA/TEST, teacher weights or posthoc prompt selection.
2. Start BOTH independent arms from the exact original random source SHA256 2e807eb9c4f82018bcc72f0924726377cafb70df1ad529aefbd8513b6e538d80. Train first pair 0 for exactly 96 updates, then pair 6 for exactly 96 updates, alternating old/new on each subsequent step for exactly 64 updates. Total fixed 256 steps per arm, maximum 2 arms; this is a bounded diagnostic pilot, not an open-ended benchmark optimization.
3. Arm A: FP16 paid centroids with fixed 5-bit indices. Arm B: same initialization, schedule, optimizer, and centroids plus bounded 5-bit proposals every fourth step (<=512 groups/proposal; accept only if same current TRAIN objective improves). Do not train unencoded dense parameters, scales, adapters or external Green Tunnel weights. If compute quota exceeds the approved pilot, stop rather than silently spending more.
4. At precommitted steps 0, 96, 128, 160, 192, 224, 256, run greedy exact full answer + EOS probes on **all four TRAIN prompts** (both sides of each pair), not just the current training prompt. Log generated token IDs and readable strings, expected IDs, EOS termination, per-pair exact/first-token counts, distinct first tokens, repetition excess, and loss of correctness on pair 0 when pair 6 is introduced. No early stopping, hyperparameter search or checkpoints selected by these results.
5. Export complete physical CB59 source at the end of each arm. Assert unique parameter count 22,368,320, bytes <=1,957,228 (0.700 bpp), exact archive size invariant from bootstrap, decode/reload validity, full SHA256. Run native C forward numerical parity against decoded codebooks, plus CPU speed against an explicitly comparable CB59 baseline; **do not claim >=90%** until measured with identical hardware, threads, sequence and generation settings.
6. A valid positive retention finding requires all four TRAIN prompts to generate exact full answers + EOS at the precommitted final checkpoint, and both original pair-0 answers still correct after learning pair 6. Otherwise report interference as failure, preserving all raw output. This is *not* a generalization or parity finding.
7. No model growth to 134,515,008 parameters unless stable useful generation on 22M is demonstrated on a materially broader, predeclared TRAIN curriculum. Never move to 360M; retain old 135M v001.

## Execution boundary
Implement only in `research/calibri2-s0-small-first-paid-architecture` / draft PR #376. CI only, maximum two new CI runs per inspection; no changes to main, LIVE Jessica, Render, Supabase, APK or unrelated relay branch. Jessica is only a text relay and does not run tests.
