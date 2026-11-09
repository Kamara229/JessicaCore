# CALIBRI 2.0 — S94: lossless multi-profile Green Tunnel bank (research only)

## Decision based on S93's full 456 physically reloaded TRAIN audit

S93 improved the **individual** strict full/EOS answers to 268/456 at 0.6792 all-inclusive bpp and ~98.77% CPU speed vs S82, but failed the zero-forgetting rule: 19 of the 282 identities correctly answered by frozen S87 OR S91 are lost. More projected-gradient steps are not a reliable guarantee after greedy decoding and FP16 repacking. S93 remains UNACCEPTED.

## New intervention (not a blind continuation of gradient replay)

S87, S91 and S93 use the same original 36M CB59 complete pretrained source, fixed 5-bit codes, the same original S82 controllers, **only two modified paid MLP codebooks** and a small internal neural Green Tunnel at layers 3 and 9.

S94 asks a question that can be answered without learning a new answer table: **Can one physically compressed core plus the two small alternate codebook/Tunnel states fit in 0.7 bpp?** Instead of further destructive retraining, retain the three real learned alternatives and later choose exactly ONE at inference using a small *prompt-only* learned router. Inference cost would be one forward model if profile selection succeeds.

Frozen full 456 TRAIN reports imply an **oracle** attainable union of 287/456 (282 correctly answered by S87 or S91, plus 5 new only in S93). This is a mathematical upper bound over independent stored profiles, **NOT** a verified S94 model, useful ability, a trained router or generalization score.

S94 must:
1. Hash-verify all three genuine physical sources and full neural sidecars, including S87→S91→S93 ancestry.
2. Reject if any raw weights, alias, row scales, 5-bit codes, frozen S82 controller or other core dictionary differs, except the two known paid core dictionaries and the small internal Tunnel.
3. Physically serialize **ONLY** alternate numeric weights (no questions/answers, lookup table, TRAIN identities, prompt hashes or QA), with lossless zlib and complete per-tensor offsets, SHA, shape, dtype.
4. Count S93 physical source + S93 full sidecar + exact alternate bank + new manifest: hard cap 0.7 bpp. Never exclude an extra copy as unbilled memory.
5. Reconstruct a *single* PyTorch 36M CPU source from physical S93, hot-swap the paid alternate dictionaries + internal Tunnel, and verify representative full/EOS token sequences against independently frozen S87/S91/S93 full 456 CPU reports.
6. Keep candidate_accepted=false and run NO test in Jessica, NO LIVE changes, NO previous sealed QA/TEST, NO FP32 teacher.

Only after S94 passes bank parity/budget should S95 train/evaluate a real **prompt-only** neural router on frozen TRAIN, measure 456 hard no-forgetting, runtime speed and held-out TRAIN split. Nothing in S94 claims the oracle answer rate is attained at inference.

## Why this is different

Projected gradient protection cannot guarantee zero forgotten outputs when the greedy decoder crosses a discrete argmax boundary. Keeping the two genuine pre-update profiles **losslessly** opens a non-destructive route to preserving old knowledge, without duplicating full 36M model weights. If budget fails, S94 must report the negative physical evidence and investigate smaller exact/quantized profile encoding, NOT silently count the bank as free.
