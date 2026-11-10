# CALIBRI S112 — three real MPS steps for first-token-aware conditional Green Tunnel

**Research only. Not a deployable, accepted, or independently validated model.** No changes to main, Jessica LIVE, Render, Supabase or APK. No tests through Jessica chat, no reuse of previously inspected QA.

## Prerequisite

S111 all-12 prompt-conditioned shared dual-donor architecture must pass its *real* physical 456 cached CPU zero-coefficient parity guard and its 0.700 bpp / 0.90 speed checks. The unit/synthetic tests alone do not authorize training. The original immutable S101 physical CB61/S94 bank/S95 frozen router/S98 copy and S101 cache are SHA-verified.

## Training hypothesis

In S110, out of 103 wrong original full+EOS TRAIN responses, 69 diverged already on the FIRST token. S104 and S109's global scalar-gain MPS training reduced proxy loss but yielded 0 new correct full answers. S106 scaling x2 obtained one gained and one lost complete answer, suggesting potential movement at output decision boundaries coupled with forgetting. A prompt-conditioned 2×10×17 compact matrix trained with direct S101 first-token cross-entropy may better separate distinct question types than one global scalar for all.

S112 trains **only 340 FP32 in-memory coefficients** in S111, with the original S93 two Green Tunnel donor blocks at layers 3/9 completely frozen. S95 signed-hash vector uses only question text. No weights of the 36M compressed source, original S82 controller, S95 router, S98 copy, S87/S91 profile bank or token embedding are altered.

Three MPS steps; each takes two S93-route wrong-FIRST examples, one S93-route wrong-LATER example, and four previously S101-correct S93-route replay examples. Optimize S91 full response+EOS frozen policy surrogate; for wrong-FIRST samples add 1.75× direct cross-entropy of the **actual first S101 decoder logits** (before copy modifications that only begin at token 1). The only supervision is authorized existing TRAIN targets. Use Adam step=0.012 and clipping and bound each trainable coefficient to ±0.25.

## Physical and behavioral guard

Serialize the proposed coefficients once to a `torch.float16` (680 raw bytes) safetensors sidecar, store the original S101 physical tensors unchanged and count a required S111-compatible runtime descriptor. Exact byte count rather than parameter count determines bpp. Physically reload into a frozen original CPU S101 and, for each of the immutable 456 TRAIN prompts, run S95 selected profile, prompt-based 16D feature projection, all S101 cached decoder positions and original S98 second-token copy.

Require original reference 353/456 full+EOS and 387 correct first, no loss of **any** of those identities, all-inclusive physical bpp <=0.700000 and speed on same machine >=90% of unhooked S101. A rejected transaction restores the *entire* coefficient tensor and full Adam optimizer state exactly. Even if retention passes, only **>=1 genuine gained full+EOS TRAIN response** merits future independent evaluation. Exact speed versus original S82 (not S101), native C CPU and MPS parity are mandatory for eventual acceptance. QA/TEST remains unopened. No automated promotion.
