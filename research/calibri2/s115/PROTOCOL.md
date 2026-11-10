# CALIBRI S115 — isolate Green Tunnel's first-token and continuation effects

**Research only: no model trained, promoted or merged. QA/TEST sealed. No LIVE Jessica, Render, Supabase or APK changes.**

## The causal reason for this experiment

S110 verified the original frozen S101 answers on all 456 authorized TRAIN samples: 353 exact complete+EOS, 387 correct first. Its 103 errors: **69 wrong first**, **18 wrong second**, **16 wrong final EOS**.

S113 independently reconstructed actual S101 first logits and reproduced **456/456** cached first choices. Gold-first vocabulary rank of the 69 wrong first tokens: 29 second, 23 ranks 3–5, 3 ranks 6–16, 1 rank 65–128, 9 ranks 129–512, 4 worse than 512. Most gold first tokens are ranked fairly high, but actual incorrect top1-versus-gold logit margin **median 7.09** and old S93 correct top1-versus-runner-up margin **median 7.05**; mere candidate rank is not enough for safe correction. Just 8/69 wrong first gold tokens occur in literal prompt candidate spans.

S112 trained 340 context-conditioned GT coefficients on real MPS, physically FP16 serialized, BPP **0.699822542**. Compared on all 456 frozen S101 TRAIN prompts it gained **one correct FIRST token** (387→388), did not lose any previously correct FIRST token, gained **one full+EOS**, but **forgot seven formerly correct full+EOS answers** (353→347), principally in S82_new family. It was **REJECTED, exact full Adam/weight rollback verified**. This demonstrates a possible mismatch between early first-token benefit and later generation damage, not proof of the source of damage.

### S115 hypothesis

The very same learned S112 Green Tunnel adjustments might help **during prompt prefill** (when first token and initial KV cache are determined) but damage previously correct **second/EOS** choices if applied again during subsequent cached decoder steps.

S115 does not learn or modify weights. It loads the exact REJECTED S112 saved FP16 sidecar and tests five predeclared, answer-independent runtime masks, each selected BEFORE generating a token:

1. `all_steps` — replicate original S112 rejected behavior, sanity control.
2. `prefill_only` — GT influences full prompt prefill/first-token, then turns OFF for every cached continuation step.
3. `prefill_and_second` — GT on for prefill and the next model step (second token), OFF for EOS/later continuation.
4. `decode_only` — GT OFF during prefill, ON for cached continuation.
5. `second_only` — GT ON only for the first cached continuation step (second token).

**Technical invariant:** The S101 original cache-aware decoder calls its frozen model once per autoregressive step; the first call has `past_key_values=None`, later calls have non-null cache. The S115 PyTorch pre-forward hook enforces this and aborts if altered. S95 alternative profiles S87/S91 always bypass S93 conditional GT. The original donor layer 3/9 GT weights are never duplicated or updated; only which decode call uses the paid shared donor residual changes.

## Hard checks before any TRAIN candidate can be considered

The original S101 physical model CB61, S94 numeric bank, S95 prompt router and S98 copy SHA-checked, original 353 full and 387 first reproduced. The rejected S112 file SHA-checked against its run report and every original weight tensor must be identical to S101. Each phase mask has one physically saved/reloaded FP16 sidecar and a **phase-specific counted runtime configuration**; ≤0.700000 all-in model bpp required. Each mode uses genuine S101 cached greedy S95/S98 generation for all 456 original authorized TRAIN questions, with **zero tolerance for loss of any of 353 full+EOS answers or 387 correct first tokens**, and paired 48-question alternated original S101 CPU benchmark requiring ≥90% relative speed.

A retained TRAIN improvement, if found, is only an unaccepted research candidate. A new, never-opened, once-only independent QA cohort, canonical native baseline speed and CPU/MPS parity would be needed before any model promotion. No oracle answer tables or per-TRAIN identity routing may enter inference.
