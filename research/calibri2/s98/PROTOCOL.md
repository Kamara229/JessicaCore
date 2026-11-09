# CALIBRI S98 — MPS-trained owner-anchored second-token noun-copy scorer (TRAIN only)

## Hard empirical boundary

S96's once-only evaluation of 64 independently newly authored prompts failed: only 2/64 strict full+EOS for S95, against 0/64 S82; the union oracle of S87/S91/S93 also yielded only 2/64. Its CPU ratio 0.899747 was just under the 0.90 boundary. **S96 prompts and answers are now retired and must never serve as training material, model selection, or evaluation again.**

S97 independently extracted a frozen 456-TRAIN token failure breakdown: S93 had 386 correct first tokens but 105 wrong second tokens, and 13 cases with correct first two tokens but wrong suffix/EOS. This motivates a different architecture from the S95 router: **a generalizable second-token copying operation** conditional on the model's existing generated first token.

## Mechanism

S98 trains a tiny neural scorer with **12 purely relative prompt features and an 8-unit hidden layer**, after locating in the prompt any literal two-word sequence whose first token matches the already-decoded first word. A generic owner-matching feature is derived exclusively from repeated proper names in the input text; it does not store explicit owner/name/noun tokens, answers, QA hints, question IDs, lexical lookup tables or a special-case S96 template.

The scorer receives only context, positions and relative owner distance. It ranks **candidate SECOND tokens actually present** in the question and predicts nothing if no reliable candidates exist. The compressed core, its first/late controllers, internal Green Tunnel, and S95 router are all FROZEN and UNCHANGED.

Use the original S82 256 authorised TRAIN prompts. Split by contrastive noun groups, not random individual prompts: fit on groups 0–5 (192), evaluate on groups 6–7 (64); also measure a separate fourfold leave-one-instruction-frame-out TRAIN diagnostic. Both splits are **held-out TRAIN, not independent QA**. GPU gradients must run on actual Apple MPS, no CPU fallback.

Save only numeric trainable FP16 weights in safetensors and charge all extra bytes plus diagnostic manifest atop the complete conservative S95 physical model budget. Refuse >0.7 bpp. This is a **conditional second-token feasibility**, not proof the complete network can choose the first token, EOS, or reach any new full answer. `candidate_accepted=false` always.

## Subsequent gate, not part of S98

To modify the compressed Green Tunnel output policy, a later research stage must physically integrate the new scorer with S95, guard every previously correct 282 historical TRAIN identity and all first tokens, complete full autoregressive EOS generation on all 456 TRAIN, and measure CPU speed. Any independent new evaluation must use an untouched set disjoint from previously inspected S79, S85 and S96. LIVE Jessica, main, Render, Supabase, APK and Jessica relay tests are off limits.
