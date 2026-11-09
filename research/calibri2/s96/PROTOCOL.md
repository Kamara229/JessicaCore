# CALIBRI S96 — one-time genuinely unseen prompt/noun validation (RESEARCH ONLY)

## Fully measured evidence before S96 authored

S95 real frozen FP16 neural router plus S94 three-profile Green Tunnel weight bank:
- physical 456/456 CPU output replay succeeded.
- **287/456 full+EOS correct on the 456 TRAIN cohort, zero full or first-token legacy forgetting** (282 protected S87/S91 union plus 5 extra S93-only).
- **0.698105797395 all-inclusive physical bpp**, under 0.7.
- Crucial caveat: 4-fold **held-out-TRAIN** evaluation still suffered 18 full-answer and 1 first-token forgetting errors in aggregate. S95's perfect TRAIN selection is not a proven generalization achievement; native C performance and sealed QA remain unverified.
- S95, S94, S93 frozen byte artifacts MUST NOT be altered or trained on S96 prompts.

## This step

Create a new, never-trained evaluation set consisting of **64 newly authored prompts**: eight tokenizer-compatible nouns from the previously excluded tail of S82 noun list, with two newly authored instruction frames, four colors, contrastive swaps, and distinct ownership terms. Verify all target nouns were excluded from the S82 new TRAIN corpus and every exact prompt is absent from the historical 456 TRAIN.

Load the actual archived S95 FP16 router weights, the exact S94 weight-only bank, and the S93 physically compressed CB61 source with its frozen controller/Green Tunnel. **No training, no choosing a new seed, no after-the-fact optimization**. Use actual 36M CPU decoding including EOS to assess frozen S87, S91, S93, and the routed S95 answer; the oracle union remains clearly labelled a ceiling.

Separately reconstruct original S82 from the immutable original 36M CB59 core and S82 controller. Measure per-prompt real CPU latency and full+EOS differences between frozen S95 and S82. Define failures independently of prior TRAIN: losses relative to S82, missing new gains, or CPU speed ratio below 0.9. Save all results even on rejection.

Never use older sealed QA / TEST (including S79 and S85), never modify LIVE Jessica, Render, Supabase, main, APK, or model deployment. A passing S96 does not substitute for formal sealed evaluation or native C/GPU parity.
