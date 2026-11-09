# CALIBRI S101 — exact compressed incremental KV-cache CPU inference

## Confirmed starting evidence

S99 physically achieved **353/456 strict full+EOS on 456 frozen TRAIN**, without losing any of the previous S95 287, S87∨S91 282, or legacy first-token successes; physical inclusive size **0.699573397602 bpp**, and paired speed ~94.28% of S95 on first 48 TRAIN. That is not an independent unseen result.

S100 one-shot developmental evaluation used new nouns wholly disjoint from the original S82 noun pool. It produced **42/64** strict full+EOS for S99, versus 0/64 for both frozen S82 and S95. S99 did not lose any of the previous zero correct items. However S99's same-run CPU speed was **0.8076094699 relative to the original S82**, while speed relative to S95 was **0.9096141159**. Its pre-registered hard baseline speed rule rejected S100. The inspected 64 S100 questions must not be inspected again for S101 tuning or validation.

## Intervention — performance only, no new weights or data

The original CPU reference re-runs the entire prompt+decoded prefix through the 36M model at every generated token, with `use_cache=False`. S101 changes only the inference execution path:

1. Feed the exact initial prompt once using a real `use_cache=True` model forward; store its causal KV activations and prompt hidden states for the frozen S82 context pointer.
2. For second/subsequent tokens, feed only the last output token with previous `past_key_values`. Leave all pretrained 5-bit compressed codes, exact S87/S91/S93 codebooks, trained Green Tunnel, S95 numeric profile selector, S82 controller, and S98 MPS-trained second-token copier **byte-for-byte unchanged**.
3. Keep the same order of native logits, first/late controller, generic pointer and S98 second-token copy override at fixed confidence 0.95. No gold targets at inference and no prompt/answer lookups.
4. Compare all 456 fully decoded CPU answers **token by token through EOS** against the independently re-generated original S99 full-prefix outputs on exactly the original authorised 456 TRAIN set. Reject if any token differs, including one single old skill.
5. Compare paired speed on the first 48 TRAIN questions of **three actual physical models**: unchanged S82 without internal Green Tunnel, the original S99 no-cache path, and S101 incremental cached path. Require S101 at least 0.9× original S82; also verify the baseline S99 353/456 and the same neural override count.
6. Store negative audit even if exact parity fails. Compressed physical size unchanged at 0.699573397602 bpp: a KV cache is transient inference activation memory, **not a new trained parameter**. S101 remains unaccepted until independent untouched sealed QA and native C / MPS parity.

No S96/S100 novel prompts, no prior inspected sealed QA or TEST, no new training, no LIVE Jessica, APK, Render, Supabase or main changes.
