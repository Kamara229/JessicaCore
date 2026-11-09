# CALIBRI S97 — frozen TRAIN token-position failure audit (research only)

S96 completed with **failure** in independent once-only 64-novel-prompt CPU evaluation:
- S95 routed **2/64** strict full+EOS; original S82 **0/64**.
- Among individually frozen S87/S91/S93 states, the perfect oracle also reaches only **2/64**: simply improving the router cannot solve this generalization gap.
- S87 **0/64**, S91 **2/64**, S93 **2/64**; individual first-token correctness was 40/64, and all finished with EOS.
- One-run CPU ratio was **0.8997473189**, technically below the >=0.9000 bound, though the difference is too small to extrapolate without a new independent timing measurement.
- Do not retune/retest on the already inspected S96 novel items.

S95 passed frozen 456 TRAIN (287/456, 0 forgetting, 0.6981 bpp), but its 4-way TRAIN heldout suffered 18 full losses and 1 first-token loss. The severe S96 failure highlights the problem of copying a new second token, not just selecting memorized profile weights.

S97 must be a strictly *read-only* structural diagnosis of first-token/second-token/EOS failure across the independently recorded frozen S82, S87, S91, S93 456 TRAIN output arrays, separating incorrectly chosen first tokens, early EOS at second position, incorrect noun/second token despite correct first, and correct body with wrong EOS. Include family breakdown, check preserved union 282, oracle upper bound 287, and summarize the once-inspected S96 report without reloading S96 questions or labels. No model training, no further inference against the prior S96 novel set.

S98 should then target train-only generalizable copy of unknown second-word nouns, while preserving old identity and CPU-speed guard. Future new unseen prompts must be disjoint from S96; never optimize on S96's observed examples. LIVE Jessica, main, Render, Supabase, APK and sealed earlier QA remain untouched.
