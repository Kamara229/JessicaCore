# NRA2 S3T — unseen wording and unseen topologies (pre-registration only)

## Confirmed previous results; do NOT reinterpret as decoder-language success
- S3R #43525586699: original frozen 43M CPU S194 616 J7 digests unchanged, new 64 input [64,90,320] genuine FP16.
- S3S #43525997831: 64 unseen topology questions, **same old S3M question grammar**: trained S3O edge+role head frozen; original 2241-parameter NRA frozen; soft edge 52/64 vs strict predicted hard edge >0.5 64/64 (12 soft false positives). Graph64/64, source64/64, target64/64. No retraining. These are **graph reasoning proxies**, not native generated full+EOS.
- S3Q input precommit: 5e9ea9cd66d5dedf0aa2cf574dcf78c8e0a7522c892f775fd8e9e014c46a26cb.

## S3T preregister first, before any CPU or new model score

- New 64 synthetic questions, 16 graph topologies **disjoint from S3G, S3M and S3Q**, same four graph families and 32/32 yes/no; four **new query phrases**, two destination-before-source and two source-before-destination.
- Same graph-header styles and edge literals as before: isolates query grammar from a novel graph distribution. Strict checks for header, length, family, grammar answer shortcuts.
- **Literal query phrase and exact R0..R5 mention character offsets** are recorded before CPU; token-level spans must later be verified with exact native tokenizer, not guessed from this prereg.
- SHA256 locks all prompts, topology IDs, literal mention spans and exact original S3O FP16 and S3R original frozen CPU hashes. No adapter weights, source/target gold, labels, predicted graph scores or NRA logits supplied to native model or J7.
- Next experiment is separate **read-only** 43M native CPU S194 on exact frozen S3T prompts, reproduce all 616 original J7 digests, ensure physical [64,T,320] FP16, and fail closed if original tokenizer cannot align novel question spans.
- Only **after** native CPU prefill and original 616 J7 verification, use unchanged S3O model (SHA `40a8903d5419cd8641b5d01e75e429e10b449304b3ba140d2245bf7f8a9d00b6`), frozen original 2241p NRA, source+target always SOFT and compare original SOFT edge probs to strict >0.5 HARD predicted edges; **no fit/calibration/tuning on S3T**. Report by family, wording, direction, positive/negative and fail codes. Hard/soft branch and score metric remain precommitted.
- Independent *final sealed* generalization still required. True native decoder full+EOS gain, unchanged 616 J7, total measured physical <=0.7 bpp and native CPU >=90% CB59 are acceptance gates not satisfied by S3T prereg.

## Isolation
No LIVE, main, QA, TEST, Render, Supabase, APK or Green Tunnel writes; no test through Jessica ordinary chat, no HTTP 429 retries. Research-only branch and draft PR; output gate stays 0.
