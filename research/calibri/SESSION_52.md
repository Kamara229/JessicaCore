# CALIBRI v52 — six-window, two-corpus-family TRAIN-only acceptance gate

**Research-only single next stage**, after v51 PR #181. v51 verified on two CPUs: full pretrained Pythia14m INT6 CB46 10,803,661 physical bytes, three reproducible CRC/SHA-bound FP16 row repair CR47 sidecars (6/12/24% per-tensor budget, each ~610k bytes), all within **6.5 physical bits per parameter**, and baseline warm inference within <=20% of pretrained FP32 teacher. v51 rejected every correction under an over-conservative two-domain gate (WikiText TRAIN B and one AG News TRAIN window) even though a rejected cap6 correction improved all three previously unused heldout test excerpts. These revealed test excerpts MUST NOT train or tune v52.

## Frozen one-step research contract

Keep physical pretrained CB46 base, PRE scale calibration on WikiText TRAIN-A [0:2048], source-only fixed deterministic original-vs-quantized SSE rank, full exact CR47 row correction and 6%/12%/24% per-tensor cap allocations UNCHANGED from v51, including entire physical byte count and SHA binding. Include separate 15,480-byte potential POST correction reserve within 6.5 total bpw. No modifying quantizer or repair candidates based on any new TEST text. The scope of v52 is ONLY changing the predeclared TRAIN acceptance test to estimate window-to-window stability instead of requiring every one of two train samples to improve.

**New six fixed 2,048-token TRAIN windows** (first 2,048 real Pythia tokens per range; WikiText is continuous token positions):
- WikiText TRAIN C [4096:6144] and D [6144:8192], disjoint from PRE fit A, from old TRAIN B, and from each other.
- AGNews TRAIN[512:640], TRAIN[640:768], TRAIN[768:896], TRAIN[896:1024]; each disjoint from previously measured AGNews TRAIN[:512] and other windows.
- Note: six windows belong to only **two corpus families**, not independent task populations. This is a heuristic selection filter, not statistically calibrated confidence interval.

All physical CB46/CR47 files are signed, saved, reread and CRC/SHA/source/shape verified **BEFORE** any training-gate sample is loaded and scored. The model loads from physical compressed source into FP32 RAM at cold start, applies FP16 repairs once (this is INT6 ON DISK, not native packed INT6 runtime).

**ONE fixed gate**, all six sampled training windows measured for baseline PRE and 6/12/24 candidate physical model; candidate eligible iff (i) positive NLL gain vs PRE on **>=4/6 windows**; (ii) arithmetic mean of six NLL gains >= +0.0005 nat/token; (iii) worst individual window NLL gain >= −0.0100 nat/token. Among eligible choose maximum mean gain (tie lower per-tensor cap: 6%, 12%, 24%); otherwise select plain PRE with zero sidecar. Do not revise thresholds after inspecting heldout. No TEST labels, TEST text or TEST PPL influence selection.

**New first-use heldout record slices** after gate: Yelp Review Full TEST[576:640], OpenBookQA main TEST[448:500], QNLI validation[512:640]. Each evaluation uses first 2048 Pythia tokens of rendered text, not supervised classification accuracy. These are new records but same broad dataset families as past CALIBRI experiments; report the limitations and every PPL for original pretrained teacher, plain INT6 and each candidate, even if rejected. If any excerpt cannot supply 2048 tokens, fail instead of silently reusing earlier revealed test records.

**Real speed gate**: batch-1 full-model Pythia 128-token prefill +32 KV-cache autoregressive steps, 2 warmups and 7 real median timings, original pretrained teacher versus TRAIN-only selected candidate, on Ubuntu22.04/24.04 pinned CPU AVX2 and 2 threads. User-defined required warm slowdown <=20% versus original; startup decompression separately accounted. Both hosts plus third independent comparator must reproduce source and candidate physical SHA, rate/row metadata and exact six-window selection decision. Negative task PPL is a result, not CI hash error.

Research branch/draft PR only. Do NOT modify production Jessica, chat, Render, Supabase, APK, credentials, model weights in main or secrets.
