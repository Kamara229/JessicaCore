# CALIBRI v49 — deterministic INT6 row-precision allocation

**Single research next step.** v48 PR #178 and GitHub Actions run 41482467308 showed the frozen TRAIN-only gate could choose the SSE-matched correction and improve both fresh Yelp and ARC-Challenge excerpts, while all warm Pythia generation speed ratios stayed below the user's 1.20x limit. But two independent CPU hosts produced **different physical CR47 files (610655 vs 610619 bytes, different SHA)** even while sharing the exact full-model CB46 SHA and choice, due to host-sensitive TRAIN-gradient ranking determining per-module row counts.

v49 targets reproducibility and stability, **not** speculative superiority over all standard quantization, with one frozen 6.50bpw full-model disk cap. The complete physical trained Pythia-14M model remains CB46 10,803,661 bytes, INT6 per eligible 2D matrix, unmodified 1D vectors. Reproduce PRE MLP scale calibration on WikiText TRAIN A tokens [0:2048].

## Predeclared selection algorithms (no test text touched)

Compute original `W` minus decoded persisted INT6 `Q` per trained 2D row using exact FP32 values promoted to Python binary64. Accumulate `sum((W_j-Q_j)^2)` with scalar `math.fsum` and fixed integer score scale 10^15, then deterministic sort by score/actual physical row byte cost (with lexicographic tensor/key/row tie breaks). This avoids host-dependent BLAS gradient-based per-module quotas. Persist real CRC/SHA CR47 parent-CB46-bound FP16 correction rows, true geometry/index/metadata overhead included.

Compare ONLY two fixed source-derived precision profiles:
1. `sse_global`: highest original-versus-INT6 row error per physical correction byte across complete pretrained source.
2. `sse_balanced_12pct`: select one valuable row from every eligible trained 2D tensor, then select by the same value-per-byte score but never spend >12% of allowed patch budget on one tensor. No learned module importance from evaluation PPL.

Cap real complete CB46 + CR47 physical bytes + **15,480 bytes reserved** for possible C45C sidecar at **<=6.50bpw of ALL original source parameters**. No actual C45C in this session. Compare actual candidate bytes explicitly rather than falsely claiming byte-exact equality between global and balanced. Unlike v48, requires **identical CR47 file SHA256 and bytes across Ubuntu22 and Ubuntu24**; integrity fails if any key or sidecar bytes disagree.

Freeze candidates and save/read/verify all physical files. Then ONE conservative gate on two TRAIN-only domains: WikiText TRAIN B tokens [2048:4096] and **AGNews TRAIN[128:256]**, which is disjoint from v48 TRAIN[:128]. For each candidate, compute min(base_train_NLL−candidate_train_NLL) across both; choose highest minimum if >=1e-5, else plain PRE. No repeated tuning or decisions on tests.

Only AFTER one-shot gate, load new fixed 2048-token excerpts: Yelp test[384:448], OpenBookQA test[128:256], GLUE QNLI validation[128:256]. These are new record windows of previously used *families*, not broadly independent benchmark tasks. Compare teacher, base, global, balanced PPL and report every regression. Benchmark selected full Pythia warm real 128-token prompt + 32 KV-cache decode steps (2 warmups+7 repeats) versus original teacher, enforcing user's 20% slowdown cap as an experimental verdict. Crucial caveat: INT6 is **only the disk/checkpoint format**. Model startup always expands entire compressed trained source into hot FP32 RAM and applies physical residual rows once, so this is no 6-bit active-RAM/native packed model.

Work entirely in isolated research branch/draft PR; no production backend, Jessica chat, Render, Supabase, APK or secrets.
