# S3U independent S3T provenance pin (research-only; gate 0)

This document is an immutable-input specification, **not** evidence that S3U passed.

- Repository: Kamara229/JessicaCore
- S3T run ID: 43527496158
- Required run conclusion: success
- Required source branch: research/calibri2-nra2-s3t-unseen-wording-prereg-20261114
- Required source head SHA: c7c8b84d1247ee6939d7c5d894f364eea40d0f2e
- Required artifact ID: 14085702245
- Required artifact name: nra2-stage3t-unseen-grammar64-preregistered-no-score
- GitHub-reported artifact archive digest: sha256:41c3c67966d14443bd1011521b6d197e9dc22839b8581561a68f4cb6a1932cb1
- Required S3T precommit physical SHA256: 034ebd6e1cf6f20ad04ccd00e70c6bc07ee08a46f47da843121524f1d7f0faf3

Before reading labels or computing any candidate correctness, a future S3U execution MUST verify the run conclusion, branch, commit, artifact ID/name, download and hash the archive bytes, inspect its members safely, independently hash the expected prereg JSON, verify the prereg commitment and each exact prompt text/hash, and recompute original 616 J7 CPU token digests. The GitHub artifact digest is an archive digest; it is **not** a substitute for a separately computed JSON digest. The JSON byte digest is not yet independently established here.

On any mismatch or unavailable byte-level evidence: FAIL CLOSED. No scoring, training, candidate modification, label exposure, or acceptance. Preserve output_gate=0. Token span ABI for both reversed query mentions must match the frozen S3O mask; no inference from character offsets alone.

Only research branch changes are permitted. LIVE, Render, Supabase, APK, main, QA, TEST, original model weights and protected references remain untouched.
