# S3X — immutable S3V provenance and complete-row ledger protocol (NO execution)

Research-only design. **output gate=0**. No model inference, training, scoring, labels inspection, threshold changes or CI execution in this PR. This document is NOT evidence of successful validation.

## Problem
S3V `disagreement_or_error_cases` records only cases with incorrect soft/hard answers or answer disagreement. S3W fills omitted cases with correct source and target roles. That inference is unsound: both answers may be correct despite incorrect role predictions. S3W role/order slices are **not independently verified**. Prior S3V aggregate scores remain historical, not independently reproduced.

## Required independent evidence before interpreting role-order slices
1. Identify exact immutable S3V run ID, commit SHA, artifact name, downloaded ZIP SHA256, member path, raw JSON byte SHA256 and canonical schema/version; verify hashes after reloading, fail closed on any mismatch. Keep source S3T prereg SHA256 `034ebd6e1cf6f20ad04ccd00e70c6bc07ee08a46f47da843121524f1d7f0faf3`, S3U physical feature manifest, tokenizer offsets and S3O FP16 SHA pinned; do not replace original artifacts.
2. Audit only existing evidence first. Check S3V record uniqueness, case IDs, membership in committed S3T 64, original prompt SHA, family, wording and mention order; assert that each recorded error satisfies the producer emission predicate. Missing cases must be marked `role_status=unknown`, NEVER assumed correct. Report role slice ranges/unknown counts rather than manufactured exact values.
3. A genuinely complete 64-row ledger requires **separately preregistered** read-only capture from frozen, pinned inputs, after verifying original 616 J7 CPU token digests and protected 584 full+EOS. Each row: immutable ID, prompt hash, family, wording, mention order, truth (sealed until gates), source/target predicted IDs, soft/hard answer predictions, role confidences, edge logit, threshold exactly sigmoid(edge_logit)>0.5, artifact and tensor hashes, provenance and schema version. Validate 64 unique rows and exact prompt identity; no recomputation on changed data.
4. Keep all truth/score exposure gated behind verified source J7, S3T prereg and physical manifests. If any check fails, fail closed; no labels or score output. No tuning based on S3T; any later independent blind acceptance must use a separately sealed dataset.
5. Original source weights, native tokenizer, S3O adapter and NRA stay physically immutable. Compare source hashes before/after. This proposal neither establishes native decoder full+EOS improvements nor all-in <=0.7 bpp or >=90% CB59 CPU speed.

## Scope and acceptance
Documentation-only protocol; do not modify main, LIVE, Render, Supabase, APK, QA, TEST, original weights or protected reference sets. Do not trigger Jessica (HTTP429). No CI/workflow added. Any future implementation must be separate isolated research draft PR and must not treat post-hoc role slices as ground truth. **Candidate not accepted.**
