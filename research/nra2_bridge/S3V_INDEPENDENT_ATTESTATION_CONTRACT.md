# S3V provenance attestation: fail-closed evidence contract (research-only)

Status: **BLOCKED / NOT ATTESTED**. This document is a specification, not an assertion that any artifact was independently verified. No scoring, training, candidate modification or output-gate change is authorized.

## Immutable producer vs derived auditor

The historical S3V producer writes JSON `output_gate: 0.0`. Its original bytes MUST remain unchanged. The downstream research audit requires an integer `output_gate: 0`. Never normalize the original bytes before physical SHA256 validation: instead validate the original producer schema, then emit a separate derived audit result with integer gate 0 and `candidate_accepted: false`. The derived report must record both input physical SHA and its own physical SHA.

## Independent inputs (not caller-supplied self-consistency)

Before processing S3V predictions, pin an independently obtained GitHub Actions run ID, run attempt, workflow path, head commit SHA, artifact ID/name, downloaded ZIP physical SHA256, exact member path and member physical SHA256. The expected values must come from a separate, immutable reviewed manifest, not from the S3V JSON or CLI arguments derived from it. Require all identifiers and hashes to match. If any independent reference is missing, return BLOCKED, never a verified flag.

Pin S3T artifact `nra2-stage3t-unseen-grammar64-preregistered-no-score`, CI run 43527496158, and prereg physical SHA256 `034ebd6e1cf6f20ad04ccd00e70c6bc07ee08a46f47da843121524f1d7f0faf3`. Verify its artifact bytes before parsing, and derive exactly 64 prompt SHA256 values using the *producer's documented byte encoding and canonicalization*, not a newly invented serialization. Require an exact index-to-prompt SHA match for all 64 S3V rows, not just uniqueness of sparse error rows. Preserve 16 disjoint topologies, four families, 32 yes/32 no, and both mention orders as precommitted; do not use correctness labels to change parsing.

## Trust gates

1. Verify original S3T bytes, source run provenance, all 64 prompt texts and SHA mapping.
2. Verify S3U native CPU capture source weight/tokenizer provenance, full [64,T,320] FP16 physical hash + reload, both query mention token spans against S3O ABI, and original 616 J7 CPU digests; require explicit true on *independently reproduced* checks.
3. Verify historical S3V artifact bytes and immutable producer schema including float 0.0; reject duplicate JSON keys and nonfinite values.
4. Only then create a derived 64-row evidence ledger, with independent row identities and recorded source/target/soft/hard outcomes. Never fill omitted roles by inference from correct answers. Reconcile row totals to all family, wording, mention-order and truth strata.
5. Keep `output_gate=0`, `candidate_accepted=false`; final blind, native full+EOS, <=0.7 bpp all-in and >=90% CB59 CPU remain separate unmet acceptance requirements.

Any missing run metadata, manifest, artifact bytes, tokenizer offset proof, J7 digest, or row identity is **FAIL CLOSED / BLOCKED**, not success. Do not run Jessica chat tests or modify LIVE, Render, Supabase, APK, main, QA, TEST, protected weights, or the unrelated CALIBRI+Green Tunnel task.
