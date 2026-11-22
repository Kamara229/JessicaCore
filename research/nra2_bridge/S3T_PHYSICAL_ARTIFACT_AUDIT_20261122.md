# S3T physical artifact provenance audit (research-only, 2026-11-22)

Source: GitHub Actions run 43527496158, artifact ID 14085702245, name `nra2-stage3t-unseen-grammar64-preregistered-no-score`.

- GitHub Actions artifact ZIP SHA256: `41c3c67966d14443bd1011521b6d197e9dc22839b8581561a68f4cb6a1932cb1`.
- ZIP contains one JSON: `NRA2_S3T_NEW_GRAMMAR64_PRECOMMITTED_NO_SCORE.json`, raw byte SHA256 `537d0470912c1dbd1c630f3488661464a3aa42ed643349e00fb58db8391919f9`, size 23505 bytes.
- JSON field `commitment_sha256` equals `034ebd6e1cf6f20ad04ccd00e70c6bc07ee08a46f47da843121524f1d7f0faf3`. This is **not** the ZIP SHA or JSON byte SHA. Do not pass the ZIP/JSON bytes to `verify_s3t_prereg_bytes` and mistake the expected mismatch for artifact tampering. The commitment serialization/preimage must be independently identified and reproduced.
- JSON `questions` is the integer 64, **not** an array of 64 original prompt strings. It contains 64 `question_sha256` values and 64 `query_char_offsets` records, but does not by itself supply the literal texts needed for independent retokenization.
- No gold labels or candidate scores were inspected. No model inference or training was performed.

**Fail-closed result:** the available physical S3T artifact establishes provenance of the preregistration manifest, not independently recoverable literal prompt texts or native tokenizer offsets. S3U must not be marked attested. Next read-only action: locate the exact pinned prompt-text producer/source commit and canonical commitment preimage; independently recompute 64 prompt SHA and verify tokenizer identity before CPU capture. Preserve `output_gate=0` and `candidate_accepted=false`.
