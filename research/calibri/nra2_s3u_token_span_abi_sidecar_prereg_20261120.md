# S3U token-span ABI evidence sidecar — research-only preregistration

Status: design-only, no execution, no candidate evaluation. Parent: S3U commit `412f9d396386848e5e28c8f48b40f4912175b319`, run `44256143791`. S3T commitment: `034ebd6e1cf6f20ad04ccd00e70c6bc07ee08a46f47da843121524f1d7f0faf3`. Gate: **0**; accepted: **false**.

## Observed reproducibility gap
In `research/nra2_bridge/stage3u_newgrammar64_real_cpu_capture.py`, native `build_query_text_mask` produces `query_mask`, `anchor_mask`, and `node_order`, and per-mention `token_positions` are validated. The saved full-token safetensors contains only `token_hidden320`, `token_mask`, and `token_lengths`. The JSON audit records `native_tokenizer_S3O_query_alignment_verified: true` but not those exact masks/positions. Therefore independent mask-level reinspection cannot be completed from the current files alone.

## Proposed additive, fail-closed evidence
Only after the original 616 J7 digest check and successful S3T physical precommit/prompt checks, emit a separate `NRA2_S3U_QUERY_ABI_EVIDENCE.safetensors` with immutable, CPU tensors:
- `query_mask`: bool [64,T]
- `anchor_mask`: bool [64,2,T], indexed by text mention order
- `node_order`: int64 [64,2]
- `mention_token_start_end`: int64 [64,2,2], end-exclusive; reject non-contiguous spans rather than silently representing them as ranges
- `token_lengths`: int64 [64], matching full-token artifact

A separate canonical JSON manifest must bind SHA256 of the exact S3T ZIP, inner prereg JSON, prereg commitment, all 64 prompt SHA256 in order, original S3U token FP16 physical SHA256, tokenizer source revision, S3O mask builder source revision, sidecar physical SHA256, shape/dtype and the fixed threshold 0.5. Do not hash a manifest that includes its own hash. Validate reload by exact tensor equality and verify each anchor is nonempty, disjoint, inside query and nonpadding, matches both literal mention offsets, and agrees with node order. Fail closed on any mismatch; never use gold, logits, or score to repair masks.

## Scope and acceptance boundary
This document changes no model code, checkpoint, training, inference, threshold, benchmark, or protected baseline. A future implementation must remain on an isolated research branch with a draft PR, no writes to LIVE/Render/Supabase/APK/main/QA/TEST, and `output_gate=0`. Sidecar existence alone is **not** an independent native CPU rerun, J7 re-attestation, S3V result, sealed blind acceptance, native full+EOS gain, <=0.7 bpp, or >=90% CB59 CPU.
