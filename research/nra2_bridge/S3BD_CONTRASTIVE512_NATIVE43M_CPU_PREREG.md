# NRA2 S3BD — frozen original 43M native-CPU capture of S3BC TRAIN512 (preregistration only)

S3BC GitHub CI **43591378375** completed **success** on 2026-11-14. Immutable training-only 512-sample data commitment SHA256:
`65832674475452dbf4d5800ab310ee07ce2d6f5a9e237307e6ec3299e19e2d4c`.
Dataset 64 S3AQ TRAIN graph topologies x 8 independent contrastive role templates;
256 source-first and 256 destination-first; 256 yes and 256 no; 512 unique prompts.
Artifacts: `nra2-stage3bc-contrastive-role-train512-prereg-no-model` from CI
43591378375. Original S3AQ commitment
`c55799793263d5b4175fdfd62c293bcc2cb1794c73fa6ba856cf8b0a961f7839`.

This commit is ONLY a **contract, not CPU inference or any measured output**.

## Authorized follow-up read-only stage

- Reuse locked S3AZ original 43M CPU capture pipeline and its exact original
S194 616-J7 digest guards. Fail closed if any original checkpoint, tokenizer,
normalizer, graph features, Green Tunnel, J7 checksum or archived source SHA
changes. S3AZ CI 43587917560 and S3BA 43590032315 must be successful.
- Preflight reconstruct **only S3BC TRAIN512** in exact original graph/style order;
confirm hash commitment above and exact prompt hashes, query offsets, graph
topologies, 32 positive/32 negative examples for each grammar.
- Run true frozen original 43M CPU layer 12, read token-level FP16 hidden
features `[512,T,320]` and source/target token alignment, without model
updates, answer scoring, adapters, or gradient steps. Protect *all 616*
original S194 J7 input-output digests and record exact SHA256 of native
physical CPU feature artifact (including reload-equality).
- Derive structural adjacency/source/target labels **after** frozen 43M
forward and original J7 validation, never inject gold role order, oracle,
reachability or graph IDs into model inference.
- **No DEV128 or future EVAL128 CPU capture, scoring, prompting, training
or role-head comparison** in S3BD. The observed S3BB DEV128 is repeatedly
scored and cannot serve as blind evaluation. Sealed EVAL128 must remain
untouched. Preserve identical original graph/normalizer and accepted weights.
- Persist S3BD output only as immutable research artifacts, with
`output_gate=0`, `candidate_accepted=false`, immutable source/target SHA
and no LIVE/Render/Supabase/APK/main/QA/TEST effects. Ordinary Jessica chat
remains a passive relay, NEVER a test runner (HTTP429).

The next implementation commit must introduce an independently checked
S3BD script and GitHub Actions workflow; no model score, performance or
training success is claimed by this preregistration.
