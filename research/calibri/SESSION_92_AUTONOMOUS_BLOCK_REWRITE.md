# CALIBRI quality doctor v92 — modular self-correction architecture

## Three design principles

1. **Facades at every level.** Top-level quality_doctor -> evidence -> diagnosis -> candidate selection -> full block rewrite -> physical verification -> disjoint quality/speed verification -> staging. Child modules may each split into yet smaller responsibility-specific facades. Future Jessica adapter is thin; research and production separate.
2. **Automatic diagnosis, not manual error hunts.** Observe KL/NLL, repeated tokens, signed class margins and layer drift; verify immutable public TRAIN provenance; rank local error AND causal effect. Select one bounded trial and validate on distinct TRAIN_VALID. Output machine-readable evidence, reason, SHA, original and candidate quality, and undo/rollback. No implicit approval for production modification.
3. **Entire source block rewrite before spending spare bits.** First test physically same-budget 5bit K32V8 assignment+codebook replacement using original teacher weights and TRAIN-only input energy. If that cannot recover quality, test whole alternative operator families (structured tensor, sparse, low-rank, input-conditioned native kernel) with the same per-block cost before buying small residuals. Source remains block-accessible and native-executable.

## v91 established evidence

- v91 CI #41727705684 PASS after fixing a report-print bug. Earlier failed CI #41725813439 saved its complete scientific artifact, unchanged.
- Matching real public BoolQ train slices: original FP32 teacher 21/32 and 21/32; CB59 14/32 and 12/32; CB88 8/32 and 17/32. Weak teacher ceiling itself must be considered.
- CB88 negative Yes-minus-No margin comes from frozen 4D tensor correction: mean tensor contribution -3.019 on TRAIN and -3.414 on TRAIN_VALID versus static bias contribution only -0.047. Bias-only did not change the answer distribution.
- Greedy distinct ratio on three new TRAIN diagnostic prompts: CB59 .0521, bias-only .0521, tensor-only .1042, CB88 .1146; NONE meet .25 target. CB59 also collapses before correction.
- Largest local errors cluster in attention v_proj layers 12–19; a temporary original FP32 operator at model.layers.13.self_attn.v_proj gave +1.520 nats teacher-KL gain over CB59 on two short TRAIN probes. One FP32 head or embedding swap was worse. Causal evidence is from only two short examples and is **not** yet enough to assert general importance.
- Therefore autonomous v92 first ranks and selects layer 13 attention value projection as a whole compressed-block rewrite candidate (not as a deployed fix).

## Stable facades and data contracts

quality_doctor.evaluate(evidence_path, model_sha, policy) returns immutable evidence fingerprints, ranked root-cause incidents, selected action, physical candidate checksum, paired TRAIN_VALID metrics, and decision reason.
- evidence.py: verify exact input artifact and source fingerprints; disallow prior official independent TEST as candidate selector.
- policy.py: no hidden archive overhead, speed >=90% CB59, diverse greedy >=.25, no quality regression, no unsigned production actions or silent automatic merges.
- features.py: capture 12 original model TRAIN and 12 nonoverlapping train VALID activations from public BoolQ train split only.
- encode.py: retune all 5bit code indices + shared 8D centroids with activation-weighted loss; keep native ABI and exact segment size.
- archive.py: replace exactly one stored compressed block and recalculate whole-source CRC32/SHA256, asserting all other weights identical.
- validate.py: measure paired teacher KL, constrained QA, generation diversity and native one-block speed against original source. True language NLL and full-model speed are currently unmeasured, therefore automatic promotion BLOCKED.
- pipeline.py: bounded one-candidate orchestration; no source mutation, no frozen TEST, no Jessica deploy.

State machine: OBSERVE -> VERIFY -> DIAGNOSE -> RANK -> REWRITE ONE FULL BLOCK IN SANDBOX -> VERIFY ARCHIVE -> VALIDATE DISJOINT -> REJECT or STAGE FOR RESEARCH REVIEW -> STOP.

## Decision principles for later operator design

- If teacher is also unreliable on QA: adjust only TRAIN prompts, task/model selection and evaluation ceiling before claiming compression repair.
- If tensor creates No class collapse: redesign/replace whole output correction family; do not spend bits on global biased residual.
- If a small number of v_proj blocks are responsible: retune their entire existing codebooks with no extra physical bytes; test alternate native operators for the full block if this does not work.
- If errors are distributed: build several small specialized facades with global VALID interaction checks, rather than correcting blindly by local RMSE.
- If greedy remains broken: trace attention values, KV cache, logit entropy/top-1 margin and tied embedding/head; do not hide collapse using decoding penalties.
- If archive expands: require explicit benefit per paid bit and recoup any added cost from insensitive blocks; entire model <=0.700 bpp with metadata and checksums.

## Guardrails for future Jessica embedding

The analyzer can run regularly or upon quality alerts, open a research PR, collect public benchmark evidence, and suggest/verify repair candidates. It must not silently modify Jessica live weights, repository main, Render services, Supabase, production prompts or user data. A later integration may support isolated canary deployment and automatic rollback **after explicit user authorization** and auditable release gates. User-private prompts must not be copied into unprotected training artifacts.

v92 is the FIRST prototype of a self-repair mechanism, not yet a proven self-healing model. Failed candidates remain as evidence. The previously seen v89 official independent TEST cannot be reused to train/select; any eventual frozen candidate needs a completely unused new independent TEST.
