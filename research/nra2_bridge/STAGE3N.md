# NRA2 S3N: true frozen 43M CPU capture for semantic-role S3M pairs

S3M PR #628, CI #43521067551 created 128 TRAIN + 64 DEV
role-counterbalanced directed graph prompts from 48 topology-disjoint
graphs. No QA/TEST or blind final set was used.

This S3N research-only follow-up extends the successful S3H original
frozen CPU prefill hook to all **192 S3M prompts**. It replays the exact
S194 original 616 generated CPU J7 source and verifies 456/456 original
plus 128/128 new protected complete responses and exact original SHA,
all while output gate is zero and Calibri/Green Tunnel/NRA are frozen.
S3N does **not** train any adapter or score DEV64.

Expected physical artifacts:
- `NRA2_S3N_REAL_CPU_GRAPH192_ALL_TOKENS_FP16.safetensors`:
  actual 43M layer12 [192,T,320], token lengths/masks.
- `NRA2_S3N_REAL_43M_GRAPH192_HIDDEN320_FP16.safetensors`:
  true final token per prompt.
- `NRA2_S3N_GRAPH192_STRUCTURAL_LABELS.safetensors`:
  directed edges, source, target and independent reachability gold
  generated **after** frozen native CPU work and original J7.
- `NRA2_S3N_192_ROLEBALANCED_REAL_CPU_AUDIT.json`:
  SHA provenance, native J7 protection, all physical reload bits,
  disjoint prompt SHA indices by TRAIN128/DEV64 and frozen-gate
  non-acceptance.

Original S3E #43512049506, S3H #43517070838 and S3M
#43521067551 are pinned prerequisites. S3N CI success establishes
only genuine token acquisition, NOT generalization.

**Next S3O:** train language-conditioned semantic *source/destination*
selection from real node mention spans and contextual query words;
retain the S3K structure-scorer as separately monitored baseline.
Only TRAIN128 has an optimizer. Lock actual FP16 model and TRAIN-only
normalizer before once-only DEV64 research readout. DEV isn't blind,
and independently preregistered unseen final work and real new native
decoder full+EOS gains are required before candidate acceptance.
No LIVE Jessica, Render, Supabase, APK, main, QA/TEST or
Calibri/Green Tunnel updates.
