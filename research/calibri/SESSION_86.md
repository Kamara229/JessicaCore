# CALIBRI v86 — genuine unseen independent test of frozen v85 adaptive native operators

**Preregistered before seeing ANY v86 TEST.** New stacked draft PR. All source weights/controls and source SHA are frozen from actual completed v85 run **41578432949**. There is no v86 training, gate tuning, checkpoint selection or weight source redesign.

## Earlier real v85 independent TRAIN-VALID evidence, not yet TEST
Completed real CPU original pretrained 135M v85 41578432949:
- Actual fully physically packaged source **11,240,769 bytes =0.6685213296 bits/original-parameter**, all full CB59 trained 2D original weights, headers, CRC/SHA and 3 extra changed 32x8 FP16 per-operator codebooks, 1536-element FP16 router weights and biases; physical SHA `2fe36504a0eaf0159859cff68ee1ac3269c55959912f3b0ea1b17bbe1d9c7f5d`.
- Original packed CB59 baseline on TRAIN-VALID Wiki KL **11.9761546**, News KL **11.3410663** (mean 11.6586105). New **all-three activation-routed** CB85 Wiki KL **11.6025740**, News KL **11.1308238** (mean 11.3666989), improved in BOTH domains (−0.37358/−0.21024 nats).
- Original next-token VALID NLL: Wiki **15.46073**→**15.07154**, News **14.46838**→**14.30905**. This is only **63 true positions per corpus**, and success is ONLY a TRAIN-VALID gate; do NOT claim unseen generalization or restored good language quality. Original FP32 NLL ~4.3–4.4, so large quality gap remains.
- Actual adaptive runtime branches are both native C with a Torch CPU gate; source physical all accounted, but not yet fused single C operator or demonstrated faster than baseline.

## Predeclared new genuinely disjoint TEST
1. Original SciQ official `test[320:344]`, **24 original questions** with length-normalized full-answer teacher-forced likelihood and support limited to 320 tokens; prior v82 `test[160:240]`.
2. Original BoolQ official `validation[3140:3164]`, **24 original questions** with a truncated 256-token passage/Yes-No task; prior v82 `validation[2990:3070]`.
3. Original WikiText official `test` tokenizer token range **[160000:160512)**, **8×64** contexts/504 actual next tokens; prior v82 `[140000:141024)`.
4. Original AG News official `test[7520:7600]` first **512** tokenizer tokens, **8×64** contexts/504 positions; v82 `test[7410:7510]`. Token and dataset revision SHA recorded.
5. Two never-tested prompts and **16 genuine greedy KV-cached tokens each**: `The new method succeeded because`; `A small village near the mountains had`. Record distinct-token ratio, top-token share and real post-prefill native CPU tokens/s, not semantic correctness.

All three actual real original 135M models run independently on exact same TEST: FP32 teacher, full CB59 original packed C, full frozen CB85 adaptive C×2+Torch gate. Use existing prereviewed v78 original full-response scoring but override ONLY the preregistered source test spans and prompts. No source training and no adaptive source selection after TEST.

## Research-quality criteria declared in advance, not automatic CI success
- SciQ CB85 correct count >= original CB59, BoolQ CB85 count >= CB59.
- Wiki and AG News full-vocabulary teacher KL improve on new TEST (strict <0); next-token NLL no worse on both.
- CB85 greedy distinct-token ratio >=0.25 (repeat-collapse not solved otherwise).
- Complete physically accounted model source <=0.700 bits/original trained parameter; original pretrained source checksum unchanged.
A failure of any research-quality condition falsifies broad quality-success claims even when GitHub workflow exits successfully. Independent QA subsets small and may have large noise; exact paired McNemar and 5000 paired bootstraps included.

## Facade structure
```text
frozen_adaptive_v86.py         thin CLI
└── v86/                       top facade
    ├── __init__.py
    ├── pipeline.py            compose only frozen real model evaluation
    ├── provenance.py          immutable source SHA and fixed new TEST ranges
    └── stats.py               pure exact paired question statistics
test_v86_frozen_adaptive.py    pure unit test
```

No unmerged PR touched `main`, live Jessica, Render, Supabase or APK. v86 does not train or choose weights after opening independent TEST.
