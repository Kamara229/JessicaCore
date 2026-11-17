# OSCAR — Open Source Compression Architecture Research (v0.1)

**OSCAR** is an independent, reusable **FP32 → physically packed quantized-transformer converter**. It is a separate algorithm/library (`research/oscar/`), not a renaming of CALIBRI, NRA, or a neural architecture. CALIBRI is a supported **consumer architecture** and validation harness, including its real 12-layer Green Tunnel and neural mechanisms.

## What OSCAR actually does

```
                 FP32 transformer weights (source)
                               │
               validate tensor geometry, tied aliases
                               │
         ┌─────────────────────┴───────────────────┐
         │                                         │
 OSCAR / cold                                OSCAR / inherited
 no source model prior                       a physically trained
 deterministic from-scratch                  5bit K32V8 dictionary prior
 source-only K32V8 codebook fit              (existing CALIBRI or future prior)
         │                                         │
         └──────────────┬──────────────────────────┘
                       │
          nearest code indices from ACTUAL FP32
        (not copied from original packed index stream)
        [32 entries ×8 weights; 5 bits per 8-vector;
          FP16 centroids & FP16 row scales; 1D FP32]
                       │
       write and PHYSICALLY reload CB59 archive
       aliases, CRC32, SHA256, original tensor names
                       │
            optional CB61 zlib runtime packaging
                       │
        architecture adapter installs matching REAL
        neural sidecars (Green Tunnel, routers, head)
                       │
            paired real FP32 vs compressed output,
            per-layer cosines, first/full/EOS, TRAIN
                       │
   all-in disk bpp + conservative effective-bit accounting
         -> accept/reject; never infer quality from MSE
```

### Why we need this split

S7A **proved that refitting source-only generic codebooks breaks** the previously learned 43M outputs even when all Green Tunnel hooks are retained: 0/8 previously exposed TRAIN examples matched full FP32 vs 8/8 original. S7B proved that **inverting the same-source FP32 mirror through the original trained codebooks** gives 8/8 real outputs, 4,546,560/4,546,560 group indices and zero hidden/logit drift on that original model. S2–S6 found no restoration of **independent** 135M original FP32 after naive uniform compression.

**Scientific limitation:** the working S142 FP32 mirror is reconstructed *from CALIBRI itself*, not an independently pretrained 43M checkpoint. S7B is a meaningful reproducible codec roundtrip and dictionary re-use sanity test, **not a universal pretrained FP32 converter**. Oscar's cold mode works on *arbitrary compatible* FP32 state dictionaries as a physical conversion algorithm, but DOES NOT guarantee preserved language ability; this is measured by a separate teacher/student evaluation before promotion. An unrelated 30×576 135M cannot be “reshaped” into a 12×320 43M; that requires independent teacher-student distillation.

## Implementation

- `research/oscar/codec.py`: standalone library `OscarConverter.convert(..., mode="cold"|"inherited")`; accepts original FP32 state; constructs/reuses all 2D codebook dictionaries and assigns indices **solely by FP32 8-vectors** with bounded memory, preserving tied aliases and original native-C-compatible K32V8 bits. Physical source integrity checked by independent CB59 parser; inherited mode reuses previously trained dictionaries but NOT historical indices. Reports 5bit code group count, normalized FP32 reconstruction error, true source disk bytes.
- `research/oscar/pack.py`: standalone CB61 streaming format, SHA/CRC verification and exact all-in physical size calculator with collision/duplicate protection.
- `research/oscar/test_codec.py`: independent synthetic CPU unit tests, tampering checks and **nontrivial changed-FP32** test, no model weights/QA needed.
- `research/oscar/experiments/calibri43.py`: separate actual S142 43.26M input adapter. For every 12-layer GT-affected original TRAIN sample, original FP32 mirror, working compressed reference, OSCAR cold, OSCAR inherited are reconstructed independently and compared on real full token/EOS and hidden states. Correct inherited output is mandatory; cold losses are recorded as negative results. Green Tunnel (layers 3/9), all 12 recurrent hooks, 48 active learned S142 adapter channels, routing S95, profile S94, S98 copier all authentic existing original physical weights. The original archive+sidecars are pinned SHA; this is not training.
- CI `.github/workflows/oscar-v01-physical-43m-fp32.yml`: real authentic CPU checkpoint, actual full 12-layer forward and recovered output, physical archive artifact.

## Acceptance gates (all separate)

1. **Reproducible codec:** inherited S142 same-source 43M 8/8 full answer + EOS, all 85 operator matrices and tied embedding/head. Must not use source packed indices in forward assignment; test can only compare index streams after construction.
2. **Functional pretrained FP32 transfer:** an **independent** FP32 43M source must demonstrate accuracy on new untouched QA before claiming transferable weights. Not done by v0.1.
3. **Physical rate:** every saved core+all inference sidecars+router+bank+metadata included once. Original 43.26M includes ~6.88M deterministic/zero-initialized structural adapter parameters; report **both** nominal bpp and conservative **36.38M-base-only bpp**. Do NOT call a ≤0.7 nominal result true ≤0.7 on the conservative denominator.
4. **Speed:** native C paired CPU ≥90% only when actually measured. Torch CPU or passing CI does not prove this.
5. **Language and training:** future independent teacher→Oscar model must use TRAIN-only gradients and replay against existing correct answers, no unseen QA leakage. Incremental optimizer is a future version, not pretended here.

## Safety
Research-only in a separate branch. No edits to LIVE Jessica, Render, Supabase, APK, main, parallel CALIBRI or NRA. No model deployment. No testing via ordinary Jessica chat.
