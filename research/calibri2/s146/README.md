# CALIBRI S146 — first real pinned S142 Neural Skill trial

Goal: move S145 beyond synthetic models by integrating a new rank-8 Neural
Skill (hidden 320; two layers 3 and 9) into the **actual** physically pinned
S142 43.26M model without changing S142/S101/S82 weights or touching LIVE.

## Experiment
- Deterministic original authorized TRAIN only: the first precommitted
  S93-eligible left/right color binding pair from original S122.
- True Apple MPS, two gradient steps through original S101 KV prefill,
  subsequent-token controllers and EOS, plus real S142 additive route.
- Prove nonzero up gradients on the first step and down/gate gradients
  after up moves (zero-output Skill initialization preserves parent).
- Store one actual FP16 safetensors for Skill weights and a runtime manifest.
  Reload on CPU using the original physical S142 compressed CB61 and Q2.
- Independently generate all 456 original +128 S122 TRAIN answers including
  EOS and protect every previously correct full and first-token identity.
- Benchmark 24 paired alternating full CPU generations with and without
  the same physically restored Skill (>=0.9 speed guard).
- Count physically persisted Skill **including routing gate and manifest**
  in top-level all-in budget <=0.7 bits per parameter.
- If there are ANY protected answer losses, no new full/pair gain, budget
  violation or speed regression, roll back new neural block and Adam state;
  preserve all negative evidence and **never promote**.
- No QA, no retired S118, no zero-cost fake parameters, no hidden answer
  dictionary, no tool or production writes.

## Source provenance
S142 Actions 43078895081 (real J7 TRAIN gain 361→362/456, 17/128 S122)
and pinned physical SHA256 5a706d9012a61f24678929a2a6ee20f77c922e7e0f4bb516c061f32742cc56c7.
S144 unsuccessful continuation is NOT the parent. All original S142 files
are strictly read-only and digest checked. Single experimental Skill
training does not claim retained independence or generalization.

This PR remains DRAFT and research-only, based on the S145 branch, with
no main/LIVE Jessica/Render/Supabase/APK changes. No Jessica chat testing.

Even an accepted TRAIN-stage development checkpoint must undergo a **new,
preregistered independent held-out QA** before scientific quality claims.
The overall 135M roadmap remains a capacity objective, not a reported
training milestone.
