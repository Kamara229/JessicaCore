# NRA2 / CALIBRI S3BA — immutable research-only experiment contract

Stage S3BA is **preregistered only** in this commit. No training, model score, or new inference is claimed. Implement and verify this contract in a separate subsequent commit; do not alter LIVE Jessica, Render, Supabase, APK, main, QA/TEST, or ordinary Jessica chat.

## Locked evidence (verified successful GitHub Actions, 2026-11-14)

- S3AW baseline query-role FP16 (CI 43583928298): `8541eea24d5074e29b6a4acd8b611d2d29d608dacf2dfe909e8df05eb0fb2130`, research exploratory DEV128 = **89/128**, previously correct S3AS items unchanged. The DEV128 has **already been scored**, therefore is not independent holdout evidence.
- S3AZ original frozen 43M CPU native TRAIN512 acquisition (CI 43587917560): `978cb86c1e30c211744a800b8c37a7e20dc04e2b7ee85126dea3e91d7894feec`, tensor `[512,95,320]`; original S194 protected **616 J7 exact digests matched**, read-only `output_gate=0`, 512 S3AY TRAIN and no DEV/EVAL prefill. Source artifact: `nra2-stage3az-real43M-s3ay-newtrain512-tokens320-616J7`.
- S3AY prereg (CI 43587711528): `90ba2d0347a333dc9ff2e85adbbaca20becb4f92051c1ec43267102b63c8a118`. Exactly 64 authorized S3AQ TRAIN graph topologies × 8 new role-balanced phrasing styles, 512 TRAIN prompts; 256 source-first/256 destination-first; 256 yes/256 no.
- Previous original CPU TRAIN-only sources: S3N 128, S3Y 128, S3AR 256, S3AV/S3AU 512. **Old replay = 1024; new S3AY = 512; total = 1536.**
- Original frozen S3O graph FP16 SHA `40a8903d5419cd8641b5d01e75e429e10b449304b3ba140d2245bf7f8a9d00b6`; S3AR CPU source hash `5c44f4d52b5f2f618b93885f288975e269a4d44d21506cd7906368fd20467f2b`.

## Precommitted experiment (requires implementation)

1. Load locked S3AW query-role head, **not S3AS**. Keep S3O graph, normalizer, original 43M, NRA2 and Green Tunnel frozen and bitwise SHA-checked. The only trainable parameters are query-role head parameters.
2. Rehydrate 1536 TRAIN-only native CPU token examples with original tokenizer and physical source/target labels, validate prompt/order SHA, split provenance and no accidental DEV or future EVAL text inclusion. Must verify S3AZ exact 616 original J7 digests, frozen full physical FP16 and output gate zero *before* training.
3. Deterministic seed **2026111496**, AdamW (weight decay 0.02), learning rate **0.00005**, **160** steps, batch full 1536 TRAIN data; gradient clip 1.0. Rehearsal groups S3N128, S3Y128, S3AQ256 weighted 2 each; prior S3AU512 and fresh S3AY512 weighted 1 each. Symmetry MSE weight 0.15 for S3AQ quartets and for each 512-sample role-reversal octet. No tuning after DEV inspection.
4. Save physical FP16 candidate and verify reload and frozen graph SHA **before** viewing already-scored exploratory DEV128. Run paired comparison vs locked S3AW 89/128; report all correct, gains, regressions, and each grammar-style; include both previously-correct retention and newly-correct counts. Do not accept the model merely because DEV improves.
5. Do **not** open, CPU prefill, score, train on, or select using the truly future independent S3AQ **EVAL128**. No full-EOS improvement, all-in bpp, or CPU speed benefit is claimed until separately physically demonstrated. Candidate remains `candidate_accepted=false`, `output_gate=0.0`.

No ordinary Jessica chat test. Research-only GitHub Actions and draft PR, no LIVE modifications. If any prerequisite is not physically verifiable or any source SHA mismatches: **FAIL CLOSED** before optimizer.
