# CALIBRI S95 — Train a prompt-only neural selector for actual S94 physical profile bank

S94 empirical result (GitHub Actions run **42697267843**) demonstrates **all three lossless S87/S91/S93 compressed core+Green Tunnel weight profiles fit 0.7 bpp**: 0.694607877631 physical all-inclusive bpp, +73,009 bytes for the two exact alternative weight profiles. Physical switched CPU generation matched frozen S87 16/16, S91 9/9, S93 13/13 actual full/EOS tokens. 287/456 remains only the TRAIN oracle union across three **different** models, not a real combined model accuracy.

S95 tests a genuinely learned, small *prompt-only* neural router:
- 512 stable hashed token n-gram/character features, rank-12 hidden layer, trained with real **456 TRAIN only**; no lookup from prompt/ID/SHA to answer and no QA access.
- Numeric FP16 weights must be counted on top of the entire S94 source, full sidecar, exact alternative weight bank and all physical manifests (<=0.7 bpp).
- Teacher chooses which of S87/S91/S93 had a correct full/EOS answer or protected first-token on immutable TRAIN. This is supervised routing, and is measured explicitly as a TRAIN result, not new generalization.
- Fourfold held-out-TRAIN routing evaluation separately estimates feasibility; no previously inspected sealed QA/TEST is accessed.
- The fully serialized router is reloaded; exactly ONE 36M physical model is reconstructed. At each prompt the gate picks only one S87/S91/S93 profile, its exact core codebooks+Green Tunnel state are swapped from S94 bank, and actual CPU generates the full response including EOS. All 456 outputs must match the corresponding independently frozen physical reference for the selected profile.
- Hard gate for no forgetting: **zero losses on all old 282 strict full/EOS and first-token union identities**, at least one new full+EOS beyond the union, all bytes <=0.7 bpp. Physical CPU speed vs S82 and untouched sealed evaluation must be tested before any acceptance.
- Zero claims of GPU retraining in this phase; S87/S91/S93 weights were already trained jointly on real Apple MPS. S95 trains only the small router on CPU, with no changes to any source model.

Even if some TRAIN prediction is improved, the result must remain `candidate_accepted=false` without a true independently sealed QA and full CPU speed certification. All GitHub actions run in research branches; LIVE Jessica, Render, Supabase, main and APK are left untouched.
