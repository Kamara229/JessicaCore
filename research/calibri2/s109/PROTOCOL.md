# CALIBRI S109 — train ONLY 20 shared Green Tunnel gains with two frozen donor blocks

**Research / not a deployed or accepted model.** No modifications to `main`, LIVE Jessica, Render, Supabase or APK; no Jessica chat tests; no previously inspected QA or blind TEST.

## Why this experiment

S103/S104 demonstrated a real 12-layer shared-GT bridge with only 10 trained gain values (one shared GT3 donor), 456/456 identical initial outputs, later **353/456** exact full+EOS and 387 correct first tokens unchanged, ~0.69964 bpp and 0.9929x CPU ratio vs S101. But S104 gained **zero** new answers.

S105 measured opposing gradients in **46 of 96** wrong/protected pairings and the strict local projection from that sampled objective yielded zero direction. That does *not* prove that training with different representations is impossible. S108 reuses both EXISTING, already paid low-rank internal GT3/GT9 blocks and adds one zero-initialized gain for each original block at each of ten extra decoder boundaries: **20 FP32 scalars, 80 raw bytes plus actual sidecar/config overhead**.

## S109 experiment

Run two REAL Apple MPS gradient steps **only after** S108 true 456 TRAIN physical CPU zero-gain parity succeeds (#42877139218). Any stage-A failure blocks training. Use S101 original compressed 36M CB59/CB61 core, original GT3/GT9 weights and controllers; S87/S91 profile bank, S95 router and S98 copy policy frozen. Train only the 2×10 gain matrix for **S93-routed** examples on authorized TRAIN: two incorrect and four previously correct examples per step; full response+EOS teacher-forced *surrogate* loss with S82 first/later/pointer components, not the truly non-differentiable S98 policy. Adam rate 0.004, gradient clipping, gain clamp 0.25.

Physically save and reload proposed original S93 sidecar with one new 20-value `dual_bridge.layer_gains` tensor, plus deployment config. Count every physical byte in all-inclusive bpp. Rebuild original S101 cached CPU runtime with exact S95/S98 policy and check **all 456 original TRAIN questions** for:
- ZERO lost among all 353 correct full answers including EOS;
- ZERO lost correct first tokens among original 387;
- all-inclusive physical bpp <=0.700000;
- CPU same-host >=0.90 times the original unhooked S101 runtime;
- exact original CB61 / frozen controllers and profile alternatives.

If any guard fails, revert entire gain matrix **and full Adam moment/state**. A successful TRAIN retention transaction is NOT independent QA and NOT acceptance. A meaningful research improvement requires >=1 gained full answer with zero losses and later new precommitted held-out QA, canonical S82 CPU ratio and CPU/MPS parity.

## Known limitations

S108 zero-gain architecture feasibility is not a quality improvement; S109 is an exploratory pilot. Both donors share weights across layers whose activation distributions may differ. The original S104 and proposed S109 objectives are only differentiable surrogates for frozen S101 actual autoregressive S98 decisions; actual serialized inference alone establishes correctness. Running an optimizer does not imply a valid improvement; do not claim one without CI outputs.
