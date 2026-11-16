# CALIBRI 2.0 independent pretrained 135M — S1 four real training cycles

**Research-only follow-up to S0. Not a release, not proof of FP32 parity.**

Frozen input: S0 independently recoded pretrained FP32 135M checkpoint at SHA256
\`50eac24ba7d7a452c495f8adced2cdef3f9cc44e024217fc21d8f2cebb87413c\`,
11,229,303 physical bytes = 0.6678394131307638 bpp, 211 paid CB59 matrices,
original tied embedding/head, no off-model inference weights.

### Precommitted four-cycle TRAIN protocol

- Fixed 4 cycles x 8 optimizer steps = **32 real end-to-end QAT updates** through all 211 compressed original operators; every step trains existing FP16 centroids, **not** dense inference weights. Rehearse four distinct protected TRAIN examples during each cycle. 
- Frozen 8 FIT + 4 REPLAY tasks are simple direct answers with explicit text and EOS. Evaluate both literal generated full answers and teacher-forced token CE; choose checkpoints from **TRAIN only**, never VALID/TEST/EVAL.
- Once per cycle attempt up to 64 **real 5-bit code index** swaps in a predetermined attention/MLP operator, proposed by an authentic QAT weight gradient with no original FP32 weights in memory. Keep index updates only if measured TRAIN loss improves and rehearsal-loss does not deteriorate.
- At every cycle take an optimizer + all-211-centroid + every stored-code-index transactional snapshot. Reject and **fully roll back** the entire cycle if average FIT TRAIN CE does not improve, any previously correct full-answer TRAIN output is lost, or replay TRAIN CE rises by >0.25 nats vs accepted state.
- Serialize each accepted checkpoint in the same complete packed \`CB59\` physical format; verify SHA256, unique parameter denominator, aliases, all 211 matrices and \`<=0.700 bpp\`. Keep final accepted archive AND report all rejected steps.
- Log direct baseline and each cycle's exact correct TRAIN answer count, first target token matched, loss, changed code groups, changed FP16 dictionary cells, and 3-way batched branch behavior. **Do not infer generalization from TRAIN**.
- Use the original pinned model \`HuggingFaceTB/SmolLM2-135M-Instruct\` only for architecture/config/tokenizer. This stage loads only the already compressed S0 archive, **not FP32 weights**. FP32 outputs are not served online at inference.
- Evaluate new untouched QA **after** training, in a separate future stage, including full answers and EOS, full-source CPU throughput, accuracy across domains and confidence intervals. Don't re-use v231/v232 QA for selection.

Strict exclusions: no ordinary Jessica testing, no LIVE, main, Render, Supabase, APK, or unrelated CALIBRI/NRA branches. Technical green CI does **not** indicate language parity.
