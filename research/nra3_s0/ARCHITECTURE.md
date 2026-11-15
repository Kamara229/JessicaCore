# NRA2 3.0 — S0 unified neural-area / Calibri mid-layer ABI (research-only)

**Status:** implementation-level scaffold and offline toy decoder tests only. No training, no physical Calibri 43M J7 comparison, no production activation. Does not touch S3CI's reserved sixth EVAL128, nor any of the first five consumed EVAL128. `output_gate=0` and `candidate_accepted=false` are hard requirements.

## Original problem is preserved

S194's frozen 43M CPU baseline retained 456+128 old full answers but obtained 0/32 complete novel answers (16/32 first-token correct). NRA's purpose is **structural transfer and generalization into full generated answers**, not tagging words as an end in itself. S195 used 21,900 learned semantic-slot weights connected to all 12 layers but did not demonstrate generalization or the required inclusive bpp.

## Single area, three *internal* computational stages

- Input: Calibri decoder hidden `[B,T,320]` at index 7 (end of eighth transformer layer).
- **Grammar (4 causal blocks, width 256):** contextual morphology, UPOS classifier, causal dependency head scores. Must learn Russian features from **labeled TRAIN data**; raw weights do not parse Russian. For offline full-sentence UD, later words can govern earlier ones; these *causal* scores only express the current prefix hypothesis, and predictions can be revised on subsequent tokens.
- **Semantics (3 causal blocks, width 256):** role logits, candidate antecedent scores, learned phrase context. Grammatical subject is not automatically semantic origin/agent. Missing coreference must abstain rather than infer from metadata.
- **Relations (128-wide GRU, six shared propagation steps):** learned directed, causal attention over *past* contextual states. No graph gold or reachability oracle in inference. Event/state tracking is not persistent across decoder calls yet.
- **Output:** 320-wide *zero-initialized readout*, not used by Calibri yet. The read hook returns the **exact original object** to preserve all generation outputs bitwise.

`S3CK` is an independent gold-free **annotation ABI** useful for future TRAIN target construction and audits; it is NOT a separate runtime grammatical adapter. The learning target heads live within one `NRA3NeuralArea` module.

## Integration: intentionally read-only

`NRA3MidLayerPort` installs exactly one forward hook after layer 8 of 12; no all-12 hooks, no modifications to Calibri, Green Tunnel or tokenizer. Running on `[1,T,320]` unpadded embeddings only (padded batches lack a mask in this hook). The input may be a single cached-decode chunk; this prototype does **not** maintain causal semantic history across chunks and **must never activate output gating**. Later design must implement a replay-equivalent incremental semantic cache whose memory is scoped to a single request/beam, invalidated on reset and compatible with native CPU KV caching.

## Capacity and requirements

The default stage dimensions are a **capacity hypothesis**, not a validated sufficient Russian-language model. Parameter count is derived from actual module tensors and checked in tests to be between 5 and 7 million. The pretrained Calibri 43,261,760 parameters and this new area are counted separately. An FP16 lower bound alone violates inclusive `<=0.7 bpp` when the parent is near 0.69: this prototype is **not eligible for promotion**. Real compressed payload bytes, metadata, all other inference-side additions, original-CPU 616 J7 full-EOS equality, independently new examples and **CPU speed >=90% of CB59** must all be verified. Whether 4+3 attention blocks can ever meet speed is not yet known; profiling/ablation may require lighter layers.

## Next research gates (not done in S0)

1. Specify true original43M source artifacts, verify tokenizer and extract unmodified layer-8 hidden320 on permitted historical TRAIN only; audit source hashes and enforce TRAIN allowlists per optimizer minibatch (including the 2048 historical replay gap).
2. Freeze blocked family/topology splits before training; create an appropriate Russian UD + semantic role TRAIN curriculum. The old sixth EVAL is reserved exclusively for S3CI and **must not be used** for NRA3 tuning.
3. Train internal grammar, semantics, relations heads on real CPU features; record full SHA and physical FP16/quantized reload. Contrast independent semantic/generalization controls with Calibri-only and S3BA using **new** independently preregistered holdouts (not consumed sets).
4. Implement cached causal state/recomputation parity. Only then research an explicitly switched-on residual link feeding layers 9–12; verify bitwise old 616 J7, new full answers+EOS, bpp and CPU under real decoding. Any failure blocks acceptance.

No writes to LIVE, main, Render, Supabase, APK; no Jessica ordinary-chat tests or releases.