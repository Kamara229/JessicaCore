# NRA2 3.0 S11 — physically accounted small 3-way relation router (NEW 72 Russian TRAIN)

**RESEARCH ONLY / DRAFT / DO NOT MERGE / zero Calibri output gate**.

S10 successfully produced original **real** frozen Calibri 43M after-GT layer8 CPU hidden320 for exactly 72 new Russian TRAIN utterances, with complete original protected **616 J7 generation digests unchanged** (CI #43732621116 SUCCESS). Physical SHA `ed262984c0fd48b60987308e7b135be7d05d3e5434fe9923f373aec178ff2b46`; actual [72,59,320] FP16 and source S9 SHA `b09d60a77a17d30c2dbf846e40ab97860def8635e2bf31a5b0ab8981832b744d`. S10 source was **read only**, no new candidate activation or weight updates. Original S8b paid NRA3 6M archive SHA `b543c5b4692caa8764382271f94ccfbddeee8b2afdb096666676b2a68e1375f9` with 43 real 5-bit CB59 matrices remains unmodified; its three-hypothesis selector collapsed on TRAIN12 and was rejected.

## New stage

S11 adds a **tiny, physically stored and authenticated** trainable router, rather than pretending static S8 hand-scored evidence is learned. Its **12→3 linear head has 39 parameters**, snapshotted to true FP16 including biases, packed with an unambiguous version header and SHA256 = **123 bytes physically paid**, on top of the unchanged 653,798-byte CB59 source. This small extra FP16 2D operator is *explicitly billed*; no claim is made that it is CB59 5-bit. The rest of the 6.1M NRA3 area still runs with all 43 previous real native-C CB59 matrices. No updated original Calibri/GT/6M weights.

The selector consumes 12 fully **causal inference-time quantities**, calculated from the three genuinely different local/global/predicted-role branch attentions and messages (candidate/query cosine agreements, causal long-range/local/role support, message energy, branch disagreement, visible token age and confidence of the model's own role prediction). No gold words, entity IDs, dependency arcs, heuristic teacher labels, future text, independent QA/EVAL, task-answer truth or extra pretrained language model enter its inference.

Only **during training**, a separately validated S9 UD corpus creates weak per-word router labels based on already-seen syntactic heads and semantically complementary previously-seen roles. These three labels are *heuristic*, not proof of an objectively correct attention route. The original genuine 43M S10 hidden, all 72 SHA-locked TRAIN identities and original complete tokenizer word alignments are independently rechecked before the optimizer. Training modifies only the explicit 39 stored FP16 values. It physically reopens its own SHA-locked source, executes the entire original native C NRA3 with **six genuinely paid three-hypothesis relation update cycles** using the new router and measures per-branch count, weak TRAIN teacher agreement and NRA-only CPU exploratory time. It does not automatically make or promote a candidate.

## Fail-closed requirements

- 72 *previously unseen to the original 12 TRAIN training steps*, but this is still new **TRAIN**, not independent EVAL.
- Retain original **616 J7** from S10 frozen source; **no new active J7 generation** with S11 candidate.
- All five previously consumed EVAL128 and the sixth sealed S3CI EVAL128 remain **untouched**.
- Do not interpret TRAIN classification/loss improvement or use of all three branches as native Russian grammar accuracy or better answer quality.
- Whole-system (Calibri 43M + GT + NRA + physically paid selector) **<=0.7 bpp, >=90% CPU** remain unverified and are mandatory before any activation.
- No main, LIVE Jessica, Render, Supabase, APK, ordinary Jessica chat tests, production outputs or model-acceptance changes. `output_gate=0`, `candidate_accepted=false` regardless of fit.

### Next gate

Evaluate genuinely withheld new Russian cases, preferably with different lexical families, sentence templates, paraphrases and directed semantic or coreference relations, preregistered before any choice of model or hyperparameters. Then do honest whole-43M+GT+NRA physical bits and CPU speed, original protection and full-EOS paired baseline. Without those, keep all gates closed.
