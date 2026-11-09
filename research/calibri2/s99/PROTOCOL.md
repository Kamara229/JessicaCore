# CALIBRI S99 — integrate REAL learned S98 neural noun-copy into the one physical S95 model

S98 learned an actual tiny numeric owner-anchored second-token selector with Apple MPS: **192/192 TRAIN fitted**, **64/64 TRAIN heldout nouns**, **four instruction-template TRAIN folds of 48/48**. Conservative physical budget with S95 and S98: **0.698632453448 bpp**. These are conditional second-word decisions with *teacher-supplied correct first word*, not actual full generated model replies.

## Different scientific question

Can the frozen S98 numeric selector improve **real autoregressive full responses including EOS**, conditioned only on the first word the physically compressed S95 model actually generated, and preserve the **282 legacy S87/S91 correct cases and 287 S95 existing full/EOS cases**? This is not obvious: S96's earlier novel 64 had only 2/64 strict full answers (and no retraining on those inspected prompts is allowed).

## Exact empirical protocol

1. Download immutable physically trained S93 CB61, S94 neural Green Tunnel alternate dictionary/tunnel bank, S95 prompt-only numeric FP16 router and S98 *already trained* FP16 tiny neural context copier. Verify hashes / provenance. No model training in S99, no S96 prompts.
2. Rebuild one 36M CPU compressed source and its S82 original frozen controllers, with the two paid Green Tunnel internal layers. For each of the exact frozen **456 TRAIN** questions, select a single profile with the unmodified numeric S95 router and physically swap that profile's only two trained codebooks plus Green Tunnel parameters.
3. Reconstruct real **full greedy output through EOS**. At step two only, if two distinct literal second-token candidate spans follow the model's actual first token, ask S98's numeric 12-to-8-to-1 neural scorer to choose a source span. Override the model's next-token distribution only at softmax confidence >=0.95; otherwise leave it unchanged. No gold labels or precomputed TRAIN answer lookup at inference.
4. Independently re-run frozen unmodified S95 generation for the first 48 prompts, checking exact token parity to archived independent 456 reports; compare paired warm CPU timing to the new S99 inference.
5. For **all 456** physically generate the actual neural-copy model's full response including EOS and require **zero lost full/EOS answers on all 282 legacy union** and zero lost first-token skills. Also guard all **287 S95 correctly answered items** against regression, require at least one new full/EOS success and a >287 total, CPU speed >=90% and all-inclusive physical size <=0.7 bpp.
6. Save a full negative report even when a guard fails. No automatic acceptance: independent untouched QA plus native C and CPU/GPU parity remain required.

LIVE Jessica, main, APK, Render and Supabase untouched. Never use previously inspected S96 or any old QA/TEST for training, parameter selection or retesting. This is research on TRAIN only; `candidate_accepted=false` unconditionally.
