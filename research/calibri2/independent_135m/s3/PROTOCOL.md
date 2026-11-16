# S3 — real joint GT+compressed-core training against the ORIGINAL pretrained FP32 135M, without NRA

**Not a guarantee of recovery.** Real S2 #43964269982 localized extreme representation drift: last token original/CB59 hidden cosine ≈0.679 at input, ≈0.04–0.08 after 30 original decoder layers; restoring original FP32 input-only, head-only or both **did not restore coherent output**. S2 physically audited scaled GT options. The 5×local rank24 + all-30 shared rank32 + slot rank16 candidate fits at **0.696586 bpp WITHOUT the 132KB S1 negative-result profile**; including S1 FAILS (0.704453 bpp). No second source profile permitted for S3.

## Precommit

- Original pinned trained FP32 `HuggingFaceTB/SmolLM2-135M-Instruct`, 134,515,008 original unique parameters, 30×576.
- S0 original paid CB59 physical SHA `50eac24ba7d7a452c495f8adced2cdef3f9cc44e024217fc21d8f2cebb87413c`.
- True S87-style local GT **5 separate blocks rank24**, at original decoder layers 4,10,16,22,28; S126-style recurrent GT rank32 through **all 30**. Neural router and 2-slot parallel head rank16 stay frozen/shadow in the complete serialized sidecar. NO NRA, grammar linker or S3 adapters.
- All FP32 teacher representations used ONLY offline to supervise already physically compressed student during **TRAIN**. FP32 state/logits/hidden caches MUST NOT ship in inference package or affect inference.
- Fixed *pre-authored TRAIN-only* corpus = 20 examples from earlier S0/S1 + 12 novel, clearly labeled TRAIN-only prompts (not any previously scored S2 12 EVAL, not sealed QA).
- Teacher GREEDY outputs up to 8 tokens, real EOS if emitted, plus full-vocab logits at each teacher-forced target and hidden states at layers 0, 10, 20, 30. This is legal offline knowledge distillation, but **teacher agreement is a diagnostic; the target is correct independent answers**.
- Three cycles × 16 gradient steps = 48 authentic SGD/Adam updates through **all 211 original paid centroid codebooks** and the new physical local+recurrent GT. Loss = teacher output CE + soft logits KL + intermediate hidden cosine discrepancy; coefficients fixed before running. Protect the original source (no original mutations). If numeric nonfinite or missing gradients, fail closed.
- After EACH cycle score 8 genuine TRAIN-only greedy responses including EOS and compare to frozen original FP32 generated responses, and retain pre/post if no improvement. Dump full physical candidate(s) for investigation even if still unusable. Zero developer acceptance absent a genuine **full response or first-token** teacher recovery without old loss; no silent deployment of merely decreasing training loss.
- Real full artifact includes ONLY one modified CB59 source + physically saved FP16 GT/router/slot sidecar + manifest, all-in **<=0.700 bpp** including same logical 134,515,008 denominator and verified SHA. Source 1D/tied parameters and group-index decoder unchanged. FP32 teacher weight archive NEVER written.
- EVAL held out until future S4 **after** architecture decision; native-C speed, full correct QA not inferred from TRAIN CE or MPS/CPU model execution.
- No ordinary Jessica chat test, LIVE/main/Render/Supabase/APK changes, no edits to parallel CALIBRI/NRA2 development.
