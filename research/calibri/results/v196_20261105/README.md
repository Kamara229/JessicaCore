# CALIBRI v196 — Two Green Tunnel cycles: **physically packed PASS, whole-answer quality REGRESSION / FAIL**

[Successful EXECUTION GitHub Actions #42104283667](https://github.com/Kamara229/JessicaCore/actions/runs/42104283667), [complete actual physical 135M source and all 24+16 native QA/VALID/training artifacts #13451128516](https://github.com/Kamara229/JessicaCore/actions/runs/42104283667/artifacts/13451128516).

IMPORTANT: **GitHub green SUCCESS is not model success**. The whole-vocabulary, all-layer recovery hypothesis performed WORSE on semantic QA than the previous real v186 saved source.

| Actual original 135M source | Original FP32 | Baseline CB59 | Previously trained v186 | Green Tunnel v196 |
|---|---:|---:|---:|---:|
| Old 24 PUBLIC DEV machine-confirmed correct | **6** | 0 | **2** | **1** |
| Newer precommitted v193 16 PUBLIC DEV machine-confirmed | **6** | 0 | **2** | **0** |
| Old 12 heldout opposing VALID pairs: both first tokens correct | 0 | 0 | **3** | **0** |
| New 16 QA machine-wrong | 6 | 0 | 2 | 4 |
| New 16 QA UNGRADED | 4 | 16 | 12 | 12 |
| CPU throughput vs original packed CB59 | — | 100% | previous 99.59% | **100.17%** |

Actual complete GT196 source **11,229,303 bytes = 0.6678394131307638 bpp**; SHA256 **8e284672755c46c9cb874d9b5273aa5ad97bd7b9b589de414527a92711249b74**. Actual native source was sealed before public DEV or heldout VALID responses. No unpaid runtime dense FP32 source. 211 physically paid codebooks received whole-model gradients in each of **192 actual backward passes**, across two fixed restoration cycles, with **49,152 output vocabulary teacher distribution** at sampled TRAIN positions (NOT every word literally included). Physical source changed **69,283** original 5-bit group assignments and **53,653** FP16 centroid cells; 73,216 paid index modifications accepted during training. Syntactic native source/optimizer numerical consistency passed.

**Catastrophic source-model response collapse** despite correct runtime and source integrity: all 16 newer real QA responses from GT196 generated mostly one word, **red**, or repetitions such as **redredred**. Most generated at most a few red tokens, then emitted EOS. All 16 native original-CB59 responses were repetitive 'speci'; old v186 retained sometimes meaningful Yes/No and 2/16 confirmed, while GT196 lost those distinctions. In 12 previous heldout contrasting VALID pairs the **GT196 first-token vocabulary collapsed to exactly ONE token ID** and 0/12 both correct; the earlier trained v186 correctly differentiated 3/12 pairs.

Training objective did decrease for some TRAIN samples, but that did not restore knowledge. This is the same danger seen in v176, which had 128 gradient steps but collapsed to a constant 'N'. More cycles or better TRAIN losses are **NOT equivalent to restored native language**.

## Scientific diagnosis, not speculation as measurement

Measured:
- Whole-network gradients, updated ALL paid source tensors and 96-target physical index swaps DO work technically.
- The same original full-size 135M model remains rate-valid and native speed valid after training.
- The chosen combined corpus/teacher/contrasting TRAIN objective **damaged** v186 existing partial behavior.
- Before choosing another set of recovery cycles, the method must include constraints against destructive behavior and protect the source's original function, as well as physically improve representational capacity in high-sensitivity fields.

Hypotheses to TEST next (not proven):
- Strong global source/teacher mismatch and highly limited K32/V8 representational capacity induce arbitrary high-probability vocabulary-token collapse.
- Tied embedding/output head and interaction between 211 source matrices require per-family trust regions, learning-rate scaling and TRAIN-only recovery-behavior anti-collapse guards.
- A balanced 211-operator per-cycle physical index-update schedule, plus more diverse public instruction/plain text and an explicit budget allocator for the **remaining original 540,760 paid bytes up to 0.7 bpp**, may preserve real original model capabilities better than retraining only same 211 codebooks.

## Mandatory research gate from owner

**Do NOT proceed to any 360M or larger model** until physically native CALIBRI 135M preserves at least original FP32 answer correctness and diversity on truly independent, sealed questions. 2/24 and 3/12 old v186 is partial, not success. Repeated 16/24 historical DEV is diagnostic, not final unseen confirmation. Need matched real full answer evaluation, both-correct opposing pairs, at most FP32 ungraded cases, <=0.7bpp, >=90% packed native CPU speed. v197 engine/gate work isolated in draft PR #336.

Every negative result is preserved; no v89 private TEST access, main merge, LIVE Jessica, Render, Supabase or APK, or Jessica-chat tests.
