# CALIBRI v194 — original 360M genuinely packed <=0.7 bpp, **native language quality FAIL** (5 Nov 2026)

## Independent GitHub Actions results — both actual jobs completed SUCCESS

[GitHub Actions run #42097753160](https://github.com/Kamara229/JessicaCore/actions/runs/42097753160) generated and ran a truly physically charged archive from the pinned original HuggingFaceTB/SmolLM2-360M-Instruct (revision `a10cc1512eabd3dde888204e902eca88bddb4951`). [Complete original 360M binary artifact #13448700205](https://github.com/Kamara229/JessicaCore/actions/runs/42097753160/artifacts/13448700205) and [all actual native QA + baseline original-FP32 answers artifact #13447388861](https://github.com/Kamara229/JessicaCore/actions/runs/42097753160/artifacts/13447388861) preserve all evidence.

### Measured physical rate **PASS**

| Property | Actual |
|---|---|
| Original unique trained model parameters | **361,821,120** |
| Entire 360M physical source file (all matrices/1D/metadata/CRC/SHA) | **29,348,539 bytes** |
| Physically charged bpp per original unique parameter | **0.648907150583139** |
| Absolute physical ceiling at 0.700 bpp | 31,659,348 bytes |
| Remaining byte headroom | 2,310,809 bytes |
| Compressed original 2D matrices | **225**, including tied input/output embeddings once |
| Source codec | CB59 5-bit K32 per 8 source weights, FP16 codebooks and row scales |
| Complete binary SHA256 | `69e2e0f67c1e642ae468a94fc28047a7b1b77afe91776cb539e58e53aaef56a6` |
| CRC, SHA, aliases, 0 dense trained 2D matrices in native runtime | Verified PASS |

The compact 360M binary was produced in a separate stage, then a separate Actions job loaded the original **packed binary only**, constructed a shell model on meta and bound its original trained matrices to a native C lookup-GEMV/embedding decoder. It did NOT load original 360M FP32 trained source as a runtime or sidecar. All original source parameters, row scales, codebooks, metadata and 1D weights count towards physical file size. This is an actual complete-source compression result, not merely a compression-size estimate.

### Actual answer-quality **FAIL**

16 fixed public v193 development prompts, 4 each arithmetic, reversed numeric Yes/No, reversed fictional folder binding and factual short, original FP32 360M responses frozen in previous independent successful v193 run. Reused EXACT previously sealed v192 machine grader SHA `e230e480c02c680ae02d17b0825b451cbf76b138ceb743f181cc6fb62ad4ac1f`.

| Actual genuine source QA comparison | ORIGINAL 360M FP32 | ACTUAL compact 360M native C |
|---|---:|---:|
| **Strict machine-confirmed** | **8/16** | **0/16** |
| Strict recognized incorrect | 6/16 | 0/16 |
| Ungraded / cannot be asserted wrong by grader | 2/16 | **16/16** |
| Exact full answer match vs original | — | **0/16** |
| EOS produced within 32 tokens | Not summarized as separate gate here | **0/16** |
| Opposite Yes/No pairs both correct | **0/2** | **0/2** |
| Opposite fictional binding pairs both correct | **0/2** | **0/2** |

**Every one of the sixteen genuine native answers was degenerate repetitive gibberish**, as confirmed by inspecting the actual archived raw text, not merely the grader. Examples:
- Input `Compute 329 plus 48. Reply with the result as one number.`, original FP32 `329 + 48 = 377`; actual compact native begins `']]']]']]']]... wors wors ...` and never produces a valid numeric result.
- Input `Is 1103 greater than 1090? Reply with only Yes or No.`, original FP32 `Yes.`; compact output similarly begins repeating `']]` and `wors`.
- Input `In a fictional workshop ...` also generates gibberish and never differentiates opposite bindings.
- The same degenerate token family appears in unrelated math, truth, binding and factual prompts; text is often repeated for 32 tokens with no EOS.

The C-native executed 16 complete generation calls; wall-clock per generated example was about 3.7–4.95 seconds including prefill. **This is NOT a comparable fixed-work matched CPU speed benchmark vs an actual uncompressed or CB59 360M baseline, so no 90% speed PASS is claimed.** Both native job and physical source job GitHub CI **SUCCESS** means the code executed as intended; the user-facing language correctness gate is unequivocally **FAIL**.

### Scientific decision: next v195 must localize & fix severe semantic destruction, not just compress even further

The 29,348,539-byte baseline is an excellent **real rate-control**, but by itself entirely fails functional usefulness. The first 360M compressed trial demonstrates that the 5-bit-per-8-weight codebook family with this training/quantization is too destructive **as used here**; it does not establish that all <=0.7bpp compression approaches must fail. The native compute path successfully runs directly on packed weights, but this experiment by itself does not distinguish **quantization geometry error** from a **semantic architecture sensitivity** or native-kernel numerical issue. Do NOT infer a cause without causal control.

Next design: take the exact sealed physical v194 archive, verify C-native outputs against an independently decoded dense reconstruction at identical source weights (on tightly bounded public prompt samples), then use **diagnostic-only unpaid FP32 family substitution** (e.g. embedding/head vs attention vs MLP) to discover which compression source families destroy language. Label every FP32 intervention as **UNPAID DIAGNOSTIC, NOT <=0.7 BPP**. No downstream improvement can be called a compact quality PASS until a new complete physical binary proves it.

No public DEV answer fitting. No NLL/KL/PPL acceptance. Private sealed v89 TEST unread; no changes to main/LIVE Jessica, Render, Supabase, APK or Green Tunnel; unmerged isolated draft PR only. Jessica did not run a test.
