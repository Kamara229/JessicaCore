# Oscar v1J — causal per-matrix and inter-layer pretrained FP32 sensitivity audit, no new model

## Confirmed evidence

Independent genuine pretrained original `karpathy/tinyllamas/stories42M.pt` FP32 SHA `fbe124df...`, 41,689,600 unique, 8×512 (NOT CALIBRI S142 12×320). Cold Oscar CB59 5bit K32/V8 severely corrupts trained network, same-architecture complete FP32 response parity 0/4. Full8 recurrent GT v1F repaired hidden similarity and 5/8 first. Dual-GT v1H original pretrained teacher first11/20, full0/20. Oscar v1I **ACTUALLY TRAINED and RESERIALIZED original tied embedding/head K32/V8 codebook/scales** from 48 FP32-supervised steps, changes in 252 FP16 codeword values and 31,936 row scales, but old20 first **11/20→11/20**, complete **0/20→0/20**, new8 3/8→3/8, repetition slightly worse. Original source centroids alone are NOT the missing fix.

## Precommitted purpose

Directly measure whether **inter-layer interaction and selective original matrix corruption** are bottlenecks. Take the proven original v1H pair of trained rank32 recurrent 8layer GT sidecars (FIRST v1F SHA `2fc4b5...`, LATER v1G SHA `506ea679...`) with exact ORIGINAL cold CB59 42M source SHA `f7b5fe...` and independent FP32 teacher. No source encoded model changes, training, new aliases or additional model outputs.

For each of **all original 57 unique physically compressed 2D matrices** (32 attention Q/K/V/O, 24 FFN W1/W2/W3, one tied 32k embedding+output):

1. In an **explicitly UNPAID diagnostic only**, substitute EXACTLY ONE original full FP32 matrix from independent pretrained teacher into the decoded compressed student. The other 56 matrix groups remain unchanged at 5 bits; both original GT sidecars unchanged. This hybrid does **not** meet 0.7 bpp; it must never be packaged, claimed or promoted as a compressed model.
2. Evaluate on **exact same 12 previously preregistered TRAIN-only story prefixes** as earlier v1F and v1I, never on any of the 20+8 previously viewed public debug prompts. Evaluate FIRST next token under v1F full8 GT; evaluate SECOND next token under v1G full8 GT given the original **baseline compressed model's own first token**, held fixed across ablations (so we test matrices, not different histories). Original independent FP32 teacher sees exactly the same context for each paired student score.
3. Compare full32k FP32 next-token KL at T2, original top1 rank, all8 hidden cosine at the corresponding layers, and final-layer cosine. Compute a **predeclared** score = 0.65×(baseline FIRST KL − swapped FIRST KL)+0.35×(baseline SECOND KL − swapped SECOND KL)+2×(swapped final hidden cosine−baseline final hidden cosine). This is a sensitivity *ranking*, not a candidate final model or QA result.
4. Restore ORIGINAL physically decoded compressed tensor *bitwise* before evaluating next candidate; assert original CB59/CB61 sha/GT FP16 sha all unchanged. Matrices whose isolated restoration **worsens** quality are important too: it indicates original trained GT is compensating a *correlated* error, so separate elementwise codebook optimization may be counterproductive.
5. Report **every** matrix and category, top-10 and bottom-10, layer-specific first/second KL/cos and ranks. Include at least one later-layer attention and MLP candidate to test whether damage is at output or earlier. No actual FP32 teacher at inference; this is offline research, no trained GT reuse in S142 12×320, no original J7/QA/CPU speed testing.
6. No answer leak: TRAIN prompts already designated; original public DEBUG cohort never accessed until a NEW actual packed source correction is coded in a subsequent separate research draft. No source-weight correction included in v1J artifact.

**Expected scientific result:** a falsifiable ranked map of original source matrices that cause biggest loss, identifying whether attention projections, FFN transitions or tied embedding dominate with two interlayer GTs. Subsequent v1K should add bit-budget to those matrices via real physical source repacking (not extra unrestricted FP32 weights) and evaluate on independent debug.

No main/LIVE/Render/Supabase/NRA/APK/ordinary Jessica chat modifications, no production/merge; draft PR only.
