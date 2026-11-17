# Oscar v1I — learn original FP32 function through the **physically stored quantizer**, not by adding another adapter

## Hypothesis (precommitted)

The actual independently pretrained original Karpathy TinyStories 41,689,600 FP32 weights, eight layers×hidden512, **fail direct K32/V8 cold codebook compression**, and even two recurrent Green Tunnel routes (v1F first / v1G later) give **0/20 original FP32 complete 8-token answers**, despite 11/20 correct first tokens and less repetition. In v1H the original single CB61 source was *untouched* and still has 61% normalized reconstruction error. The next work must directly correct the **source quantizer**, not merely graft on increasingly large external GTs.

## Specific physical intervention: tied embedding+LM-head quantizer

The most widely reused original source tensor is `tok_embeddings.weight` (`output.weight` tied alias), genuine 32,000×512. Its K32/V8 index stream contains one 5bit code per 8D vector. Before touching other layers, run real functional calibration of the original FP16 **32k row-scales and shared 32×8 codebook centroids**, preserving every packed index, **exact original tied alias and all 57 other source tensor descriptors**. This changes the actual original CB59/CB61 compressed model binary and occupies **zero extra model tensor parameters**, unlike an additional GT/head sidecar.

Physically refine scalars in the original CB59 source:
- original independent `tok_embeddings.weight` normalized FP16 scale per word is multiplied by a trainable positive `exp(log_gain)`;
- original 32×8 FP16 codebook centroids are differentiably calibrated;
- immutable existing 5bit K32V8 packed indices and all OTHER tensors remain byte identical;
- original encoded CB59 header length, descriptor layout, checksum CRC32+SHA256 and alias `output.weight → tok_embeddings.weight` re-authenticated; CB61 recompressed from NEW actual binary; no teacher FP32 at inference;
- new source is loaded into **fresh original 8-layer model**, using both physically trained prior v1H full-eight recurrent rank32 GTs unchanged (v1F initial token, v1G later).

### Original independent FP32 teacher and clean declared data separation

Teacher is the actual pretrained `karpathy/tinyllamas/stories42M.pt`, SHA `fbe124df8794c2e033432c22b59cff64485455b4644046c52a456d0bbd8859fe`, 8 layers×512, never CALIBRI 12×320. Input original tied source CB59 SHA `f7b5fe62...`, original CB61 SHA `724ac408...`, two genuinely trained GT FP16 SHA `2fc4b5c...` and `506ea679...`. Public LLaMA proxy tokenizer is pinned but its exact equivalence with the training tokenizer isn't established.

**TRAIN ONLY**: exactly the same 12 previously preregistered v1F TRAIN story prefixes (not any public-debug). **48 CPU optimizer updates**, each using genuine full 32k independent teacher next-token distributions and all 8 decoder hidden states at BOTH (1) the original prompt under *v1F FIRST* GT and (2) the original prompt plus the student's **own actually generated first token** under *v1G LATER* GT. Optimizer changes **only original tied K32/V8 32×8 centroids and the 32k FP16 row scales**, no external GT or other original compressed matrices. Regularize the original codeword/scale values to avoid destroying their prior 5bit source meaning. Teacher is stop-gradient; learn soft-KL(T2)+teacher argmax CE+8-layer hidden cosine, control actual first-token margin and FP32 output. No prompting→answer lookup table at inference.

**DEBUG/evaluation only after training**: the same 20 public DEBUG previously examined for v1H (8 original v1F +4 old v1G+8 first evaluated v1H), plus eight newly predeclared PUBLIC debug story prefixes from separate fixed sentences. These are not a clean sealed QA; they must **never influence training, optimizer trajectory or checkpoint selection**. Report authentic original FP32 first/full 8 tokens/EOS, sequence repetition and hidden cosines after first and later contexts. Baseline v1H untouched old source vs physically reloaded v1I new source, SAME two GT sidecars, no new head. Assert zero-change decoder before optimizer.

**Physical bpp acceptance:** original v1H is 3,561,227 real bytes / 41,689,600 trained weights = **0.68338 bpp** for one original CB61 source+2 GT FP16+paid 3072B config. New *recompressed* CB61 size can change and must be measured; require <= 3,647,840 B total (**0.7 bpp**) with two physically real unchanged GT sidecars and paid config. Avoid counting the diagnostic old CB59 or teacher model as deployed. Tokenizer and executable/native C runtime excluded; no CPU90/J7/clean QA claim. Report loss of first-token or full-response quality as a negative result, never assume improvement.

**No original 12×320 S142 CALIBRI, Green Tunnel3/9, NRA2, LIVE, main, Render, Supabase, APK or ordinary Jessica tests.** Draft PR research only.
