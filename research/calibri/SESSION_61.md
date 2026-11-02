# CALIBRI v61 — paid activation-guided residual "syndrome" side information under hard ≤0.700bpw

## One experimental step requested by user (before first CI, no Jessica chat dependency)

The user's packet-error-correction idea is valuable **as an architectural metaphor**, but requires a strict distinction: conventional forward-error-correction codes can reconstruct **erased transmitted bits only when enough redundancy and structural assumptions exist**, and a CRC can detect errors but does NOT restore original parameter information discarded by a lossy quantizer. We cannot claim that an 0.7bpw file contains all 16 original bits/weight with lossless exact decoding; that would contradict the bit budget for arbitrary trained tensors.

**Practical adaptation:** treat the base K32V8 0.667839413bpw original trained model CB59 as a high-rate "base signal", and spend its **physically available** ~0.032bpw headroom, up to a hard 0.7bpw, on selected ORIGINAL-minus-quantized residual "syndromes" `R[:,j]=W[:,j]−Q[:,j]`. The decoder doesn't need full original W. At runtime the native C 5bit codebook kernel evaluates `Qx`, then adds `sum_{selected channels j} R[:,j]*x[j]` using paid FP16 one-dimensional vectors. This is sparse error **side-information reconstruction**, mathematically exact for an individually selected FP16-approximated source input column, not true packet-erasure parity. It preserves the already proven **direct computation over resident packed pretrained weights** without full dense model restoration.

### Frozen model/storage

Real trained `HuggingFaceTB/SmolLM2-135M-Instruct` pinned HF SHA `12fd25f77366fa6b3b4b768ec3050bf629380bac`, UNIQUE source parameter count exactly **134,515,008**, tied input embedding and output head stored just once. Full pretrained CB59 K32V8 source originally **11,229,303 bytes, 0.6678394131bpw**, one packed 5bit index per 8 original weight components, FP16 8D centroid dictionaries and FP16 row scale, source 1D vectors exact and original tensor alias manifest. Run this unchanged from true FP32-converted public pretrained source, with cross-CPU exact source SHA.

New version **CB61** is a SINGLE physically stored SHA256+CRC32 archive with explicit parent CB59 full bytes and SHA256 plus a sorted manifest and FP16 residual vectors, with no external model, hidden sidecar, or uncounted correction. Max entire CB61 physical bytes `floor(0.70 × 134515008 / 8)` including all metadata, checksum, parent file, alias/tied source, residual vectors; failure if cap exceeded.

### Frozen selection from TRAIN-only ACTUAL teacher activations

Capture real teacher's all original trained `nn.Linear` **input-channel second moments** `d_j = mean_TRAIN(x_j^2)` on WikiText2 TRAIN token positions **[0:256]**, two separate forward passes each 128 tokens; reproducible pinned dataset revision and token SHA. Attribute canonical LM-head activation to the original tied embedding source name. Do not inspect test/dev evaluation prompt content, answer labels, or original teacher response to choose syndromes. Round stored mean-square values to 5 decimal places for numerical stability. For each unique original source 2D matrix, compute **original weight residual per 8D group** against the EXACT physically quantized CB59 codebooks and FP16 scales (without modifying base). Candidate input channels `j` ranked by **source residual column energy × TRAIN activation second moment**, i.e. `score_j = d_j · Σ_i (W_ij − Q_ij)^2`. Keep at most top THREE columns per matrix before global selection, then globally rank proposed columns by approximate saved TRAIN-activation response-error proxy divided by ACTUAL FP16 residual-vector bytes plus manifest overhead. Greedily store as many high-return source columns as possible, with conservative fixed 16kB metadata reserve; if final physically serialized CB61 exceeds hard cap, remove last lowest-priority one-column repair, serialize and SHA again. Source FP16 residual vectors are immutable after written; no tuning using any model development outputs.

During native full-model inference, `PackedLinear` calls unchanged C packed GEMV to consume all original 5bit codes and then adds each selected 1D channel correction `x_j · R[:,j]`. `PackedEmbedding` decodes ONLY requested rows and adds selected coordinate residuals for the requested token, not dense vocab matrix; shared LM-head alias applies identical correction vector on its own input-channel. Hook buffers only 1D source residuals; require zero 2D pretrained dense model parameters and all 212 original linear/embedding modules using native compressed source. A separate process **must never load original pretrained teacher** when executing the CB61 student.

### One predeclared development diagnostic (not semantic success gate)

After the CB59/CB61 files and corrections are frozen, compare original FP32 teacher and both native direct packed student variants on EXACTLY eight previously unused short English prompts:
1. `The lungs exchange oxygen and carbon dioxide because blood travels through`
2. `A spacecraft must adjust its orbit before approaching the distant station`
3. `During winter, animals may store energy to survive periods of`
4. `In mathematics a prime number is divisible only by one and`
5. `To prepare a glass of lemonade first squeeze fresh lemons into`
6. `The city council voted to renovate the old bridge after engineers found`
7. `We can test our scientific hypothesis by measuring the same process`
8. `After the long concert the musicians carefully placed their instruments inside`

Each fixed prompt tokenized using first 8 real tokenizer tokens, whole pretrained 135M forward and one actual autoregressive next-token KV-cache continuation, 8× actual FP32 teacher output distributions vs CB59 and CB61 packed student full-vocabulary original→compressed KL divergence, normalized logit RMSE, and top1 identity. Report every deterioration, including failure of semantic non-regression. The new prompts are **development diagnostics**, not first-use heldout proper task-answer accuracy or proof of nonfatal error. The initial v59/v60 tiny-prompt KL ~11–14 indicated major pre-existing functional failure, so current error-repair hypothesis may still fail; this stage is a direct feasible test of redundancy-aware partial restoration, not an unsupported quality claim. No output can be fed back into ranking in the same stage.

### Independent reproducibility and safe scope

Two independent GitHub Actions Ubuntu22.04 and Ubuntu24.04 pinned CPU torch2.5.1, transformers4.46.3, NumPy1.26.4, AVX2 and 2 threads, original fixed model SHA, 0.7bpw **measured from actual source archive file**, complete parent CRC/SHA and new CB61 CRC/SHA, synthetically corrupted archive rejected, tied embedding/head alias retained. Third checker independently recomputes physical byte cap, parent hashes, calibrated TRAIN token SHA, 8-prompt original-vs-direct response arithmetic, zero 2D original weights in packed process and crosshost source SHA. The new full model is expected to be slow; optimizing 128+32 warm speed remains a later independent stage at user's direction.

Relevant prior work: QMoE Frantar/Alistarh, MLSys 2024 (direct execution and sub-bit MoE); LQER Zhang et al., ICML 2024 and BALANCER (IJCAI 2026) use quantization error compensation with low-rank/activation awareness at higher bitrates. They do NOT establish that arbitrary 135M dense models can retain language task ability at 0.7bpw.

Research branch/draft PR ONLY. Production main, deployed Jessica ordinary chat, Render, Supabase, APK, settings, secrets or user data **must not be modified**.
