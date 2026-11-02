# CALIBRI v63 — Real signal-coding transfer: source entropy and chunked 5-bit index coding

**Status at commit: experiment preregistered; CI results must be checked. Research-only derivative of v62.** No production Jessica, Render, Supabase, APK, main, secrets, or live model actions.

## The question
Can a lossless source coder exploit CB59's distribution of 5-bit vector indices to physically **free bytes**, while keeping every stored quantized weight and the present native model outputs mathematically unchanged? The original dense model is public SmolLM2-135M-Instruct, pinned revision 12fd25f77366fa6b3b4b768ec3050bf629380bac, with 134,515,008 UNIQUE trained parameters. Existing CB59 is 11,229,303 bytes (0.6678394 bpw); max source archive size at 0.700 bpw is 11,770,063 bytes.

## Implementation — source-only, no LLM teacher
The research/calibri/entropy_chunks_v63.py script is a standard-library + NumPy auditor for the **original real CB59 file**. It first verifies SHA256, CRC32, manifest and complete segment layout. CI downloads the *already produced* original CB59 source from the successful v59 Actions run 41507201766 artifact calibri-v59-ubuntu-22.04. The v59 artifact expires after seven days; if unavailable, fail loudly and do not replace with a toy.

1. Extract all original K32V8 matrices' 5-bit indices; calculate source marginal H(Z), lag-one conditional H(Z_i | Z_{i-1}), and optimistic information lower bounds. These do not include model/decoder metadata and are NOT achieved compressed sizes.
2. Actually encode consecutive packed index **byte ranges** with zlib level 9 (DEFLATE baseline, NOT rANS/CABAC). Store raw bytes in chunks that do not benefit. Evaluate restart intervals of 64, 256, 1024 matrix rows (byte-aligned).
3. Physically serialize an offline **CB63 candidate** containing the original CB59 manifest, all non-index matrix data, all 1D tensors and tied alias information, chunk tables (16 bytes per chunk), CRC and SHA. Count every byte rather than projecting an entropy-only saving.
4. Random-access and decompress each index chunk, fully reassemble the original CB59 byte stream in an *offline check*, demand exact source SHA and bit-for-bit match for every chunk size. Test CRC/SHA corruption rejection.
5. Save JSON with measured physical archive sizes, net gains or losses, source-only entropy and rate checks. Retain best candidate even if it makes things worse: a negative result is publishable internal evidence.

The full CB59 original physical archive is not added to source control. The CB63 candidate is **NOT an inference-ready archive**: v63 does not implement a new native packed C/GEMV kernel or benchmark decoding hot paths. Restoring exact CB59 bits means unchanged output when running CB59, but does not prove CB63 can execute directly without any decoding overhead.

## Acceptance and next decision
- Hard integrity: real trained CB59 checksums, synthetic unit tests, real exact round trips, explicit complete actual source bytes.
- Source gain: measured CB63 physical bytes < 11,229,303 to claim **any** net saving. If not, the lossless approach FAILED this gate. Any source candidate > 11,770,063 also fails the full rate cap.
- Quality evidence: **none** — unchanged source, so no new accuracy or KL claim. The CB59 functional loss remains severe.
- Native speed and RAM: **not evaluated**, required before integrating index entropy with direct inference.
- If entropy is near 5 bits/symbol or chunk overhead dominates, move to source-index reshaping, task-aware trellis/feedback or low-rank binary factorization — not to imaginary free parity correction.
- If meaningful bytes are freed, the next separate trial can implement a finite-state rANS/CABAC source codec with bounded restarts and a native decoder, then spend recovered bytes only on independently blind-tested functional repair.

## Competitive baseline revision — 2025/2026 discoveries
The v62 architecture list is a map, NOT a claim of novel sub-1-bit compression. Particularly important new publications:
- **NanoQuant, ICML 2026**: sub-1-bit post-training low-rank binary factorization + alternating optimization + model reconstruction. https://proceedings.mlr.press/v306/chong26a.html
- **LittleBit, NeurIPS 2025**: binary latent low-rank factorization with multi-scale scaling and compensation, reports as low as 0.1 bpw. https://proceedings.nips.cc/paper_files/paper/2025/hash/a917c1d57088897beba47f96b4fd7f3c-Abstract-Conference.html
- **LittleBit-2, ICML 2026**: latent geometry alignment for those binary factors (repository). https://github.com/SamsungLabs/LittleBit
- **Radio, ICML 2025**: explicit rate-distortion allocation in LLM quantization. https://proceedings.mlr.press/v267/young25b.html
- **QTIP, NeurIPS 2024**: finite-state trellis-coded quantization. https://papers.nips.cc/paper/2024/hash/6de2e84b8da47bb2eb5e2ac96c63d2b0-Abstract-Conference.html
- **DeepCABAC, 2019**: signal/video codec entropy modeling directly transferred to neural weight coding. https://arxiv.org/abs/1907.11900

These works mean neither "sub-1bpw" nor "residual correction" alone supports a novelty claim. CALIBRI's differentiated question should be simultaneous physical rate, heldout language competence, bounded native packed computation, actual throughput, and reproducibility. Avoid comparing sparse QMoE results as if dense 135M and 1.6T sparse expert models were the same experiment.

## Expected output
GitHub Actions will produce results/calibri_v63_entropy.json and results/CALIBRI_v63_source_chunked_candidate.cb63. Numeric gains must be copied here or reported in follow-ups **only after** verified successful execution.
