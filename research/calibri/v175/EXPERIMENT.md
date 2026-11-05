# v175: full-covariance group error compensation, unchanged physical source

This is a new candidate experiment, not a claim of recovered FP32 quality.
Base: original SHA-locked CB59, 11,229,303 bytes, 0.6678394131 bpp.
Only the existing 5-bit indices and integrity checks may change. Original
FP16 centroids, FP16 scales, 1D tensors, aliases and metadata stay identical.
The original native C kernel executes the resulting archive directly.

## Difference from previous experiments

v66 updated 64 rows of four matrices using output-residual coordinate descent;
v73 used local 8x8 covariances without inter-group compensation; v172 used
diagonal feature energy while refitting dictionaries. v175 processes every
row of all 211 matrices, using whole-operator second moments and sequential
compensation into still-unquantized groups. It keeps dictionaries fixed so
assignment effects are isolated. This is a known second-order quantization
principle, not an invented new theorem or a full GPTVQ reproduction.

Source background: GPTQ https://arxiv.org/abs/2210.17323 and GPTVQ
https://arxiv.org/abs/2402.15319 . These publications do not establish quality
at CALIBRI's 0.66784-bpp rate. Their higher-rate outcomes are not extrapolated.

## Frozen calibration

Use historical public TRAIN wiki_fit/news_fit from paid_lowrank_v76.get_tokens:
256 tokens each, 8 windows of 64. Dataset revisions, record/token spans and
actual token hashes are saved before fitting. No QA labels or generated
teacher responses are input to fitting. Shared q/k/v and gate/up input moments
are captured once: 121 moment matrices cover 211 trained weight matrices.
This is a small calibration pilot; low sample rank and teacher/student
trajectory mismatch remain limitations, not solved problems.

Normalize H by mean diagonal, add fixed 0.05 I damping. The tied input/output
matrix adds 1.0 I to retain source fidelity for its input-embedding role as
well as head activation sensitivity. This is a preregistered heuristic, not
an exact joint embedding/head objective. No selection over damping values.

For H^-1=U.T U, choose each existing 8D centroid under the conditional metric
||error @ inverse(U_BB)||^2; update remaining normalized weights by
-error @ inverse(U_BB) @ U_BR. The sign is tested against the direct constrained
quadratic optimum. If the entire proposed matrix has worse regularized
calibration error than its original codes, retain original codes for that
matrix. This guard is not an answer-quality gate.

## Actual model verification

Serialize and reload the entire candidate; assert exact byte count and
unchanged fields. Seal SHA before QA. Check representative real native matrix
and embedding outputs against literal decoding, and absence of dense trained
2D parameters in both compact models. Run the same 96 historical development
questions on original FP32, native CB59 and native CB175, with 64-token cap,
EOS/truncation flags, exact tokens/texts and conservative parser results.
Unparsed prose and lexical gold occurrence are not semantic grades.

Measure native CPU speed separately on eight matched pairs: same prompts,
exactly 32 steps, alternating model order after warmup. Forced post-EOS work
is only a throughput diagnostic. Preserve complete end-to-end answer timing.

Private v89 TEST unopened. No KL/NLL/PPL quality gate. No Green Tunnel,
main, LIVE Jessica, Render, Supabase or APK changes. Do not merge or deploy.
