# CALIBRI v180 — actual orthogonal-source representation (BTC-inspired, not BTC reproduction)

After v175-v179, the byte-rate / native execution problem has been solved,
but actual SmolLM2-135M answers remain collapsed. This v180 experiment
targets a DIFFERENT *operator coordinate system* rather than adjusting
the K32x8 indices or per-row scale. The rationale is the use of
invertible transforms to mitigate quantization distortion; related
research BTC-LLM (Gu et al., ACL 2026) demonstrates learnable transformed
binary patterns in a different architecture/regime:
https://aclanthology.org/2026.acl-long.1066/ .
Their results at 0.7–0.8 bits on LLaMA/Qwen DO NOT establish the behavior
of our source format nor validate this unlearned experiment.

The exact unquantized operation W*x is preserved by replacing W with
W' = W * E, E = D * H, where D is a predetermined signed diagonal
vector and H a normalized orthogonal Walsh-Hadamard transform on
32 or 64 adjacent features. At inference, linear computes x' = x * E,
then native packed 5-bit lookup GEMV with quantized W', and tied
input embedding decodes only requested rows of W' and returns row*E.T.
No sidecars, no dense pretrained 2D weights in runtime; the original
CB59 physical format stores all 211 transformed K32/V8 matrices plus
existing FP32 1D norms and tied-source alias. Manifest explicitly
requires the transform-aware native loader to avoid using the ordinary
CB59 decoder in the wrong coordinates.

**Both experimental arms are frozen before training or QA:** H32 and H64.
The signed-pattern RNG and transformations are deterministic and
are NOT fit on gold answers, teacher logits or independent TEST.
Actual complete bytes include packed 5-bit codes, FP16 32x8 tables
and row scales, raw FP32 vectors, manifest, alias names, CRC+SHA.
The cap is 0.7 bits per 134,515,008 original unique parameters.

Unit controls: orthogonal inversion and exact FP32 linear identity;
reconstructed compressed source versus C native linear/embedding,
same physical tied input/output head and no full dense trained tensors.
Full run compares original FP32, original native CB59, H32 and H64
for actual answers and 8 matched CPU-speed pairs per candidate. Frozen
first-token triage on 20 historical development prompts advances to
96 actual answers per arm only if an arm has >=3 unique first tokens
AND >=2 that match the original FP32. Otherwise 24 frozen historical
development questions (32-token cap) per arm. Response quality, EOS,
actual content, not KL/NLL/PPL, is the scientific gate.

Limitations: all evaluation questions have been used historically
in this research, so they are DEV, not an independent blind TEST.
Local weight RMSE/first-token diversity are diagnostics only.
The signed Hadamard is an orthogonal input mixing heuristic; no
guarantee it learns the transformations BTC-LLM optimizes.
For the main target we need coherent, correct answers and >=90%
native CB59 speed; negative results remain reportable.

All research on this draft branch. No main, LIVE Jessica, Render,
Supabase, APK, Green Tunnel, or private v89 TEST access.
The user's later Jessica chat connector is **only a relay**, never
an executor of tests: tests and choices run directly in research GitHub
by the assistant, not through Jessica.
