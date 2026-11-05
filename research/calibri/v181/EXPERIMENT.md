# CALIBRI v181 — physically real information recovery-rate ladder

This is a **diagnostic**, not a <=0.7 bpp CALIBRI candidate. The
user's ultimate acceptance threshold remains <=0.7 bpp, correct coherent
answers approximating real original FP32 ability, and >=90% native CPU
speed. The v175-v180 source experiments showed that 5-bit indices per
8 original weights frequently destroy answers. Before more unprincipled
source refits, determine the approximate source information threshold
at which the SAME pretrained FP32 model can answer again.

Two **precommitted high-rate controls**:
- ORACLE_V2: 5 bits per TWO original weights, plus actual 32x2 FP16
  dictionary, row scales, raw 1D vectors, manifest+SHA/CRC, aliases:
  nominal 2.5 bpp + overhead.
- ORACLE_V1: 5 bits per ONE original weight, with 32x1 FP16 dictionary,
  full source accounting: nominal 5.0 bpp + overhead.

All original trained 211 matrices are compressed. A new native C
look-up GEMV/embedding directly consumes the packed five-bit indices.
It is not a source restoration trick: no FP32 original matrix or
trained dense sidecar survives inference. No reuse of original teacher
answers at runtime. The original pretrained FP32 weights are used only
to fit dictionaries/indices; QA gold, teacher logits and independent
TEST are not used to select source quantization. Only 24 historical
development questions and 20 fixed initial-token probes are checked
for every arm (FP32, native original CB59, ORACLE_V2, ORACLE_V1).

Accept no quality claims from a single token or a low source
operator error. Store complete actual model answers, token IDs,
EOS and conservative machine grading. Report native C numeric
roundtrip and physical bytes/SHA. Compute full physical bpp
against 134,515,008 original unique parameters. These variants
deliberately exceed <=0.7 bpp; any coherent answers they produce
are **only evidence for the minimum information-rate threshold**
and cannot satisfy the final CALIBRI target. Runtime speed here
is diagnostic and may fall below the 90% requirement.

If even V1 (5-bit per weight) fails relative to the real FP32,
debug source implementations, calibration and model/tokenizer before
doing further subbit research. If V1 succeeds and V2 fails, a
substantial representational/training breakthrough is required
to reach <=0.7 bpp. This is much stronger evidence than randomly
editing the same CB59 codebooks.

Only isolated research PR and GitHub Actions under assistant
control. **Jessica will NOT run tests** and will only serve as a
chat relay later. Do not change main, LIVE Jessica, Render,
Supabase, APK, Green Tunnel or private v89 TEST.
