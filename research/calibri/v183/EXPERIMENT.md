# CALIBRI v183: exact first-body-block source corruption isolation

v182 conclusively reproduced V1 native results in exactly the same
FP32 PyTorch model (max relative discrepancy 2.93e-5) and isolated
major activation divergence to the first internal transformer block,
even with the tied embedding/output head kept original FP32. This
v183 study is a controlled, **unpaid oracle** locating which of the
first block's seven operators needs a better source representation.

The starting model is the original **physically frozen v181 V1 archive**
decoded in RAM for only the 210 body matrices. Its tied embedding and
LM output head are ORIGINAL FP32, as in the measured v182 BODY_ONLY
control, and the 61 1D tensors are original. This is neither the
physical V1 native model nor a <=0.7-bpp candidate: it contains
unpaid full dense and original FP32 head tensors.

For each of seven original projection matrices (q/k/v/o, gate/up/down
of layer 0), temporarily restore ONLY that original FP32 matrix
while holding all other 209 V1 body matrices fixed. Also restore
all attention 0 matrices, all MLP 0 matrices, or all seven.
All comparison model architecture, tokenizer and public cases
(IDs 0, 16, 64, 80) are fixed before evaluation. No optimization,
original answer or logit fitting; no v89 independent TEST.

Measure the last-position output of the very first transformer
block and the final next-token logits versus original FP32 on
all four prompts. Record first-token agreement and per-case
relative L2/cosine/maximum deviation. Compare each arm with the
same frozen V1_BODY_ONLY control, including negative regressions.
Use original FP32_ALL_BLOCK0 as numerical control: first-block
output must exactly match FP32 within tolerance. Always reset
source weights between arms, never leave one operator repaired
when testing another.

This identifies a relative *source sensitivity* under an unpaid
diagnostic substitution, not deployable efficiency. Only after
this ranking can a new physically costed CALIBRI trainable or
structured source with <=0.700 total bits/original parameter and
>=90% CB59 native CPU speed be proposed. Real coherent answers,
not KL/NLL/PPL, remain the quality gates.

All code/tests in separate isolated research draft. No main, LIVE
Jessica, Render, Supabase, APK, Green Tunnel or private TEST.
Jessica is literal chat relay and MUST NOT execute any tests.
