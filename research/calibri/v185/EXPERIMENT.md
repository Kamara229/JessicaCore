# CALIBRI v185: actually change the 5-bit indices, then measure real answers

**New distinction from unsuccessful v176/v184:** this model
*rewrites physically packed 5-bit group codes*, not just their
FP16 codebook values, FP16 scale or unaccounted float adapter.
It is the original pretrained CB59 file format and exact original
total physical file byte count, 11,229,303 bytes = ~0.66784
bits/original unique parameter, so native inference should use
the existing verified C packed GEMV kernel without new CPU overhead.

## Frozen exact experiment

- The immutable original 134,515,008-unique-weight SmolLM2-135M
  original CB59 and pinned HuggingFace checkpoint only. The source
  checkpoint and pinned public TRAIN datasets follow v176.
- 24 predeclared full-model backpropagation steps on **16 original
  public TRAIN windows/independent supervised examples**; no
  v169 development questions or independent private v89 TEST are
  supplied to source optimization.
- All 211 original per-matrix K32/V8 FP16 centroids are trained by
  quantization-aware FP16 straight-through gradients. Original
  61 1D weights, original positive per-row scales, source tie,
  index bit width and metadata are immutable.
- At each predetermined step select one original *predeclared*
  attention output or MLP down-projection source matrix across
  first/middle/last decoder blocks. Retain its actual full-model
  TRAIN loss gradient through the temporary decoded dense W.
  For each 8-weight source group, evaluate all 32 available source
  code assignments with a gradient-projected score plus regularization
  against the original pretrained FP32 weights (TRAIN weights only,
  not query answers).
- Propose only the 512 most promising group-code changes per step,
  with deterministic tied ordering and strict positive score
  improvement. **Accept only if actual full-model supervised TRAIN
  forward loss improves on the same frozen TRAIN input**.
  Otherwise revert proposed codes; retain the independently trained
  FP16 codebooks. This is alternating projected discrete / continuous
  source optimization, not a hidden FP32 inference model.
- At step 24 export *every changed 5-bit code stream and FP16
  codebook* back into **the same original CB59 physical allocation**,
  recalculate CRC/SHA, verify unchanged raw 1D tensors, unchanged
  scales, unchanged tied embedding/output alias, unchanged exact
  source file length. Compare saved native full-model output to
  its quantized TRAIN surrogate before unsealing any public DEV QA.
- Test actual first-token matches and diversity on 20 frozen
  historical DEVELOPMENT questions; full raw answer text, EOS and
  machine-confirmed correctness on 24 frozen historical
  DEVELOPMENT questions. Compare original pretrained FP32,
  original native CB59 and candidate native CB185 under exactly
  identical inputs. 8 matched 32-token native CPU paired runs
  measure >=90% speed relative to original CB59 baseline.
- Final acceptance is **coherent correct responses relative to
  the original FP32**, <=0.700 *complete physically counted*
  bits/original parameter and >=90% physical native CPU throughput,
  NOT improving TRAIN CE, NLL, KL, PPL, index mutation counts or
  Github CI completion. An unsuccessful experiment must be reported.

Changing code indices at fixed K32/V8 may still be inherently too
restrictive for reasonable 135M model behavior: this experiment
tests rather than presumes that hypothesis. The limited 24-step,
up-to-512-changes/step schedule is a controlled pilot, not
a claim of whole-model convergence or of global-optimal rate.

**Isolation:** research branch/draft PR only. No main/LIVE Jessica,
Render, Supabase, APK, Green Tunnel or private v89 TEST modifications.
All tests via direct GitHub workflow owned by ChatGPT; Jessica is
only a literal chat relay and runs **no tests**.
