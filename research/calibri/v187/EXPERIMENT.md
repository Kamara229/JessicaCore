# CALIBRI v187 — physical 4D neural-weight tiling pilot (source and CPU only)

This study is **orthogonal** to the ongoing v186 balanced-behavior
source training. v186 is left untouched. The pilot specifically tests
the owner's *multidimensional weights* hypothesis, not another
floating correction to the same 1x8 vector quantizer.

## Actual four-dimensional tensor organization

Original pretrained matrix `W[m,n]` is rearranged (losslessly
before quantization) into a tensor of shape
`[m/2, n/4, 2, 4]`. The axes denote adjacent output-row pairs,
input-column blocks, within-pair output identity and four
within-block input coordinates. A 32-entry FP16 dictionary maps
each multidimensional 2x4 block (8 original values) to exactly
one 5-bit index.

This differs materially from the baseline K32/V8 **1x8**
contiguous-row grouping:
- Baseline: one 5-bit index covers eight input features of ONE
  original output neuron.
- v187: one 5-bit index covers four input features of TWO
  ADJACENT output neurons, preserving possible cross-output
  structure in a shared dictionary.
- Same number of original weights per code, same codebook
  `32x8` FP16 and same independent FP16 scale for each
  original output row. Therefore **exactly the same per-matrix
  5-bit packed bytes + FP16 codebook + FP16 row-scale bytes**.
  No unpaid index auxiliary file or dense original matrix.

A native C kernel directly computes both output neurons from
the packed 4D tile code and a common four-input block. A second
C function decodes only the embedding row demanded by input
token IDs, respecting the original tied-head geometry in a future
full model. Both are numerically checked against literal
decoded 4D source, but the native kernel never receives the
original FP32 pretrained weight matrix during CPU inference.

## Precommitted real pretrained pilot scope

Load the same pinned original genuine SmolLM2-135M FP32 weights.
Evaluate **exactly eight** original real tensors selected before
the error comparison: the shared embedding/output matrix,
block 0 attention O, block 0 MLP gate/down, block 15 attention
O/down and block 29 attention O/down, plus the eighth
precommitted matrix listed in the protocol. No PRIVATE TEST,
no training labels and no historical DEV prompts.

For EACH selected tensor compare K32/V8 1x8 versus 2x4:
- Fully paid per-matrix source payload: MUST be identical.
- Relative FP32 weight reconstruction error (diagnostic only).
- Native 4D GEMV vs literal decoded 4D source: numeric pass.
- Native 4D embedding vs exact decoded selected rows.
- Five matched single-matrix CPU GEMV timings, alternating
  1x8 and 2x4 order, same original shape and input values.
- Include failed/negative relative-error/speed measurements.

**Not a full LLM or real-answer result**: this pilot does not
create and price a complete 134,515,008-parameter 4D model
archive, replace all layers, generate a user response or
establish full-model CPU throughput. It tests whether
cross-row structure has measurable source-rate/quality
promise before spending the resources to implement full
physical code and actual QA. Matrix RMSE alone is not
end-user correctness, and no positive source-level
result is allowed to be marketed as a CALIBRI quality PASS.

Subsequent full-model admission (only if warranted) requires
physical <=0.700 bpp for complete unique-weight archive,
actual answer accuracy/coherence relative to original FP32,
native CPU speed >=90% CB59 and frozen independent blind
TEST. NLL/KL/PPL are not final gates.

All artifacts remain on a separate isolated research branch
and draft PR. Never modify main, LIVE Jessica, Render,
Supabase, APK, Green Tunnel or PRIVATE v89 TEST. Jessica
is relay-only and must never run model tests.
