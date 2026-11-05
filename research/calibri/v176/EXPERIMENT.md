# v176: train the compressed model as a coupled system

v174 showed that restoring only one or two source families did not recover
answers. v175 improved all 211 calibration operators at exactly the old byte
budget, but produced 0/96 machine-correct development answers. Better local
matrix fits are therefore insufficient in these experiments.

## Intervention

Train ALL existing K32x8 codebooks, row scales and raw normalization vectors
jointly through the real 30-layer student trajectory. Preserve the tied input
embedding/output projection as one parameter source. The 5-bit assignments,
matrix dimensions, number of dictionaries, metadata and complete byte count
stay fixed. Forward computation rounds codebooks/scales to the actual FP16
storage precision using a straight-through gradient; norms retain their
already-paid FP32 slots. No extra adapter, dense correction or Green Tunnel.

Unlike v84's three independent down-projection dictionaries, v85's external
gates, v113's row-scale fit and v114/v115's newly trained toy models, this
pilot updates the full existing SmolLM2-135M compressed model together.
The intellectual precedent for joint codebook optimization is AQLM:
https://proceedings.mlr.press/v235/egiazarian24a.html . Its 2/3-bit findings
DO NOT establish quality at our 0.66784 bpp. This is not an AQLM reproduction.

## Fixed pilot and limitations

128 updates, seed 176, two shuffled passes over 64 samples. Half are 2,048
public TRAIN tokens (WikiText token positions 30000:31024, AG News TRAIN
records 18000:18100, first 1024 tokens); half are 32 short supervised examples
with values/names absent from the historical 96-question suite. Save resolved
dataset revisions, token hashes and exact task texts before fitting.

Normalized intermediate-hidden-state error stabilizes the compressed chain.
A hard token-ranking margin trains output decisions: original teacher argmax
on public text, ground-truth answer tokens INCLUDING EOS on instruction
examples. Prompt tokens are excluded from the supervised answer margin.
No KL, NLL, PPL or probability-distribution-matching objective or quality gate.
Hidden error and ranking loss are training signals, not evidence of useful
answers. The tiny training corpus, fixed indices, 128 updates and subjective
objective weights are explicit limitations. A negative pilot does not prove
that sub-bit models or joint training are impossible.

Select the final step in advance; no post-hoc checkpoint selection on QA.
Archive a progress checkpoint every 32 updates. Training can materialize
dense decoded matrices for autograd; this is NOT a low-RAM training claim.
Only the final complete 11,229,303-byte CB59 archive is used for inference.

## Checks and outcome

Tests cover tied-source gradients, exact physical rounding, prompt masking,
hard-negative margin, complete serialization, invalid fields, alias identity,
native numeric agreement and train/evaluation question separation. Compare
the full surrogate/native logits before and after fitting on a TRAIN window.
After exporting the source, discard the optimizer and training model, seal
SHA, and run the old native packed executor without dense trained matrices.

Generate all 96 historical development questions for FP32, native CB59 and
native CB176, 64-token cap with EOS/truncation flags. Preserve complete texts,
token IDs and conservative machine extraction; ungraded prose needs semantic
review. Repeat exact 32-token native timing on eight warmed, alternating
pairs. The reference is the original model's actual answers, not a claim that
it answers all questions correctly. Target unchanged bytes and >=90% CB59
CPU speed, with actual answer recovery; technical CI success is not quality.

This suite has been seen in prior research and is not a new independent TEST.
No private v89 TEST, production changes, main merge, deployment or messages
through Jessica. Research-only draft PR.
