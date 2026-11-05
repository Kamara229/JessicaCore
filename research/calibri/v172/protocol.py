"""CALIBRI v172: ACTUAL original operator-input-conditioned K32/V8 coding.

No additional physical model bytes: the frozen CB59 container is rewritten
within original 5bit indices + FP16 centroids only. Unlike v170 source
weight-RMSE refit, optimize a proxy for each operator's actual output
distortion on ORIGINAL FP32 teacher activations from preexisting diagnostic
unlabeled prompt text; never optimize NLL/KL, QA labels, or outputs.

The proxy is diagonal input-feature energy. This is an APPROXIMATION,
not a claim that source FP32 competence is restored. The definitive
result must come from the real generated answers in the next CI stage.
"""
from v117.protocol import ORIGINAL_CB59_SHA,TOTAL_UNIQUE
from v169.protocol import FULL_ARCHIVE_BYTES
from v170.protocol import CASE_IDS,NUM_CASES
ARMS=("original_FP32","original_CB59","activation_aware_CB172")
WEIGHT_SAMPLE_GROUPS=32768
PASSES=6
CHUNK_GROUPS=8192
ACTIVATION_INPUTS=16
MAX_GENERATED_TOKENS=32

def check():
    if ARMS!=("original_FP32","original_CB59","activation_aware_CB172"):
        raise AssertionError("precommitted actual compact arms changed")
    if len(set(CASE_IDS))!=40 or NUM_CASES!=40:
        raise AssertionError("new QA task count changed")
    if (WEIGHT_SAMPLE_GROUPS,PASSES,CHUNK_GROUPS,ACTIVATION_INPUTS)!=(32768,6,8192,16):
        raise AssertionError("input-weighted FP32 source fit changed")
    if FULL_ARCHIVE_BYTES!=11292174:
        raise AssertionError("source archive format changed")
    return True
