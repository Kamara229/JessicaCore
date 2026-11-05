"""CALIBRI v170: replace source K32x8 codewords, not add weights or metadata.

Research hypothesis: CB59 used just 4096 evenly spaced source groups per
whole trained matrix to fit centroids, which can distort layer operators.
Use more source-only data to refit ALL original 2D operators but retain the
exact number of 5bit/8D codes, 32 FP16 centroids, row FP16 scales, 1D
source parameters, and byte-for-byte archive length.

No KL/NLL, teacher logits or QA labels participate. Objective is to
recover meaningful answers at SAME physical bits on untouched models.
"""
from v117.protocol import ORIGINAL_CB59_SHA,TOTAL_UNIQUE
from v169.protocol import FULL_ARCHIVE_BYTES

ARMS=("original_FP32","original_CB59","equal_byte_CB170")
FIT_SAMPLES_PER_MATRIX=32768
LLOYD_PASSES=6
CHUNK_GROUPS=8192
GENERATION_TOKENS=32
# 40 actual different free-response tasks; 8 in each class.
CASE_IDS=(0,1,4,5,8,9,12,13,
          16,18,20,22,24,26,28,30,
          48,49,50,51,52,53,54,55,
          64,65,66,67,68,69,70,71,
          80,82,84,86,88,90,92,94)
NUM_CASES=40
# Every model gets identical actual free-response prompts.
def check():
    if len(CASE_IDS)!=NUM_CASES or len(set(CASE_IDS))!=NUM_CASES:
        raise ValueError("duplicated or missing predeclared source-only QA")
    if min(CASE_IDS)<0 or max(CASE_IDS)>=96:
        raise ValueError("invalid reference questions")
    if (FIT_SAMPLES_PER_MATRIX,LLOYD_PASSES,CHUNK_GROUPS)!=(32768,6,8192):
        raise ValueError("source-only codebook geometry changed")
    if FULL_ARCHIVE_BYTES!=11292174:
        raise ValueError("v161 immutable compared size changed")
    if ARMS!=("original_FP32","original_CB59","equal_byte_CB170"):
        raise ValueError("unregistered compact candidate")
    return True
