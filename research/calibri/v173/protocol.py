"""CALIBRI v173: diagnose if repetitions arise from compressed bits or CPU kernel.

Physically stored original source (CB59) stays immutable and <0.7bpp.
All dense dequantization is explicitly an UNPAID diagnostic reference,
NEVER a proposed runtime/archive or production model.

If exact SAME compressed weight values executed as dense FP32 also repeat,
the failure is in representation/quantization, not an indexing/ctypes bug.
"""
from v117.protocol import ORIGINAL_CB59_SHA,TOTAL_UNIQUE
from v94.generation import NEW_PROMPTS

EXPECTED_BYTE_SIZE=11229303
EXPECTED_MATRICES=211
PREFILL_CASE_IDS=(0,2,4,6,8,10,12,14)
GENERATION_CASE_IDS=(0,2,4,6,8,12)
MAX_OUTPUT_TOKENS=32
ARMS=("real_original_FP32_teacher","native_original_CB59",
      "UNPAID_dense_decoder_of_exact_CB59_weights")

def check():
    assert len(PREFILL_CASE_IDS)==8 and len(set(PREFILL_CASE_IDS))==8
    assert len(GENERATION_CASE_IDS)==6 and len(set(GENERATION_CASE_IDS))==6
    assert max(PREFILL_CASE_IDS)<len(NEW_PROMPTS)
    assert max(GENERATION_CASE_IDS)<len(NEW_PROMPTS)
    assert (EXPECTED_MATRICES,EXPECTED_BYTE_SIZE)==(211,11229303)
    assert 8*EXPECTED_BYTE_SIZE/TOTAL_UNIQUE<0.7
    return True
