"""Prerecorded SciQ/BoolQ and true greedy generation diagnostic.

Physically paid original CB161 layer0 improves short heldout text NLL
(v161), but teacher greedy top1 matching ~0; v162 tests longer NLL
robustness. Need determine if REAL language/semantic usefulness changes.
This is exploratory 16+16 validation rows and four true autoregressive
32-token prompts, NOT the protected v89 independent TEST.
No training, candidate selection, QA labels in optimizer or output adapters.
"""
from v146.protocol import (SCI_Q,SCI_Q_COMMIT,BOOL_Q,BOOL_Q_COMMIT,
                           MAX_SUPPORT_CHARS,MAX_PASSAGE_CHARS)
from v117.protocol import ORIGINAL_CB59_SHA,TOTAL_UNIQUE,MAX_SOURCE_BYTES
from v161.protocol import MATRIX_KEYS
from v94.generation import NEW_PROMPTS,GENERATION_TOKENS

DATASETS=("SciQ","BoolQ")
ARMS=("FP32_teacher","original_CB59","fully_paid_CB161_block0")
PACKED_ARMS=ARMS[1:]
ROW_BEGIN=704
COUNT=16
ROW_END=ROW_BEGIN+COUNT
SPLIT="validation"
GENERATION_INDICES=(2,5,8,12)
LABELS={"SciQ":("A","B","C","D"),"BoolQ":("Yes","No")}
PHYSICAL_CB161_RUN=41920207168
MODEL_FULL_BYTES=11292174


def check():
    if (ROW_BEGIN,COUNT,ROW_END,SPLIT)!=(704,16,720,"validation"):
        raise ValueError("predeclared fresh QA validation rows modified")
    if ROW_BEGIN<=623:
        raise ValueError("must not reuse v150 SciQ/BoolQ labels")
    if (ARMS,PACKED_ARMS)!=(("FP32_teacher","original_CB59","fully_paid_CB161_block0"),
                            ("original_CB59","fully_paid_CB161_block0")):
        raise ValueError("unregistered actual packed model comparison")
    if (GENERATION_INDICES,GENERATION_TOKENS)!=((2,5,8,12),32):
        raise ValueError("greedy prompts or token count changed")
    if len(set(GENERATION_INDICES))!=4 or max(GENERATION_INDICES)>=len(NEW_PROMPTS):
        raise ValueError("incorrect original historical four generation prompts")
    if len(MATRIX_KEYS)!=7 or MODEL_FULL_BYTES!=11292174:
        raise ValueError("source model no longer original physically paid block0")
    return True
