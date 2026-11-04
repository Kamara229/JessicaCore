"""Predeclared post-v149 validation only; no modifications to any weights."""
from v146.protocol import SCI_Q,SCI_Q_COMMIT,BOOL_Q,BOOL_Q_COMMIT,MAX_SUPPORT_CHARS,MAX_PASSAGE_CHARS
from v94.generation import NEW_PROMPTS,GENERATION_TOKENS

CB149_SHA="28d7a34f004a99f0162319e4333cd7db49746bd3ea5d9ff3a203c775119c2c22"
CB147_SHA="639be185c259260c614823831275113ffa2d83231eb54805ebfc67be30da5bbd"
DATASETS=("SciQ","BoolQ")
ARMS=("FP32","CB59","CB147_prior","CB149_contextual")
ROW_BEGIN=600
QUESTION_COUNT=24
ROW_END=ROW_BEGIN+QUESTION_COUNT
SPLIT="validation"
GENERATION_INDICES=(4,5,8,9,12,13)
GENERATION_ARMS=("CB147_prior","CB149_contextual")


def check_protocol():
    if (ROW_BEGIN,ROW_END,QUESTION_COUNT,SPLIT)!=(600,624,24,"validation"):
        raise ValueError("posthoc source slice change prohibited")
    if ARMS!=("FP32","CB59","CB147_prior","CB149_contextual"):
        raise ValueError("registered model arms changed")
    if (GENERATION_INDICES!=(4,5,8,9,12,13) or
            len(set(GENERATION_INDICES))!=6 or GENERATION_TOKENS!=32):
        raise ValueError("free-generation prompts changed")
    if len(CB149_SHA)!=64 or len(CB147_SHA)!=64:
        raise ValueError("unfrozen codebook SHA")
    if any(k<0 or k>=len(NEW_PROMPTS) for k in GENERATION_INDICES):
        raise ValueError("not original established diagnostic corpus")
    if ROW_BEGIN<=415:
        raise ValueError("validation includes v146 or v148 already inspected questions")
    return True
