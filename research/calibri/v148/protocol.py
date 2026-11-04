"""All eval choices fixed BEFORE accessing fresh SciQ/BoolQ VALID rows."""
from v146.protocol import (SCI_Q,SCI_Q_COMMIT,BOOL_Q,BOOL_Q_COMMIT,
                            MAX_PASSAGE_CHARS,MAX_SUPPORT_CHARS)
from v147.codec import COUNT as VOCAB
DOMAINS=("SciQ","BoolQ")
ARMS=("FP32","CB59","CB140_head_half","CB147_int8_paid")
START=400
COUNT=16
END=START+COUNT
SPLIT="validation"
FROZEN_CB147_SHA="639be185c259260c614823831275113ffa2d83231eb54805ebfc67be30da5bbd"
LABELS={"SciQ":("A","B","C","D"),"BoolQ":("Yes","No")}

def verify_protocol():
    if (START,COUNT,END,SPLIT)!=(400,16,416,"validation"):
        raise ValueError("fresh 16+16 validation slice altered")
    if ARMS!=("FP32","CB59","CB140_head_half","CB147_int8_paid"):
        raise ValueError("model arms changed after seeing validation data")
    if VOCAB!=49152 or len(FROZEN_CB147_SHA)!=64:
        raise ValueError("paid source geometry invalid")
    if START in range(200,208):
        raise ValueError("v146 validation questions repeated")
    return True
