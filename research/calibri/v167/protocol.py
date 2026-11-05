"""Predeclared behavioral QA: REAL answers and CPU speed, never KL/NLL gates.

This is source-only research, not the protected independent v89 TEST.
All weights frozen, all arms see the same questions, strict physical size.
"""
from v146.protocol import SCI_Q,SCI_Q_COMMIT,BOOL_Q,BOOL_Q_COMMIT
from v117.protocol import ORIGINAL_CB59_SHA,TOTAL_UNIQUE,MAX_SOURCE_BYTES

ARMS=("FP32","CB59","CB161")
DOMAINS=("SciQ","BoolQ","Arithmetic","Binding","Logic")
ROW_START=800
ROW_COUNT=96
SYNTHETIC_COUNT=32
PHYSICAL_BYTES=11292174
PARENT_RUN=41920207168
GEN_TOKENS=32

def counts():
    return {"SciQ":ROW_COUNT,"BoolQ":ROW_COUNT,
            "Arithmetic":SYNTHETIC_COUNT,"Binding":SYNTHETIC_COUNT,
            "Logic":SYNTHETIC_COUNT}

def check():
    if ARMS!=("FP32","CB59","CB161") or DOMAINS!=("SciQ","BoolQ","Arithmetic","Binding","Logic"):
        raise AssertionError("arm/domain changed")
    if (ROW_START,ROW_COUNT,SYNTHETIC_COUNT,PHYSICAL_BYTES,GEN_TOKENS)!=(800,96,32,11292174,32):
        raise AssertionError("predeclared dimensions changed")
    if ROW_START<720 or ROW_START+ROW_COUNT>1000:
        raise AssertionError("reuse or invalid SciQ validation")
    if 8*PHYSICAL_BYTES/TOTAL_UNIQUE>=0.7:
        raise AssertionError("physically paid archive exceeds original weight cap")
    return True
