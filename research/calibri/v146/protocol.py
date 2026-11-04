"""Immutable planned choice evaluation; not the frozen independent v89 TEST.

All four model arms are frozen BEFORE any SciQ or BoolQ validation rows are
loaded. We use one fixed contiguous region per source; no model selection.
"""
SCI_Q="allenai/sciq"
SCI_Q_COMMIT="2c94ad3"  # Hugging Face immutable 2024 Parquet conversion commit
BOOL_Q="google/boolq"
BOOL_Q_COMMIT="a78586b41ee7ad5529450607e6515d337e6acddd"
SPLIT="validation"
START=200
COUNT=8
END=START+COUNT

DOMAINS=("SciQ","BoolQ")
ARMS=("FP32","CB59","CB140_full","CB140_head_half")
SELECTED_GAINS={"CB59":(0.0,0.0),"CB140_full":(1.0,1.0),
                "CB140_head_half":(0.0,0.5)}
SCI_Q_LABELS=("A","B","C","D")
BOOL_Q_LABELS=("Yes","No")
MAX_SUPPORT_CHARS=1200
MAX_PASSAGE_CHARS=1800

def validate():
    if COUNT!=8 or SPLIT!="validation" or START!=200 or END!=208:
        raise AssertionError("changed planned QA-validation sample")
    if ARMS!=("FP32","CB59","CB140_full","CB140_head_half"):
        raise AssertionError("changed preregistered model comparison")
    if SELECTED_GAINS["CB140_head_half"]!=(0.0,0.5):
        raise AssertionError("gain reselected after v145")
    return True
