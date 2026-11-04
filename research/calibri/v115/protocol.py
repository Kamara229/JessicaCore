"""Prospective v115 splits / seeds, frozen before runs."""
import hashlib
from v114.protocol import COLORS,OBJECTS,example
SPLITS=("v115_a","v115_b")
SEEDS=(115,116,117)
TRAINING_STEPS=500
BATCH=64
LR=.009
MAX_SEQUENCE=6

def partition(salt):
    if salt not in SPLITS:
        raise ValueError("unknown preregistered salt")
    enumerated=sorted(
        (hashlib.sha256(f"{salt}/{color}/{obj}".encode()).digest(),color,obj)
        for color in COLORS for obj in OBJECTS)
    train=[(color,obj) for _,color,obj in enumerated[:28]]
    heldout=[(color,obj) for _,color,obj in enumerated[28:]]
    if len(train)!=28 or len(heldout)!=8 or set(train)&set(heldout):
        raise AssertionError("FIT/VALID leakage")
    return train,heldout
