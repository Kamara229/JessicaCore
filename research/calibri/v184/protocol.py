"""Frozen CALIBRI v184: one different jointly trainable physical source."""
import math
from v175.protocol import SOURCE_SHA, SOURCE_BYTES, TOTAL_UNIQUE
from v177.protocol import SMOKE_IDS
from binary_geometry_v64 import MODEL, MODEL_SHA

RANK=2
INT8_MAX=127
SEED=184
STEPS=16
LR_ALPHA=0.015
LR_BETA=0.0025
MAX_GRAD_NORM=1.0
MAX_BYTES=math.floor(0.700*TOTAL_UNIQUE/8)
HEADROOM=MAX_BYTES-SOURCE_BYTES
TRAIN_CASES=12
EVAL_IDS=(0,1,2,3,4,5,6,7,16,17,18,19,48,49,50,51,64,65,66,67,80,81,82,83)
SPEED_IDS=(0,16,64,80)
assert len(EVAL_IDS)==24 and len(set(EVAL_IDS))==24
assert RANK==2 and SOURCE_SHA=="50eac24ba7d7a452c495f8adced2cdef3f9cc44e024217fc21d8f2cebb87413c"
assert 500_000<HEADROOM<550_000
