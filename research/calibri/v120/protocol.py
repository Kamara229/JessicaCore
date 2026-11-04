"""V120 preregistration: a model-agnostic ~10MB neural critic prototype.

It cannot create facts not in source or rescue an information-destroyed
CB59 model by just polishing text. Target is correctness of response
selection, not old teacher KL or weights.
"""
from v118.protocol import FIT as TRAIN_FACTS
from v118.protocol import VALID as VALID_A
from v119.protocol import VALID as VALID_B
from v119.protocol import OPEN as OPEN_B

MAX_CHECKPOINT_BYTES=10_000_000
BYTE_VOCAB=258
MAX_TEXT_BYTES=160
WIDTH=224
LAYERS=4
HEADS=4
FEED_FORWARD=640
SEED=120
OPTIM_STEPS=140
LR=0.0008
BATCH_SIZE=20
CANDIDATE_TOP_K=(1,8,32,128,512)
BASELINE_SOURCE_RUN=41822331451
BASELINE_ARCHIVE_SHA="50eac24ba7d7a452c495f8adced2cdef3f9cc44e024217fc21d8f2cebb87413c"

assert not {q for q,_ in TRAIN_FACTS}&{q for q,_ in VALID_A}
assert not {q for q,_ in TRAIN_FACTS}&{q for q,_ in VALID_B}
assert MAX_CHECKPOINT_BYTES==10_000_000
