"""Precommitted balanced contradiction-aware 5bit-group code training."""
from v175.protocol import SOURCE_SHA,SOURCE_BYTES,TOTAL_UNIQUE
from v184.protocol import MAX_BYTES
from v185.protocol import CASE_IDS,SPEED_IDS,SMOKE_IDS
SEED=186
STEPS=96
EPOCHS=4
TRAIN_PAIR_COUNT=24
VALID_PAIR_COUNT=12
CENTROID_LR=0.006
MARGIN=1.5
PAIR_MARGIN_WEIGHT=0.20
MAX_GRAD_NORM=1.
# Fixed physical 5-bit K32/V8 projection constraints inherited from v185.
MAX_INDEX_SWAPS_PER_STEP=512
TRAIN_DOMAINS=("yesno_numeric","yesno_binding","arithmetic","binding")
SCHEDULE=tuple(
    [f"model.layers.{i}.self_attn.o_proj.weight" for i in range(30)]
    +[f"model.layers.{i}.mlp.down_proj.weight" for i in range(30)]
    +[f"model.layers.{i}.mlp.gate_proj.weight" for i in range(30)]
    +[f"model.layers.{i}.self_attn.v_proj.weight" for i in (0,3,9,15,22,29)]
)
assert len(SCHEDULE)==STEPS==TRAIN_PAIR_COUNT*EPOCHS
assert SOURCE_BYTES==11229303 and MAX_BYTES==11770063
assert len(CASE_IDS)==24 and len(SMOKE_IDS)==20
assert MAX_INDEX_SWAPS_PER_STEP==512
