"""Immutable experimental protocol before reading any QA answer.
We deliberately try the PREVIOUSLY UNCHANGED physical code indices.
"""
from v175.protocol import SOURCE_BYTES,SOURCE_SHA,TOTAL_UNIQUE
from v184.protocol import MAX_BYTES
from v177.protocol import SMOKE_IDS
SEED=185
STEPS=24
LEARNING_RATE_CODEBOOKS=0.008
WEIGHT_GRAD_MAX_NORM=1.0
MAX_INDEX_UPDATES_PER_STEP=512
MIN_SCORE_IMPROVEMENT=0.04
SOURCE_GUARD_STRENGTH=0.12
PUBLIC_TRAIN_INDICES=(0,2,4,6,8,16,18,20,22,24,48,49,50,51,52,53)
CASE_IDS=(0,1,2,3,4,5,6,7,16,17,18,19,48,49,50,51,64,65,66,67,80,81,82,83)
SPEED_IDS=(0,16,64,80)
# Each iteration uses a deterministic target operator in an early,
# middle or late transformer block; this is not a stochastic
# post-hoc choice from evaluation results.
STEPS_OPERATORS=tuple(
    f"model.layers.{layer}.self_attn.o_proj.weight" if idx % 2==0
    else f"model.layers.{layer}.mlp.down_proj.weight"
    for idx,layer in enumerate((0,0,1,1,2,2,3,3,5,5,8,8,
                                10,10,12,12,15,15,20,20,25,25,29,29))
)
assert STEPS==len(STEPS_OPERATORS)==24
assert len(PUBLIC_TRAIN_INDICES)==16
assert len(CASE_IDS)==24 and len(set(CASE_IDS))==24
assert MAX_BYTES==11770063 and SOURCE_BYTES==11229303
