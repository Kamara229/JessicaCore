"""Frozen reuse of physically packed 135M v186, not a new FP32 model.

All 211 source matrices' FP16-centroids get model-wide gradients.
The complete 49,152-token output distribution is a TRAIN teacher
signal at chosen positions (NOT evaluating all literal language words).
Actual QA accepts only whole generated answers, not training losses.
"""
from .policy import POLICY,suggested_cycles
from v186.protocol import SCHEDULE as ORIGINAL_LAYER_SCHEDULE
from v175.protocol import TOTAL_UNIQUE,SOURCE_BYTES
from v184.protocol import MAX_BYTES
from binary_geometry_v64 import MODEL,MODEL_SHA
INITIAL_135M_SHA256="933daaecf9011f77e3705308faab9d6074b5fb7c4c847d5e1bdb830ad47b965a"
INITIAL_TRAINED_ARTIFACT_RUN=42085087419
INITIAL_ARCHIVE_NAME="CALIBRI_v186_balanced_5bit_full_source.cb59"
CODEBOOK_MATRIX_COUNT=211
VOCAB_SIZE=49152
SEED=196
TEXT_INDICES=(0,4,8,12,16,20,24,28)
INSTRUCTION_INDICES=(32,33,34,35,36,37,38,39,48,49,50,51)
PUBLIC_TRAIN_INDICES=TEXT_INDICES+INSTRUCTION_INDICES
TRAIN_PAIRS=24
VALID_PAIRS=12
DEV_HISTORICAL_IDS=(0,1,2,3,4,5,6,7,16,17,18,19,48,49,50,51,64,65,66,67,80,81,82,83)
FROZEN_SOURCE_135M_UNIQUE=134_515_008
assert len(ORIGINAL_LAYER_SCHEDULE)==POLICY.steps_per_cycle==96
assert POLICY.cycles==2
assert len(PUBLIC_TRAIN_INDICES)==20 and len(set(PUBLIC_TRAIN_INDICES))==20
assert TOTAL_UNIQUE==FROZEN_SOURCE_135M_UNIQUE
assert SOURCE_BYTES==11_229_303 and MAX_BYTES==11_770_063
assert suggested_cycles(FROZEN_SOURCE_135M_UNIQUE)==2
