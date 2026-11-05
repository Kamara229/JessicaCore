"""CALIBRI v180: frozen orthogonal block transformations of actual FP32 weights."""
from v175.protocol import SOURCE_SHA, SOURCE_BYTES, TOTAL_UNIQUE
from v177.protocol import SMOKE_IDS, SPEED_IDS
import math
ARMS = ("H32","H64")
BLOCKS = {"H32":32,"H64":64}
SEED = 180
MAX_BYTES = math.floor(0.700 * TOTAL_UNIQUE / 8)
TRIAGE_MIN_DISTINCT = 3
TRIAGE_MIN_FP32_MATCH = 2
REDUCED_QA_CASE_IDS = tuple(SMOKE_IDS) + (4,5,6,7)
assert len(set(REDUCED_QA_CASE_IDS)) == 24
