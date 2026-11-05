"""v181 intentionally ABOVE-budget recovery oracle, NOT a candidate <=0.7 bpp."""
from v175.protocol import SOURCE_SHA, SOURCE_BYTES, TOTAL_UNIQUE
from v177.protocol import SMOKE_IDS, SPEED_IDS
GROUPS={"ORACLE_V2":2,"ORACLE_V1":1}
K=32
BITS=5
SEED=181
CASE_IDS=tuple(SMOKE_IDS)+(4,5,6,7)
assert len(set(CASE_IDS))==24
