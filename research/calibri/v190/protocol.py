"""Pre-registered new public balanced behavior QA, no source fitting."""
from binary_geometry_v64 import MODEL,MODEL_SHA,EXPECTED_UNIQUE
OLD_MODEL=MODEL
OLD_SHA=MODEL_SHA
OLD_UNIQUE=EXPECTED_UNIQUE
NEW_MODEL="HuggingFaceTB/SmolLM2-360M-Instruct"
NEW_SHA="a10cc1512eabd3dde888204e902eca88bddb4951"
NEW_UNIQUE=361821120
SEED=190
MAX_GENERATION=32
DOMAINS=("arithmetic","opposite_yesno","fictional_binding","factual_short")
DOMAIN_SIZE=16
TOTAL=len(DOMAINS)*DOMAIN_SIZE
EXPECTED_PUBLIC_HISTORICAL=192
assert TOTAL==64
assert OLD_UNIQUE==134515008 and NEW_UNIQUE>OLD_UNIQUE
assert NEW_SHA!=OLD_SHA and len(NEW_SHA)==40
