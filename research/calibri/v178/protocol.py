"""Frozen v178 source geometry and QA protocol: no logits/probability fitting."""
import math
from v175.protocol import SOURCE_SHA, SOURCE_BYTES, TOTAL_UNIQUE
from v177.protocol import SPEED_IDS, SMOKE_IDS

MAX_BYTES = math.floor(0.700 * TOTAL_UNIQUE / 8)
SEED = 178
BITS = 5
GROUP = 8
ENTRIES = 32
COLUMN_CLAMP = (0.25, 4.0)
SAMPLES = 4096
KMEANS_ITERATIONS = 5
EARLY_GATE_UNIQUE = 3
EARLY_GATE_FP32_FIRST_TOKEN_AGREEMENT = 2
REDUCED_QA = 24
FULL_QA = 96
