"""Pre-registered v177 matrix-wise variable-bit budgets, source-only quantizer."""
import math
from v175.protocol import SOURCE_BYTES, SOURCE_SHA, TOTAL_UNIQUE
from v174.protocol import family

ARMS = ("ADAPT_A", "ADAPT_B")
MAX_BYTES = math.floor(0.700 * TOTAL_UNIQUE / 8)
SEED = 177
SPEED_IDS = (0, 16, 64, 80)
SMOKE_IDS = (0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 64, 65, 66, 67, 80, 81, 82, 83)
# Both arms reserve six bits for the tied input-embedding/LM head.
# A protects attention; B protects the MLP down-projection instead.
# Both pay for this by reducing 60 gate/up MLP matrices to four bits.
def allocation(arm, name):
    if arm not in ARMS:
        raise ValueError("Unknown frozen allocation")
    fam = family(name)
    if fam == "embedding_head":
        return 6
    if fam == "attention":
        return 6 if arm == "ADAPT_A" else 5
    if name.endswith((".gate_proj.weight", ".up_proj.weight")):
        return 4
    if name.endswith(".down_proj.weight"):
        return 5 if arm == "ADAPT_A" else 6
    raise ValueError("Unclassified trained matrix: " + name)
