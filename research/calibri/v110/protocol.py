"""Preregistered diagnostic probes; NOT TRAIN, VALID or official v89 TEST."""
from binary_geometry_v64 import EXPECTED_UNIQUE, MODEL_SHA
from v86.provenance import ORIGINAL_CB59_SHA

# These prompts are new to v110, for localization only. Do not optimize on them.
PROBES = (
    ("arithmetic", "What is 12 plus 19? Answer with the number.", "31"),
    ("geography", "Which city is the capital of Germany? Answer briefly.", "Berlin"),
    ("science", "Which gas do green plants absorb during photosynthesis?", "carbon dioxide"),
    ("instruction", "Write one short instruction for safely crossing a road.", None),
)
# Explicitly balanced boundary and factor interventions. All share original prefixes.
CUTS = (0, 1, 4, 8, 15, 22, 29, 30)
FACTORS = ("self_attn", "mlp")
BUDGET = .700
CB59_BYTES = 11229303
INT6_BYTES = 101449198

def validate_geometry(n):
    if n != 30:
        raise ValueError(f"expected pinned SmolLM2 30 blocks, received {n}")
    if tuple(sorted(set(CUTS))) != CUTS or CUTS[-1] != n:
        raise AssertionError("invalid predeclared boundaries")
    assert 8 * CB59_BYTES / EXPECTED_UNIQUE < BUDGET
    return {"layers": n, "cuts": CUTS, "model_revision": MODEL_SHA,
            "frozen_cb59_sha256": ORIGINAL_CB59_SHA}
