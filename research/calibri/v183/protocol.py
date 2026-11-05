"""Precommitted seven first-layer operators and 11 experimental arm subsets."""
from v182.protocol import FROZEN_CASE_IDS, MODEL_SOURCE_SHA256, MODEL_SOURCE_BYTES, V1_TOTAL_BPP
from v174.protocol import SOURCE_SHA
SCOPE="model.layers.0."
PROJECTIONS={
    "Q0":"self_attn.q_proj.weight",
    "K0":"self_attn.k_proj.weight",
    "V0":"self_attn.v_proj.weight",
    "O0":"self_attn.o_proj.weight",
    "GATE0":"mlp.gate_proj.weight",
    "UP0":"mlp.up_proj.weight",
    "DOWN0":"mlp.down_proj.weight",
}
OPERATOR_SET={
    "V1_BODY_ONLY":(),
    **{"FP32_"+key:(key,) for key in PROJECTIONS},
    "FP32_ALL_ATTENTION0":("Q0","K0","V0","O0"),
    "FP32_ALL_MLP0":("GATE0","UP0","DOWN0"),
    "FP32_ALL_BLOCK0":tuple(PROJECTIONS),
}
SEED=183
assert len(PROJECTIONS)==7
assert len(OPERATOR_SET)==11
assert V1_TOTAL_BPP>0.7
