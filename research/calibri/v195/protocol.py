"""Frozen original source control / bounds for v195, no BPP quality PASS."""
from v194.protocol import (
    MODEL,REVISION,ORIGINAL_UNIQUE_PARAMETERS,MAX_0P700_PHYSICAL_BYTES,
    FROZEN_ORIGINAL_QA_RUN,FROZEN_GRADER_SHA)
SOURCE_SHA256="69e2e0f67c1e642ae468a94fc28047a7b1b77afe91776cb539e58e53aaef56a6"
SOURCE_BYTES=29_348_539
FROZEN_V194_RUN=42097753160
QA_IDS=(0,1,24,25,48,49,72,73)
NATIVE_LINEAR_PATHS=(
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.15.mlp.down_proj.weight",
    "model.layers.31.mlp.gate_proj.weight",
)
NATIVE_EMBED_ROWS=(0,1,2,100,1000,40000)
ARMS=(
    "fp32_embed_head_only",
    "fp32_attention_only",
    "fp32_mlp_only",
    "fp32_transformer_only",
    "fp32_embed_head_attention",
    "fp32_embed_head_mlp",
)
TOKEN_CAP=32
assert SOURCE_BYTES<=MAX_0P700_PHYSICAL_BYTES
assert len(SOURCE_SHA256)==64
assert len(QA_IDS)==8 and len(set(QA_IDS))==8
