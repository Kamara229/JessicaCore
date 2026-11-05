"""Original capability comparison precommitted before any 360M generations.
Do not tune any CALIBRI compressed source using historical DEV QA.
"""
from binary_geometry_v64 import MODEL as ORIGINAL_MODEL,MODEL_SHA as ORIGINAL_SHA
OTHER_MODEL="HuggingFaceTB/SmolLM2-360M-Instruct"
STYLE_CHOICES=("direct","brief")
CASE_IDS=tuple(range(96))
EXPECTED_GENERATIONS=192
TOKEN_CAP=32
SOURCE_ARTIFACT_RUN=42088612519
SOURCE_ARTIFACT_NAME="calibri-v188-original-pretrained-FP32-192-actual-DEV-answers"
SOURCE_ARTIFACT_FILE="v188_actual_original_FP32_192QA_complete.json"
SEED=189
assert ORIGINAL_MODEL!=OTHER_MODEL
assert len(CASE_IDS)==96 and EXPECTED_GENERATIONS==192
