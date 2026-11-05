"""Frozen 360M complete original-source 0.700-bpp compact rate-first control.

This is NOT a semantic compression quality PASS. The baseline CB59
per-8-weight K32 FP16 codebook representation is rate-first; actual
QA must be measured, including negative results. Distinct 360M source
from earlier smaller 135M CB59 baseline.
"""
MODEL="HuggingFaceTB/SmolLM2-360M-Instruct"
REVISION="a10cc1512eabd3dde888204e902eca88bddb4951"
ORIGINAL_UNIQUE_PARAMETERS=361_821_120
MAX_0P700_PHYSICAL_BYTES=31_659_348
SOURCE_CODEC="CB59_K32V8"
CODEBOOK_ENTRIES=32
WEIGHTS_PER_VECTOR=8
BITS_PER_VECTOR=5
FP16_CODEBOOK=True
FROZEN_ORIGINAL_QA_RUN=42095901870
FROZEN_ORIGINAL_QA_ARTIFACT="calibri-v193-actual-original-135M-vs-360M-frozen-grader-new-96QA"
FROZEN_GRADER_SHA="e230e480c02c680ae02d17b0825b451cbf76b138ceb743f181cc6fb62ad4ac1f"
SMOKE_IDS=(0,1,2,3,24,25,26,27,48,49,50,51,72,73,74,75)
CAP=32
assert MAX_0P700_PHYSICAL_BYTES==ORIGINAL_UNIQUE_PARAMETERS*7//80
assert len(SMOKE_IDS)==16 and len(set(SMOKE_IDS))==16
