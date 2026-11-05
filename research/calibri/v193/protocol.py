"""Precommitted controlled source-only original model test; no CALIBRI compression."""
OLD_MODEL="HuggingFaceTB/SmolLM2-135M-Instruct"
OLD_REVISION="12fd25f77366fa6b3b4b768ec3050bf629380bac"
OLD_UNIQUE=134_515_008
NEW_MODEL="HuggingFaceTB/SmolLM2-360M-Instruct"
NEW_REVISION="a10cc1512eabd3dde888204e902eca88bddb4951"
NEW_UNIQUE=361_821_120
GRADER_VERSION="v192-locked-grader-v1"
GRADER_SHA256="e230e480c02c680ae02d17b0825b451cbf76b138ceb743f181cc6fb62ad4ac1f"
DOMAIN_COUNTS={"arithmetic":24,"opposite_yesno":24,"fictional_binding":24,"factual_short":24}
COUNT=96
CAP=32
assert sum(DOMAIN_COUNTS.values())==COUNT
