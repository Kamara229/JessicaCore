"""v191 read-only exploratory analysis on v190 public QA raw artifacts."""
PRIOR_RUN=42092017343
ARTIFACT_NAME="calibri-v190-original-135M-vs-360M-new-public-balanced-64QA"
ARTIFACT_FILE="v190_original_135M_vs_360M_new_public_64QA_complete.json"
ARMS=("135M","360M")
EXPECTED_PER_ARM=64
DOMAINS=("arithmetic","opposite_yesno","fictional_binding","factual_short")
assert len(DOMAINS)*16==EXPECTED_PER_ARM
