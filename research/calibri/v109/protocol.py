"""Frozen before encoding. Labeled TRAIN holdouts never enter the encoder."""
from v108.spec import ALL_PROMPTS

# Reuse v108 for paired historical regression; score new prompts separately.
FRESH = (
    ("arithmetic", "Compute 17 minus 8. Give the result.", ("9",)),
    ("arithmetic", "What is six multiplied by seven? Give the result.", ("42",)),
    ("geography", "Name the capital of Portugal.", ("Lisbon",)),
    ("geography", "Name the capital of Kenya.", ("Nairobi",)),
    ("science", "What is the chemical symbol for oxygen?", ("O",)),
    ("science", "Which organ pumps blood around the human body?", ("heart",)),
    ("instruction", "Give one way to keep a paper map dry on a rainy walk.", ()),
    ("creative", "Write one sentence about a kite caught in an oak tree.", ()),
)
PROMPTS = {"v108_regression": ALL_PROMPTS, "v109_new": FRESH}
QA_SPANS = {"boolq": (9100, 16), "sciq": (10080, 16)}
TIMING = (FRESH[0][1], FRESH[-1][1])
