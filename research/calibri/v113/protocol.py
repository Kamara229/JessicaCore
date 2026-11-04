"""Frozen before v113 training. No official TEST or scored QA prompt reuse."""
from v112.pipeline import MODEL, PINNED_REVISION
MODES=("global","layer_control","role")
# Original TRAIN passages do not overlap v112 VALID; teacher inputs only.
FIT=(
 "The orchard caretaker carried baskets from the barn to the market.",
 "For every integer, multiplying by one gives the same integer.",
 "A telescope gathers light so the observer can see distant objects.",
 "The engine's crankshaft moves when the piston transfers mechanical force.",
 "During autumn, the leaves of maple trees gradually change their color.",
 "Computers store the instructions of a running program in memory.",
)
# Entirely new v113 VALID texts, not used for any book, scale or hyperparameter.
VALID=(
 "The lighthouse keeper spotted a distant ship before the morning fog appeared.",
 "A coral reef contains diverse animals and shelters many young fish.",
 "In a database, an index can accelerate queries on frequently searched columns.",
 "The mountain path became slippery after heavy snow melted in the afternoon.",
 "The bicycle mechanic examined the chain and tightened the loose wheel.",
 "The orchestra rehearsed carefully before performing at the city concert hall.",
)
GENERATE=(
 "The old scientist opened a small notebook and wrote",
 "The hikers discovered a narrow cave behind the waterfall and",
)
SHRINKAGE=0.5
MIN_MULTIPLIER=-2.0
MAX_MULTIPLIER=2.0
