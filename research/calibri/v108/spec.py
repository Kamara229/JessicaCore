"""Fresh original-chat synthetic QA prompts, never official v89 TEST.

Written before observing v108 output. Do NOT fit/tune v106 thresholds
using any result from this preregistered diagnostic run.
"""
from __future__ import annotations

GOLD_PROMPTS=(
    ("arithmetic","What is 12 plus 19? Answer with the number.",("31",)),
    ("arithmetic","What is half of 46? Answer with the number.",("23",)),
    ("arithmetic","What is 9 times 4? Answer with the number.",("36",)),
    ("geography","What is the capital of Spain? Answer with the city.",("Madrid",)),
    ("geography","What is the capital of Greece? Answer with the city.",("Athens",)),
    ("science","Which planet is closest to the Sun? Answer with the name.",("Mercury",)),
)
OPEN_PROMPTS=(
    ("instruction","Give one useful step to switch off an unused light."),
    ("instruction","Explain one safe way to cross a quiet street."),
    ("creative","Write a short sentence about a blue bicycle beside a lake."),
)
ALL_PROMPTS=tuple((name,prompt,keys) for name,prompt,keys in GOLD_PROMPTS)+tuple(
    (name,prompt,()) for name,prompt in OPEN_PROMPTS)
GENERATED_TOKENS=32
assert len(ALL_PROMPTS)==9
assert len({x[1] for x in ALL_PROMPTS})==len(ALL_PROMPTS)
