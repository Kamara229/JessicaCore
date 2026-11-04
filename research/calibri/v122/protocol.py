"""Predeclared before training: small REAL source tasks and separate evaluations.

No claim of broader language competence from this small study.
v89 formal independent TEST must remain unopened.
"""
import hashlib
from v120.protocol import MAX_CHECKPOINT_BYTES

GREEN_SOURCE_RUN=41824783272
GREEN_SOURCE_SHA="81dd5ed9ee813843972cc837446fde908fc7d3dd9d2be3122efe7e2bd8ae96aa"
V119_SOURCE_RUN=41822331451
SEED=122
STEPS=16
LR=0.001
DEPTH_BASIS=3
ROLE_COUNT=2
MAX_DELTA=0.70
ORIGINAL_LAYER_COUNT=30

# Fresh short supervised targets; not from previous v118 or v119 TRAIN.
TRAIN=(
 ("What is 13 plus 11? Reply with the number.","24"),
 ("What is 17 plus 9? Reply with the number.","26"),
 ("What is 22 minus 8? Reply with the number.","14"),
 ("What is 9 multiplied by 4? Reply with the number.","36"),
 ("What is 30 divided by 5? Reply with the number.","6"),
 ("What is the capital of Denmark? Reply briefly.","Copenhagen"),
 ("What is the capital of Finland? Reply briefly.","Helsinki"),
 ("What is the capital of Greece? Reply briefly.","Athens"),
 ("What is the capital of Argentina? Reply briefly.","Buenos Aires"),
 ("What is the capital of Ireland? Reply briefly.","Dublin"),
 ("What chemical symbol represents silver? Reply briefly.","Ag"),
 ("What chemical symbol represents carbon? Reply briefly.","C"),
 ("Is eight less than twelve? Reply Yes or No.","Yes"),
 ("Is seventeen less than seven? Reply Yes or No.","No"),
 ("Does an octagon have eight sides? Reply Yes or No.","Yes"),
 ("Does a circle have four corners? Reply Yes or No.","No"),
)
VALID=(
 ("What is 16 plus 17? Reply with the number.","33"),
 ("What is 45 minus 19? Reply with the number.","26"),
 ("What is 7 multiplied by 5? Reply with the number.","35"),
 ("What is the capital of Chile? Reply briefly.","Santiago"),
 ("What is the capital of Austria? Reply briefly.","Vienna"),
 ("What is the capital of Switzerland? Reply briefly.","Bern"),
 ("What chemical symbol represents potassium? Reply briefly.","K"),
 ("What is the freezing point of water in Celsius? Reply briefly.","0"),
 ("What is the opposite of quiet? Reply with one word.","Loud"),
 ("What gas do plants use for photosynthesis? Reply briefly.","Carbon dioxide"),
 ("Is five greater than fifteen? Reply Yes or No.","No"),
 ("Does a rectangle have four sides? Reply Yes or No.","Yes"),
)
OPEN=(
 "Describe in two sentences how a bicycle works.",
 "Explain what makes the sky look blue.",
 "Give two tips for learning a new language.",
 "Write three different sentences about a clever fox.",
)
assert len(TRAIN)==STEPS and len(VALID)==12 and len(OPEN)==4
assert not {x for x,_ in TRAIN}&{x for x,_ in VALID}
assert MAX_CHECKPOINT_BYTES==10_000_000
