"""Frozen new ground-truth VALID, not reused for optimization."""
from v118.protocol import FIT
TRAIN_SELECT=(0,1,10,11,20,21,30,31)
STEPS=8
SEED=119
LR=.003
VALID=(
    ("What is 21 plus 16? Reply with the number.","37"),
    ("What is 52 minus 31? Reply with the number.","21"),
    ("What is 9 multiplied by 3? Reply with the number.","27"),
    ("What is the capital of New Zealand? Reply briefly.","Wellington"),
    ("What is the capital of Brazil? Reply briefly.","Brasilia"),
    ("What is the capital of Belgium? Reply briefly.","Brussels"),
    ("What chemical symbol represents sodium? Reply briefly.","Na"),
    ("Which planet is closest to the Sun? Reply briefly.","Mercury"),
    ("What gas makes up most of Earth's atmosphere? Reply briefly.","Nitrogen"),
    ("What is the opposite of high? Reply with one word.","Low"),
    ("Is fifteen less than twenty? Reply Yes or No.","Yes"),
    ("Does a pentagon have six sides? Reply Yes or No.","No"),
)
OPEN=(
    "Explain in one sentence how bees help flowers.",
    "Describe a child building a bridge from wooden blocks.",
    "Give two ways to save clean drinking water.",
    "Explain why metal feels cold when touched.",
)
assert STEPS==len(TRAIN_SELECT) and len(VALID)==12 and len(OPEN)==4
assert len(set(TRAIN_SELECT))==len(TRAIN_SELECT)
assert not {x for x,_ in VALID}&{x for x,_ in FIT}
