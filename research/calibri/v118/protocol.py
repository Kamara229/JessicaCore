"""Frozen ground-truth TRAIN/VALID protocol; no old TEST or teacher mimicry."""
from v117.protocol import (
    SOURCE_BYTES, TOTAL_UNIQUE, MAX_SOURCE_BYTES, ORIGINAL_CB59_SHA, roles
)

HEAD_RANK=2
ROLE_RANK=1
HEAD_DIM=576
VOCAB=49152
HEADER=b"CLB118\x00\x00"
SEED=118
TRAIN_STEPS=140
LR=.012
WEIGHT_DECAY=.001
MAX_NEW_TOKENS=28
# Real separate ground truth examples, never fitted from original LM logits.
FIT=(
    ("What is 14 plus 8? Reply with the number.", "22"),
    ("What is 11 plus 23? Reply with the number.", "34"),
    ("What is 40 minus 17? Reply with the number.", "23"),
    ("What is 7 multiplied by 6? Reply with the number.", "42"),
    ("What is 9 plus 12? Reply with the number.", "21"),
    ("What is 8 multiplied by 4? Reply with the number.", "32"),
    ("What is 19 minus 6? Reply with the number.", "13"),
    ("What is 24 divided by 6? Reply with the number.", "4"),
    ("What is 5 multiplied by 5? Reply with the number.", "25"),
    ("What is 6 plus 18? Reply with the number.", "24"),
    ("What is the capital of Italy? Reply briefly.", "Rome"),
    ("What is the capital of France? Reply briefly.", "Paris"),
    ("What is the capital of Spain? Reply briefly.", "Madrid"),
    ("What is the capital of Japan? Reply briefly.", "Tokyo"),
    ("What is the capital of Canada? Reply briefly.", "Ottawa"),
    ("What is the capital of Australia? Reply briefly.", "Canberra"),
    ("What is the capital of Norway? Reply briefly.", "Oslo"),
    ("What is the capital of Egypt? Reply briefly.", "Cairo"),
    ("Which planet is known as the red planet? Reply briefly.", "Mars"),
    ("What chemical symbol represents gold? Reply briefly.", "Au"),
    ("What chemical symbol represents iron? Reply briefly.", "Fe"),
    ("What is the chemical formula of water? Reply briefly.", "H2O"),
    ("What gas do plants absorb during photosynthesis? Reply briefly.", "Carbon dioxide"),
    ("What gas do people need for breathing? Reply briefly.", "Oxygen"),
    ("What does a thermometer measure? Reply briefly.", "Temperature"),
    ("Which organ pumps blood? Reply briefly.", "The heart"),
    ("What is the opposite of cold? Reply with one word.", "Hot"),
    ("What is the opposite of empty? Reply with one word.", "Full"),
    ("What is the opposite of slow? Reply with one word.", "Fast"),
    ("What is the opposite of dark? Reply with one word.", "Light"),
    ("Is seven greater than five? Reply Yes or No.", "Yes"),
    ("Is three greater than nine? Reply Yes or No.", "No"),
    ("Does a triangle have three sides? Reply Yes or No.", "Yes"),
    ("Does a square have five sides? Reply Yes or No.", "No"),
)
VALID=(
    ("What is 18 plus 9? Reply with the number.", "27"),
    ("What is 32 minus 14? Reply with the number.", "18"),
    ("What is 3 multiplied by 8? Reply with the number.", "24"),
    ("What is the capital of Portugal? Reply briefly.", "Lisbon"),
    ("What is the capital of Sweden? Reply briefly.", "Stockholm"),
    ("What is the capital of India? Reply briefly.", "New Delhi"),
    ("What gas do plants release during photosynthesis? Reply briefly.", "Oxygen"),
    ("What is the largest planet in our solar system? Reply briefly.", "Jupiter"),
    ("Which organ controls the body and thought? Reply briefly.", "The brain"),
    ("What is the opposite of early? Reply with one word.", "Late"),
    ("Is ten less than two? Reply Yes or No.", "No"),
    ("Does a hexagon have six sides? Reply Yes or No.", "Yes"),
)
OPEN=(
    "Write one clear sentence explaining why plants need sunlight.",
    "Give two practical tips for safely crossing a busy road.",
    "Describe a kind robot helping an injured bird in two sentences.",
    "Explain briefly how rain forms.",
)
# Prevent silently reusing the formal independent v89 test.
assert len(FIT)==34 and len(VALID)==12 and len(OPEN)==4
assert not ({q for q,_ in FIT}&{q for q,_ in VALID})
assert MAX_SOURCE_BYTES-SOURCE_BYTES==540_760

def factor_data_bytes():
    return (sum((rows+cols)*ROLE_RANK*2 for _,rows,cols in roles())+
            2*(VOCAB*HEAD_RANK+HEAD_RANK*HEAD_DIM+VOCAB))
assert factor_data_bytes()==493056
