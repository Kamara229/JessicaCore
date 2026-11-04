"""Pre-registered 8-bank source-only encoding, no heldout TEST optimization."""
BANKS=8
VOCAB=49152
DIM=576
K=32
V=8
SAMPLES_PER_BANK=4096
ROW_PART=VOCAB//BANKS
GENERATE_TOKENS=32
PROBES=(
 ("arithmetic","Question: What is fourteen plus five? Answer:"),
 ("factual","Question: What is the capital of Italy? Answer:"),
 ("instruction","Give one sentence about how to wash your hands before eating:"),
 ("creative","Write a single sentence about the river after rain:"),
)
assert VOCAB%BANKS==0 and ROW_PART==6144 and DIM%V==0
