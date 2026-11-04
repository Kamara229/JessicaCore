"""True and deliberately corrupted TRAIN candidates; no evaluation leakage."""
import random
from .protocol import TRAIN_FACTS,SEED

def make_train_pairs():
    r=random.Random(SEED)
    source=list(TRAIN_FACTS)
    pairs=[]
    for i,(q,good) in enumerate(source):
        other=r.choice([a for k,a in source if k!=q and a!=good])
        # Correct answers are provided ONLY as labeled training signals.
        pairs.append((q,good,1))
        pairs.append((q,other,0))
        pairs.append((q,"Yes Yes Yes Yes Yes",0))
        pairs.append((q,"2020 2020 2020",0))
        pairs.append((q,"",0))
    r.shuffle(pairs)
    return pairs

def validation_pairs(cases):
    # Negative control must not be the real expected gold for the same q.
    return [(q,gold,1) for q,gold in cases] + [
        (q,"Yes Yes Yes Yes Yes",0) for q,gold in cases] + [
        (q,"",0) for q,gold in cases]
