"""Exact G44 512 TRAIN set and fixed evaluation sets; no DEV selection."""
import random
from collections import defaultdict
from research.calibri2.fusion_growth.g44_512_joint import suite,SEED,BATCH

def epoch_batches(rows):
    assert len(rows)==512 and all(len(r["g"])==2 for r in rows)
    pairs=list(range(256));random.Random(SEED).shuffle(pairs)
    order=[2*i+j for i in pairs for j in (0,1)]
    for j in range(0,512,BATCH):
        replay=(j//BATCH)%16
        yield order[j:j+BATCH],[rows[i] for i in order[j:j+BATCH]+[2*replay,2*replay+1]]

def microbatches(rows):
    buckets=defaultdict(list)
    for r in rows:buckets[len(r["p"])].append(r)
    for group in buckets.values():
        for i in range(0,len(group),4):yield group[i:i+4]
