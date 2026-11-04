"""Predetermined tiny English QA with disjoint color/object combinations.

This is a narrow compositional benchmark. Its context contains one fact,
and the autoregressive model predicts the answer token by token from the
textual context. A PASS does NOT imply open-domain LLM language quality.
"""
import hashlib
import random
import torch

COLORS=("red","blue","green","yellow","white","black")
OBJECTS=("bird","fish","car","kite","boat","cup")
TOKENS=("<pad>","<bos>","<ans>","<eos>","the","is",".","what","color","of","?")+COLORS+OBJECTS
STOI={word:i for i,word in enumerate(TOKENS)}
EOS=STOI["<eos>"]
MODES=("dense","shared","shared_adapter")
SEED=114
STEPS=500
BATCH=64
LR=.009
D=32
DEPTH=3
RANK=10
ADAPTER_RANK=2

def split():
    ordered=sorted(
        (hashlib.sha256(f"{c}/{o}/114".encode()).digest(),c,o)
        for c in COLORS for o in OBJECTS)
    return [(c,o) for _,c,o in ordered[:28]],[(c,o) for _,c,o in ordered[28:]]

TRAIN,VALID=split()
assert len(TRAIN)==28 and len(VALID)==8 and set(TRAIN).isdisjoint(VALID)

def example(color,obj):
    text=f"<bos> the {obj} is {color} . what color is the {obj} ? <ans>"
    answer=f"the {obj} is {color} . <eos>"
    return {
        "context":[STOI[t] for t in text.split()],
        "answer":[STOI[t] for t in answer.split()],
        "query":text,
    }

def batch(examples,n,generator):
    rows=[example(c,o) for c,o in generator.choices(examples,k=n)]
    context=torch.tensor([r["context"] for r in rows],dtype=torch.long)
    answers=torch.tensor([r["answer"] for r in rows],dtype=torch.long)
    previous=torch.cat(
        [torch.full((n,1),STOI["<ans>"]),answers[:,:-1]],dim=1)
    return context,previous,answers
