"""Prospective host-blind evaluation, explicit independent source split.

Numbers/countries are frozen before model output is generated. The candidate
generator sees ONLY q, not gold. Cross-host weight transfer means unchanged
critic checkpoint; it does not imply successful semantic transfer.
"""
from v120.protocol import MAX_CHECKPOINT_BYTES
from v123.protocol import PYTHIA,PYTHIA_SHA
from v117.protocol import ORIGINAL_CB59_SHA

SEED=124
STEPS=140
LR=.00035
BATCH=18
MAX_NEW_TOKENS_PYTHIA=22
V120_CRITIC_SHA="81dd5ed9ee813843972cc837446fde908fc7d3dd9d2be3122efe7e2bd8ae96aa"
TRAIN_CITIES=(("Finland","Helsinki"),("Denmark","Copenhagen"),("Greece","Athens"),("Ireland","Dublin"),
             ("Italy","Rome"),("Spain","Madrid"),("France","Paris"),("Portugal","Lisbon"),
             ("Poland","Warsaw"),("Austria","Vienna"),("Japan","Tokyo"),("Australia","Canberra"))
VALID_CITIES=(("Norway","Oslo"),("Sweden","Stockholm"),("Germany","Berlin"),("Canada","Ottawa"),
             ("Chile","Santiago"),("India","New Delhi"))
TRAIN_ANSWERS=tuple((f"What is the capital of {c}? Answer briefly.",a) for c,a in TRAIN_CITIES)
VALID_ANSWERS=tuple((f"What is the capital of {c}? Answer briefly.",a) for c,a in VALID_CITIES)

def arithmetic(seedstart,n,op):
    out=[]
    for k in range(n):
        a=seedstart+3*k
        b=2+(7*k)%19
        if op=="plus":
            answer=a+b
            q=f"What is {a} plus {b}? Reply with the number."
        elif op=="minus":
            answer=a-b
            q=f"What is {a} minus {b}? Reply with the number."
        else:
            x=2+(k%7);y=3+((2*k)%9)
            answer=x*y
            q=f"What is {x} multiplied by {y}? Reply with the number."
        out.append((q,str(answer)))
    return tuple(out)

TRAIN=TRAIN_ANSWERS+arithmetic(31,20,"plus")+arithmetic(45,12,"minus")+arithmetic(1,7,"multiply")
VALID=VALID_ANSWERS+arithmetic(141,7,"plus")+arithmetic(198,5,"minus")+(
 ("What is 5 multiplied by 11? Reply with the number.","55"),
 ("What is 9 multiplied by 7? Reply with the number.","63"),
)
assert len(TRAIN)==51 and len(VALID)==20
assert not {q for q,_ in TRAIN}&{q for q,_ in VALID}
assert len(set(q for q,_ in TRAIN))==len(TRAIN)
assert len(set(q for q,_ in VALID))==len(VALID)
