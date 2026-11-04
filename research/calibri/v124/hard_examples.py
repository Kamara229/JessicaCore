"""Real archived LLM failures as hard negatives + plausible wrong answers.

No oracle or reference answer is passed into the host generator. Truth is
used *only after autonomous generation* for labels and training critic.
"""
import random
import re
from .protocol import SEED

def canonical(text):
    return re.sub(r"[^\w]+"," ",text,flags=re.UNICODE).strip().casefold()

def examples(tasks,generated):
    """Map source q to actual autonomous candidate texts from BOTH hosts."""
    rng=random.Random(SEED)
    rows=[]
    for q,truth in tasks:
        rows.append((q,truth,1,"independently_correct_reference"))
        if re.fullmatch(r"-?\d+",truth):
            number=int(truth)
            other=str(number+(1 if number%2 else -1))
        else:
            pool=[ans for qq,ans in tasks if qq!=q and ans!=truth and not re.fullmatch(r"-?\d+",ans)]
            other=rng.choice(pool) if pool else truth+"x"
        rows.append((q,other,0,"plausible_false_reference"))
        for host in ("CB59_v119","Pythia14M"):
            response=generated[host][q]
            label=int(canonical(response)==canonical(truth))
            rows.append((q,response,label,"actual_generated_"+host))
    return rows

def hard_samples(rows):
    return [r for r in rows if r[3].startswith("actual_generated_")]

def positive_and_negative(rows):
    return sum(r[2] for r in rows),sum(not r[2] for r in rows)
