"""Functional ground-truth, fluency and CPU. No KL, no NLL or teacher matching."""
import time
import torch
from .protocol import VALID,TRAIN,TOKENS,example,EOS

@torch.inference_mode()
def assess(model,pairs=VALID):
    model.eval()
    rows=[]
    for color,obj in pairs:
        case=example(color,obj)
        generated=model.generate(torch.tensor(case["context"],dtype=torch.long))
        answer=" ".join(TOKENS[t] for t in generated if t!=EOS)
        expected=" ".join(TOKENS[t] for t in case["answer"] if t!=EOS)
        rows.append({
            "prompt":case["query"],
            "gold":expected,
            "answer":answer,
            "raw_generated_ids":generated,
            "exact":generated==case["answer"],
            "grammar":(
                len(generated)==6 and generated[0]==4 and
                generated[2]==5 and generated[4]==6 and
                generated[-1]==EOS),
            "completed":bool(generated and generated[-1]==EOS),
        })
    n=len(rows)
    return {
        "samples":n,
        "exact_correct":sum(row["exact"] for row in rows),
        "correct_rate":sum(row["exact"] for row in rows)/n,
        "grammar_rate":sum(row["grammar"] for row in rows)/n,
        "eos_rate":sum(row["completed"] for row in rows)/n,
        "rows":rows,
    }

def paired_speed(models,repeat=3):
    """Same CPU, same actual autoregressive work; alternating order."""
    sample=example(*VALID[0])
    for model in models.values():
        model.generate(torch.tensor(sample["context"]))
    durations={key:[] for key in models}
    orders=[list(models),list(reversed(models))]
    with torch.inference_mode():
        for trial in range(repeat):
            for name in orders[trial%2]:
                begin=time.perf_counter()
                for color,obj in VALID:
                    case=example(color,obj)
                    models[name].generate(torch.tensor(case["context"]))
                durations[name].append(time.perf_counter()-begin)
    return {
        name:{
            "total_seconds":sum(values),
            "examples_per_second":len(VALID)*repeat/sum(values),
            "repetitions":repeat,
            "prompt_count":len(VALID),
        }
        for name,values in durations.items()
    }
