"""Actual user answers, independently true facts, EOS, diversity; not KL."""
import re
import torch
from v108.generation import build_chat
from v94.generation import generate_one
from .protocol import VALID,OPEN
def norm(x):
    return re.sub(r"[^\w]+"," ",x).strip().casefold()
@torch.inference_mode()
def assess(model,tok):
    out={"factual":[],"open":[]}
    for q,gold in VALID:
        answer=generate_one(model,tok,build_chat(tok,q),stop_at_eos=True)
        text=answer["text"].strip()
        eos=tok.eos_token_id in answer["ids"]
        exact=norm(text)==norm(gold)
        out["factual"].append({"question":q,"reference_truth":gold,
              "generated":text[:500],"eos":eos,
              "correct":exact,"strict_correct_and_eos":exact and eos,
              "token_ids":answer["ids"]})
    for q in OPEN:
        answer=generate_one(model,tok,build_chat(tok,q),stop_at_eos=True)
        tokens=[x for x in answer["ids"] if x!=tok.eos_token_id]
        out["open"].append({"question":q,"generated":answer["text"][:600],
              "eos":tok.eos_token_id in answer["ids"],
              "distinct_token_ratio":len(set(tokens))/max(1,len(tokens)),
              "human_semantic_grade_pending":True})
    out["summary"]={"correct":sum(x["correct"] for x in out["factual"]),
                    "strict_correct_and_eos":sum(x["strict_correct_and_eos"]
                          for x in out["factual"]),"total_gold":len(VALID)}
    return out

def speed(model,tok,paid,attach,restore):
    times={"CB59":[],"joint":[]}
    for repeat in range(2):
        for q in (VALID[0][0],OPEN[0]):
            order=("CB59","joint") if repeat%2==0 else ("joint","CB59")
            for name in order:
                changed=attach(model,paid) if name=="joint" else None
                try:
                    y=generate_one(model,tok,build_chat(tok,q),stop_at_eos=False)
                    times[name].append(y["elapsed_seconds"])
                finally:
                    if changed is not None:restore(model,changed)
    rates={name:32*len(v)/sum(v) for name,v in times.items()}
    return {"token_per_second":rates,
            "joint_vs_frozen_original_CB59":rates["joint"]/rates["CB59"],
            "paired_benchmarks":4}
