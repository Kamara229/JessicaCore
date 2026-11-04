"""Score genuine first assistant turn; gold answers and open free text."""
import re
import torch
from v108.generation import build_chat
from v94.generation import generate_one
from .protocol import VALID,OPEN

def simplify(t):
    return re.sub(r"[^\w]+"," ",t).strip().casefold()
@torch.inference_mode()
def assess(model,tok):
    report={"gold":[],"open":[]}
    for q,ref in VALID:
        y=generate_one(model,tok,build_chat(tok,q),stop_at_eos=True)
        text=y["text"].strip()
        eos=tok.eos_token_id in y["ids"]
        correct=simplify(text)==simplify(ref)
        report["gold"].append({
            "question":q,"expected":ref,"actual":text[:500],
            "correct":correct,"eos":eos,"strict":correct and eos,
            "generated_ids":y["ids"]})
    for q in OPEN:
        y=generate_one(model,tok,build_chat(tok,q),stop_at_eos=True)
        ids=[x for x in y["ids"] if x!=tok.eos_token_id]
        report["open"].append({
            "question":q,"actual":y["text"][:500],
            "eos":tok.eos_token_id in y["ids"],
            "distinct_token_ratio":len(set(ids))/max(1,len(ids)),
            "generated_ids":y["ids"],
            "semantic_quality_requires_human_check":True})
    report["score"]={
        "exact_factual":sum(x["correct"] for x in report["gold"]),
        "strict_correct_and_EOS":sum(x["strict"] for x in report["gold"]),
        "factual_questions":len(report["gold"]),
        "open_examples":len(report["open"])}
    return report

def speed_single(model,tok,paid,attach,restore):
    prompts=(VALID[0][0],OPEN[0])
    durations={"CB59":[],"new_head":[]}
    for repeat in range(2):
        for question in prompts:
            order=("CB59","new_head") if repeat%2==0 else ("new_head","CB59")
            for name in order:
                changed=attach(model,paid) if name=="new_head" else None
                try:
                    result=generate_one(model,tok,build_chat(tok,question),stop_at_eos=False)
                    durations[name].append(result["elapsed_seconds"])
                finally:
                    if changed is not None:restore(model,changed)
    rates={k:32*len(v)/sum(v) for k,v in durations.items()}
    return {"tokens_per_second":rates,
            "trained_over_CB59":rates["new_head"]/rates["CB59"],
            "paired_runs":4,"fixed_tokens_including_prefill":32,
            "not_a_native_big_model_speed_pass":True}
