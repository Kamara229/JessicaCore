"""Correct FIRST subtoken diagnostic; never call it a full-answer score."""
import torch
from v108.generation import build_chat
from v94.generation import generate_one

def prompt_input(tok,q):
    rendered=build_chat(tok,q)
    ids=tok.encode(rendered,add_special_tokens=False)
    return torch.tensor([ids],dtype=torch.long),rendered

def gold_first_token(tok,gold):
    options=tok.encode(" "+gold,add_special_tokens=False)
    if not options:raise ValueError("correct reference lacks representable gold subtoken")
    return options[0]

def score(model,src,target,alternate):
    with torch.no_grad():
        logit=model(input_ids=src,use_cache=False).logits[0,-1].float()
        rank=1+int((logit>logit[target]).sum().item())
        alt_rank=1+int((logit>logit[alternate]).sum().item())
        return {"target_first_subtoken_rank":rank,
                "alt_unspaced_first_subtoken_rank":alt_rank,
                "target_first_subtoken_is_top1":bool(logit.argmax().item()==target),
                "target_first_subtoken_probability":float(torch.softmax(logit,dim=-1)[target]),
                "predicted_token":int(logit.argmax()),
                "target_subtoken":target,"other_gold_subtoken":alternate}

def answer(model,tok,prompt):
    with torch.no_grad():
        item=generate_one(model,tok,prompt,stop_at_eos=True)
    return {"text":item["text"][:450],
            "token_ids":item["ids"],
            "reached_eos":bool(item["eos_generated"]),
            "elapsed_seconds":item["elapsed_seconds"],
            "full_decoding_steps":item["full_model_token_steps"]}
