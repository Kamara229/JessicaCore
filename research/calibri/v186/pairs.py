"""Frozen train/validation opposite-response pairs, disjoint from 96 DEV.

No development case, answer annotation, model logits or heldout
validation signal influences sample generation or optimization.
"""
import hashlib
import torch
from . import protocol as p


def raw_pairs(split):
    if split not in ("TRAIN","VALID"):
        raise ValueError("Public data split must be frozen before model evaluation")
    count=6 if split=="TRAIN" else 3
    offset=0 if split=="TRAIN" else 41
    out=[]
    def append(kind,first,second):
        assert first[0]!=second[0] and first[1]!=second[1]
        out.append({"domain":kind,"positive":first,"negative":second,
                    "pair_index":len(out),"split":split})
    for idx in range(count):
        j=offset+idx
        n=403+17*j
        append("yesno_numeric",
            (f"Is {n+3} greater than {n}? Reply with only Yes or No.", "Yes"),
            (f"Is {n} greater than {n+3}? Reply with only Yes or No.", "No"))
    for idx in range(count):
        j=offset+idx
        owner=f"Tomi{split}{j}";other=f"Vesa{split}{j}"
        append("yesno_binding",
            (f"In a fictional story, {owner} carries a silver book and {other} carries a gold book. Does {owner} carry a silver book? Reply with only Yes or No.","Yes"),
            (f"In a fictional story, {owner} carries a gold book and {other} carries a silver book. Does {owner} carry a silver book? Reply with only Yes or No.","No"))
    for idx in range(count):
        j=offset+idx
        a=354+9*j;b=4+j%5
        append("arithmetic",
            (f"What is {a} plus {b}? Reply with only the number.",str(a+b)),
            (f"What is {a} plus {b+200}? Reply with only the number.",str(a+b+200)))
    for idx in range(count):
        j=offset+idx
        owner=f"Reni{split}{j}";other=f"Zumi{split}{j}"
        append("binding",
            (f"In a fictional story, {owner} has a red folder and {other} has a blue folder. What color is {owner}'s folder? Reply with one word.","red"),
            (f"In a fictional story, {owner} has a blue folder and {other} has a red folder. What color is {owner}'s folder? Reply with one word.","blue"))
    expected=p.TRAIN_PAIR_COUNT if split=="TRAIN" else p.VALID_PAIR_COUNT
    if len(out)!=expected:
        raise AssertionError("Unexpected split balance")
    if sum(x["domain"]=="yesno_numeric" for x in out)!=count:
        raise AssertionError("Numeric contradiction class drift")
    if sum(x["domain"]=="yesno_binding" for x in out)!=count:
        raise AssertionError("Contextual contradiction class drift")
    return out


def tokenize_one(tok,question,answer):
    if tok.eos_token_id is None:
        raise ValueError("Model has no EOS")
    prompt=tok.apply_chat_template(
        [{"role":"user","content":question}],
        tokenize=False,add_generation_prompt=True)
    prefix=tok.encode(prompt,add_special_tokens=False)
    reply=tok.encode(answer,add_special_tokens=False)
    if not prefix or not reply:
        raise ValueError("Missing chat prefix or answer tokens")
    ids=torch.tensor(prefix+reply+[tok.eos_token_id],dtype=torch.long)
    if not 6<=len(ids)<96:
        raise AssertionError("Paired example exceeds fixed training context")
    pos=torch.arange(len(prefix)-1,len(ids)-1,dtype=torch.long)
    target=ids[pos+1].clone()
    if int(target[0])!=reply[0] or int(target[-1])!=tok.eos_token_id:
        raise AssertionError("Wrong first-answer/EOS next-token positions")
    return {"ids":ids,"positions":pos,"targets":target,
        "prompt_ids":torch.tensor(prefix,dtype=torch.long),
        "prompt_sha":hashlib.sha256(prompt.encode()).hexdigest(),
        "question":question,"answer":answer,
        "first_expected_id":reply[0]}


def build_pairs(tok,split,development_questions=()):
    raw=raw_pairs(split)
    forbidden=set(development_questions)
    if any(q in forbidden for pair in raw for q,_ in
           (pair["positive"],pair["negative"])):
        raise AssertionError("Known public DEV question reused for source fit")
    result=[]
    for pair in raw:
        left=tokenize_one(tok,*pair["positive"])
        right=tokenize_one(tok,*pair["negative"])
        if left["first_expected_id"]==right["first_expected_id"]:
            raise AssertionError(f"Contrasting {split} {pair['domain']} pair {pair['pair_index']} has same answer first token: {pair['positive'][1]} / {pair['negative'][1]}")
        result.append({"split":split,"domain":pair["domain"],
            "pair_id":pair["pair_index"],"left":left,"right":right})
    return result


def manifest(pairs):
    return [{"pair_id":x["pair_id"],"split":x["split"],
        "domain":x["domain"],
        "questions":[x[s]["question"] for s in ("left","right")],
        "answers":[x[s]["answer"] for s in ("left","right")],
        "prompt_sha256":[x[s]["prompt_sha"] for s in ("left","right")]}
        for x in pairs]
