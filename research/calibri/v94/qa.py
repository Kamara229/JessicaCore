"""Answer correctness only, exact paired original TRAIN-heldout BoolQ and SciQ.
No KL, NLL, probability distribution comparison, or teacher outputs.
"""
from __future__ import annotations
import hashlib,json,random
import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from .provenance import BOOLQ_RANGE,SCIQ_RANGE

DATASET_BOOLQ="google/boolq"
DATASET_SCIQ="allenai/sciq"
ANSWER_LETTERS=("A","B","C","D")

def source_rows(dataset,span):
    revision=HfApi().dataset_info(dataset).sha
    data=load_dataset(dataset,revision=revision,
                      split=f"train[{span[0]}:{span[1]}]")
    if len(data)!=span[1]-span[0]:raise AssertionError("incomplete public original TRAIN-only source")
    return data,{"dataset":dataset,"revision":revision,
        "original_split":"train","row_range":span,"count":len(data)}

def encode_case(tokenizer,prompt,index,gold):
    ids=tokenizer.encode(prompt,add_special_tokens=False)
    if not 1<len(ids)<=600:raise AssertionError("corrupt original prompt length")
    fingerprint=hashlib.sha256(torch.tensor(ids,dtype=torch.int64).numpy().astype("<i8").tobytes()).hexdigest()
    return {"record_index":index,"prompt_SHA":fingerprint,"tokens":ids,"gold":gold}

def build_boolq(tok):
    rows,origin=source_rows(DATASET_BOOLQ,BOOLQ_RANGE)
    yes=tok.encode(" Yes",add_special_tokens=False)
    no=tok.encode(" No",add_special_tokens=False)
    if len(yes)!=1 or len(no)!=1 or yes==no:
        raise AssertionError("cannot constrain real Yes/No tokenizer to one token")
    cases=[]
    for n,row in enumerate(rows):
        snippet=tok.decode(tok.encode(str(row["passage"]),add_special_tokens=False)[:256],skip_special_tokens=True)
        prompt=f"Context: {snippet}\nQuestion: {row['question']}\nAnswer:"
        cases.append(encode_case(tok,prompt,BOOLQ_RANGE[0]+n,0 if row["answer"] else 1))
    return cases,origin,[yes[0],no[0]]

def build_sciq(tok):
    rows,origin=source_rows(DATASET_SCIQ,SCIQ_RANGE)
    ids=[]
    for letter in ANSWER_LETTERS:
        code=tok.encode(" "+letter,add_special_tokens=False)
        if len(code)!=1:raise AssertionError("four genuine answer letters need single token")
        ids.append(code[0])
    if len(set(ids))!=4:raise AssertionError("duplicate candidate answer tokens")
    cases=[]
    for n,row in enumerate(rows):
        idx=SCIQ_RANGE[0]+n
        correct=str(row["correct_answer"])
        options=[correct]+[str(row[f"distractor{i}"]) for i in range(1,4)]
        if len(set(options))!=4:
            raise AssertionError("ambiguous original SciQ MC choices")
        salt=hashlib.sha256(f"CALIBRI_v94_{idx}_{row['question']}".encode()).digest()
        random.Random(int.from_bytes(salt[:8],"little")).shuffle(options)
        gold=options.index(correct)
        support=tok.decode(tok.encode(str(row["support"]),add_special_tokens=False)[:192],skip_special_tokens=True)
        prompt=("Context: "+support+"\nQuestion: "+str(row["question"])+
                "\nOptions:\n"+"\n".join(f"{letter}) {option}" for letter,option in zip(ANSWER_LETTERS,options))+\
                "\nAnswer:")
        cases.append(encode_case(tok,prompt,idx,gold))
    return cases,origin,ids

@torch.inference_mode()
def score_cases(model,cases,option_ids):
    rows=[]
    option=torch.tensor(option_ids,dtype=torch.long)
    for case in cases:
        ids=torch.tensor(case["tokens"],dtype=torch.long)[None,:]
        last=model(input_ids=ids,use_cache=False).logits[0,-1,:]
        pred=int(last[option].argmax().item())
        rows.append({"record_index":case["record_index"],"prompt_SHA":case["prompt_SHA"],
            "gold":case["gold"],"pred":pred,"correct":int(pred==case["gold"])})
    return rows
