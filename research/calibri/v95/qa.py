"""Original ground-truth task questions, identical format across model candidates.

New public original TRAIN-only slices are disjoint from v94 diagnostics.
No teacher distributions, no KL, no original official independent TEST.
"""
from __future__ import annotations
import hashlib,random
from v94.qa import source_rows,encode_case,ANSWER_LETTERS
from v95.spec import SPLITS,verify

def build(tok,splits=("TRAIN",)):
    verify()
    if not splits or any(s not in ("TRAIN","TRAIN_VALID") for s in splits):
        raise AssertionError("only disjoint public TRAIN research splits are permitted")
    yes=tok.encode(" Yes",add_special_tokens=False)
    no=tok.encode(" No",add_special_tokens=False)
    if len(yes)!=1 or len(no)!=1:raise AssertionError("original tokenizer options need one token each")
    results={};sources={}
    choices=[]
    for letter in ANSWER_LETTERS:
        ids=tok.encode(" "+letter,add_special_tokens=False)
        if len(ids)!=1:raise AssertionError("SciQ original options need one token each")
        choices.append(ids[0])
    if len(set(choices))!=4:raise AssertionError("colliding SciQ answer tokens")
    for domain in ("BoolQ","SciQ"):
        results[domain]={}
        for split,span in SPLITS[domain].items():
            if split not in splits:continue
            rows,origin=source_rows("google/boolq" if domain=="BoolQ" else "allenai/sciq",span)
            items=[]
            for index,row in enumerate(rows):
                i=span[0]+index
                if domain=="BoolQ":
                    snippet=tok.decode(tok.encode(str(row["passage"]),add_special_tokens=False)[:256],
                        skip_special_tokens=True)
                    prompt=f"Context: {snippet}\nQuestion: {row['question']}\nAnswer:"
                    gold=0 if row["answer"] else 1
                else:
                    correct=str(row["correct_answer"])
                    options=[correct]+[str(row[f"distractor{j}"]) for j in range(1,4)]
                    if len(set(options))!=4:raise AssertionError("ambiguous original SciQ answer options")
                    salt=hashlib.sha256(f"CALIBRI_v95_{i}_{row['question']}".encode()).digest()
                    random.Random(int.from_bytes(salt[:8],"little")).shuffle(options)
                    gold=options.index(correct)
                    support=tok.decode(tok.encode(str(row["support"]),add_special_tokens=False)[:192],
                        skip_special_tokens=True)
                    prompt=("Context: "+support+"\nQuestion: "+str(row["question"])+
                        "\nOptions:\n"+"\n".join(f"{letter}) {option}" for letter,option in zip(ANSWER_LETTERS,options))+
                        "\nAnswer:")
                items.append(encode_case(tok,prompt,i,gold))
            results[domain][split]=items
            sources[domain+"_"+split]=origin
    return results,sources,{"BoolQ":(yes[0],no[0]),"SciQ":tuple(choices)}
