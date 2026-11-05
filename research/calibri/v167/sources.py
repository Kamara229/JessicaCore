"""Frozen QA construction: balanced answer positions, heterogeneous abilities.

Public VALIDATION examples are disjoint from v163. Synthetic examples are
deterministic/novel and not an independent benchmark; all original outputs
and full prompts are preserved. No QA label used in training.
"""
from dataclasses import dataclass
from hashlib import sha256
import json
from datasets import load_dataset
from huggingface_hub import HfApi
from .protocol import (SCI_Q,SCI_Q_COMMIT,BOOL_Q,BOOL_Q_COMMIT,
                       DOMAINS,ROW_START,ROW_COUNT,SYNTHETIC_COUNT,check)

@dataclass(frozen=True)
class Case:
    domain: str
    index: int
    prompt: str
    labels: tuple
    gold: int
    pair_id: str
    sha: str
    clipped: bool=False

def digest(s):
    return sha256(s.encode("utf8")).hexdigest()

def case(domain,index,prompt,labels,gold,pair_id="",clipped=False):
    if not prompt or not 0<=gold<len(labels):
        raise ValueError("invalid QA case")
    return Case(domain,index,prompt,tuple(labels),int(gold),pair_id,digest(prompt),clipped)

def sciq(records):
    out=[]
    for i,r in enumerate(records):
        idx=ROW_START+i
        good=str(r["correct_answer"]).strip()
        bad=[str(r[x]).strip() for x in ("distractor1","distractor2","distractor3")]
        if not good or len(set([good]+bad))!=4 or any(not x for x in bad):
            raise ValueError("bad sciq answers")
        gold=idx%4
        choices=bad.copy()
        choices.insert(gold,good)
        support=str(r["support"]).strip()
        clipped=len(support)>1200
        q=(f"Answer this science question.\nContext: {support[:1200]}\n"
           f"Question: {str(r['question']).strip()}\n"
           +"\n".join(f"{j}. {x}" for j,x in zip("ABCD",choices))
           +"\nReply with only A, B, C, or D.")
        out.append(case("SciQ",idx,q,("A","B","C","D"),gold,clipped=clipped))
    return out

def boolq(records):
    out=[]
    for i,r in enumerate(records):
        text=str(r["passage"]).strip()
        question=str(r["question"]).strip()
        if type(r["answer"]) is not bool or not text or not question:
            raise ValueError("bad BoolQ validation")
        q=(f"Read the passage and answer the question.\n"
           f"Passage: {text[:1800]}\nQuestion: {question}\n"
           "Reply with only Yes or No.")
        out.append(case("BoolQ",ROW_START+i,q,("Yes","No"),
                        0 if r["answer"] else 1,clipped=len(text)>1800))
    return out

def math_cases():
    out=[]
    for i in range(SYNTHETIC_COUNT):
        a=9+((i*17)%79)
        b=3+((i*11)%37)
        mode=i%3
        value=(a+b if mode==0 else a-b if mode==1 else a*b)
        question=(f"Compute {a} + {b}." if mode==0 else
                  f"Compute {a} - {b}." if mode==1 else
                  f"Compute {a} * {b}.")
        distractors=[value+1,value-2,value+7]
        if len(set([value]+distractors))!=4:
            raise AssertionError("duplicate arithmetic choice")
        good_idx=i%4
        options=distractors.copy()
        options.insert(good_idx,value)
        prompt=(question+" Choose the correct answer.\n"+
                "\n".join(f"{lab}. {v}" for lab,v in zip("ABCD",options))+
                "\nReply with only A, B, C, or D.")
        out.append(case("Arithmetic",i,prompt,("A","B","C","D"),good_idx))
    return out

def binding_cases():
    out=[]
    names=(("Mira","Niko"),("Tara","Beni"),("Lina","Oren"),("Dara","Pavi"),
           ("Sena","Viko"),("Rina","Fedo"),("Nela","Karo"),("Fina","Yaro"),
           ("Vera","Ludo"),("Zita","Nemo"),("Raya","Bato"),("Niva","Olan"),
           ("Kira","Vaso"),("Lora","Bino"),("Yana","Peno"),("Dima","Zoro"))
    colors=("blue","red","green","yellow")
    for j,(a,b) in enumerate(names):
        first=colors[j%4]
        second=colors[(j+1)%4]
        options=colors[j%4:]+colors[:j%4]
        for side in (0,1):
            x,y=(first,second) if side==0 else (second,first)
            prompt=(f"In an imaginary story, {a} has the {x} badge and "
                    f"{b} has the {y} badge. What color is {a}'s badge?\n"+
                    "\n".join(f"{lab}. {v}" for lab,v in zip("ABCD",options))+
                    "\nReply with only A, B, C, or D.")
            out.append(case("Binding",j*2+side,prompt,("A","B","C","D"),
                            options.index(x),pair_id=f"binding_{j}"))
    return out

def logic_cases():
    out=[]
    names=("Ada","Boris","Cora","Deni")
    for i in range(SYNTHETIC_COUNT):
        ordered=names[i%4:]+names[:i%4]
        question=(f"In a fictional race, {ordered[0]} finished before "
                  f"{ordered[1]}, and {ordered[1]} finished before "
                  f"{ordered[2]}. Who finished first?\n"+
                  "\n".join(f"{letter}. {n}" for letter,n in zip("ABCD",names))+
                  "\nReply with only A, B, C, or D.")
        out.append(case("Logic",i,question,("A","B","C","D"),
                        names.index(ordered[0])))
    return out

def load_all():
    check()
    data={}
    provenance={}
    api=HfApi()
    for domain,dataset_id,revision,formatter in (
        ("SciQ",SCI_Q,SCI_Q_COMMIT,sciq),
        ("BoolQ",BOOL_Q,BOOL_Q_COMMIT,boolq)):
        sha=api.dataset_info(dataset_id,revision=revision).sha
        if not sha.lower().startswith(revision.lower()):
            raise ValueError("dataset upstream revision changed")
        rows=load_dataset(dataset_id,revision=sha,
                          split=f"validation[{ROW_START}:{ROW_START+ROW_COUNT}]")
        if len(rows)!=ROW_COUNT:
            raise ValueError("frozen sample size not available")
        data[domain]=formatter(rows)
        provenance[domain]={"repo":dataset_id,"revision":sha,
                           "split":"validation","begin":ROW_START,
                           "end":ROW_START+ROW_COUNT,
                           "clipped_count":sum(x.clipped for x in data[domain])}
    data["Arithmetic"]=math_cases()
    data["Binding"]=binding_cases()
    data["Logic"]=logic_cases()
    for domain in DOMAINS:
        expected=ROW_COUNT if domain in ("SciQ","BoolQ") else SYNTHETIC_COUNT
        if len(data[domain])!=expected or len({x.sha for x in data[domain]})!=expected:
            raise AssertionError(f"missing or duplicated {domain} questions")
    for domain in ("Arithmetic","Binding","Logic"):
        provenance[domain]={"source":"deterministic_synthetic_not_independent",
                            "predeclared_cases":len(data[domain])}
    return data,provenance

def generation_probes():
    """Free text rather than only next-choice; exact grade where possible."""
    out=[]
    for a,b in ((7,5),(19,8),(26,17),(55,12),(13,29),(31,24),(42,7),(64,14)):
        out.append(("arithmetic",f"Calculate {a} + {b}. Reply with the result.",str(a+b),""))
    for q,g in (("What is the capital of France?","Paris"),
                ("What is the chemical symbol for oxygen?","O"),
                ("How many days are in a week?","7"),
                ("What planet is known as the red planet?","Mars"),
                ("What is the chemical formula for water?","H2O"),
                ("How many sides does a triangle have?","3"),
                ("What is the capital of Japan?","Tokyo"),
                ("What is 9 multiplied by 6?","54")):
        out.append(("factual",q+" Reply briefly.",g,""))
    for x in binding_cases()[:16]:
        # Exact lexical answer independent of arbitrary A/B/C/D option letters.
        import re
        m=re.search(r"has the (\w+) badge",x.prompt)
        if not m:raise ValueError("binding gold text missing")
        word=m.group(1)
        q=x.prompt.split("\n")[0]+" Reply with the color as one word."
        out.append(("binding",q,word,x.pair_id))
    for q,g in (("If A is before B and B before C, which is first?","A"),
                ("A room has four chairs. Two are removed. How many remain?","2"),
                ("What number comes after 19?","20"),
                ("Is 10 greater than 8? Reply Yes or No.","Yes"),
                ("Is 3 greater than 9? Reply Yes or No.","No"),
                ("Complete the sequence: 2, 4, 6, ?","8"),
                ("What is half of 14?","7"),
                ("If today is Monday, what day is tomorrow?","Tuesday")):
        out.append(("reasoning",q,g,""))
    for q in ("Describe how to wash your hands in two steps.",
              "Explain how to save electricity in one sentence.",
              "Give safe advice for crossing the street.",
              "Explain what recycling means in one sentence."):
        out.append(("instruction",q,None,""))
    for q in ("Write a short story about a curious robot.",
              "Describe a forest in one sentence.",
              "Invent a friendly animal character.",
              "Continue: A fox discovered a strange box."):
        out.append(("creative",q,None,""))
    if len(out)!=48:
        raise AssertionError("free-response task count changed")
    return [{"id":i,"domain":dom,"question":q,"gold":gold,
             "pair_id":pid,"sha":digest(q)} for i,(dom,q,gold,pid) in enumerate(out)]
