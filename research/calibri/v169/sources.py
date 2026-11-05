"""Synthetic new controlled questions: balanced pairs, factual answers, no logits.

No private TEST or original FP32 answer key leaks into either model input.
All tasks and prompt formats are fixed before archived sources are opened.
"""
from dataclasses import dataclass
from hashlib import sha256
from .protocol import DOMAINS,DOMAIN_COUNTS,FORMATS,check

@dataclass(frozen=True)
class Question:
    id:int
    domain:str
    question:str
    gold:str
    answer_kind:str
    pair_id:str

def digest(s):return sha256(s.encode("utf8")).hexdigest()
def questions():
    check()
    out=[]
    def add(domain,q,gold,kind,pair=""):
        if not q or not gold:raise ValueError("bad behavioral case")
        out.append(Question(len(out),domain,q,str(gold),kind,pair))
    # Diverse arithmetic: no answer choices and no provided answer.
    for i in range(16):
        a=7+((i*19)%87); b=3+((i*13)%29)
        mode=i%4
        if mode==0: q=f"Calculate {a} plus {b}."; ans=a+b
        elif mode==1: q=f"Calculate {a} minus {b}."; ans=a-b
        elif mode==2: q=f"Calculate {a} times {b}."; ans=a*b
        else:
            # exact quotient; do not ask a question with a fractional key
            x=a*b
            q=f"Calculate {x} divided by {b}.";ans=a
        add("arithmetic",q,ans,"number")
    # Unseen new names and symmetric swapped-world binding.
    names=(("Kavi","Sorin"),("Lemi","Poria"),("Dario","Vena"),
           ("Alin","Faro"),("Sira","Miro"),("Talin","Bora"),
           ("Rivo","Zela"),("Bina","Orin"),("Naro","Tesi"),
           ("Varo","Lina"),("Zivo","Pera"),("Nimi","Saro"),
           ("Kena","Taro"),("Fira","Bavi"),("Tena","Loro"),
           ("Zira","Mena"))
    colors=("red","blue","green","yellow","purple","orange","black","white")
    for j,(a,b) in enumerate(names):
        first=colors[(j*3)%len(colors)]
        second=colors[(j*3+1)%len(colors)]
        for side in (0,1):
            left,right=(first,second) if side==0 else (second,first)
            add("binding",f"In a fictional story, {a} has a {left} notebook, and {b} has a {right} notebook. What color is {a}'s notebook?",left,"color",f"b_{j}")
    # Both yes and no on every symmetric pair; avoids v167 all-Yes proxy.
    for j,(a,b) in enumerate(names[:8]):
        color=colors[j]
        other=colors[(j+1)%len(colors)]
        for side in (0,1):
            left,right=(color,other) if side==0 else (other,color)
            add("yesno",f"In a fictional story, {a} has a {left} notebook, and {b} has a {right} notebook. Does {a} have a {color} notebook?","Yes" if side==0 else "No","yesno",f"yn_{j}")
    # 16 everyday reference questions; external validation not needed.
    facts=(
        ("What is the capital of France?","Paris","word"),
        ("What is the capital of Japan?","Tokyo","word"),
        ("What is the capital of Italy?","Rome","word"),
        ("What is the capital of Germany?","Berlin","word"),
        ("What is the capital of Spain?","Madrid","word"),
        ("What is the capital of Canada?","Ottawa","word"),
        ("What is the chemical formula for water?","H2O","word"),
        ("What is the chemical symbol for oxygen?","O","word"),
        ("How many months are in a year?","12","number"),
        ("How many days are in a week?","7","number"),
        ("What planet is known as the red planet?","Mars","word"),
        ("Which star does Earth orbit?","Sun","word"),
        ("What gas do humans need to breathe?","oxygen","word"),
        ("How many sides does a square have?","4","number"),
        ("How many centimeters are in a meter?","100","number"),
        ("What is the chemical symbol for gold?","Au","word"),
    )
    for q,gold,kind in facts:add("factual",q,gold,kind)
    # Fictional ordering facts, not memorization of named entities.
    people=(("Jora","Navi","Kiri"),("Pano","Tora","Miki"),
            ("Nela","Favi","Bero"),("Lero","Cavi","Zena"),
            ("Rena","Zavi","Horo"),("Fina","Soli","Mara"),
            ("Yaro","Beli","Nemi"),("Xena","Vari","Deni"))
    for i,(first,second,third) in enumerate(people):
        add("logic",
            f"In a fictional race, {first} finished before {second}, and {second} finished before {third}. Who was first?",
            first,"name",f"ord_{i}")
        # Balanced numeric comparison, half Yes and half No.
        n=10+i*3
        truth=i%2==0
        a,b=(n+3,n) if truth else (n,n+3)
        add("logic",f"Is {a} greater than {b}?","Yes" if truth else "No","yesno",f"cmp_{i}")
    assert len(out)==96,(len(out),DOMAIN_COUNTS)
    assert [sum(x.domain==d for x in out) for d in DOMAINS]==[DOMAIN_COUNTS[d] for d in DOMAINS]
    assert len(set(x.question for x in out))==96
    assert sum(x.gold=="Yes" for x in out if x.domain=="yesno")==8
    assert sum(x.gold=="No" for x in out if x.domain=="yesno")==8
    return out

def variants(c):
    tail=("Reply with only the final answer." if c.answer_kind not in ("yesno",) else
          "Reply with only Yes or No.")
    brief=("Give a short factual answer, without discussing unrelated information." if c.answer_kind!="yesno" else
           "Answer this yes/no question briefly.")
    return {"direct":c.question+"\n"+tail,"brief":c.question+"\n"+brief}

def all_cases():
    qs=questions()
    cases=[]
    for q in qs:
        texts=variants(q)
        for style in FORMATS:
            prompt=texts[style]
            cases.append({
                "question_id":q.id,"domain":q.domain,"style":style,
                "question":q.question,"gold":q.gold,"kind":q.answer_kind,
                "pair_id":q.pair_id,"prompt":prompt,"prompt_sha":digest(prompt),
            })
    if len(cases)!=192 or len(set(c["prompt_sha"] for c in cases))!=192:
        raise AssertionError("duplicate or missing response prompts")
    return cases
