"""S39 novel synthetically annotated TRAIN-only Russian event/scope graph source.

This is *new training material*, not S38. S38's immutable 32 adversarial
sentences are entirely reserved and MUST NOT appear in any training
forward, loss, tuning, model selection or regression suite here.

The S39 shadow NRA2-compatible adapter is research-only; no writes to
original NRA2, Calibri43M, Green Tunnel, 616 J7, LIVE or evaluation assets.
Every gold predicate/recipient location here is a TRAIN supervision
annotation only, never an inference argument or hidden-320 feature.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from research.nra3_s18.s18c3_raw_russian_grammar_cues import raw_surface_grammar
from research.nra3_s20.s22_new256_counterfactual_train_graph_source import text_only_inputs

STAGE="NRA3_S39_NOVEL80_TRAIN_ONLY_PREDICATE_NEGATION_SCOPE_PASSIVE_QUOTE"
PATTERNS=(
    "local_not_dative",
    "recipient_fronted",
    "negative_pronoun_nonlocal",
    "scope_with_adverb",
    "short_passive_predicate",
    "infinitive_complement",
    "quoted_negative_mention",
    "negated_event_not_rejected",
    "quoted_mention_after_adverb",
    "contrast_alternative_rejected",
)

@dataclass(frozen=True)
class Lexicon:
    agent: str
    past: str
    passive: str
    infinitive: str
    patient: str
    recipient: str
    alternative: str

FAMILIES=(
    Lexicon("Ася","вручила","вручено","вручить","письмо","архивариусу","водителю"),
    Lexicon("Илья","отправил","отправлено","отправить","извещение","сторожу","мастеру"),
    Lexicon("Елена","показала","показано","показать","чертёж","помощнику","секретарю"),
    Lexicon("Саша","передал","передано","передать","рапорт","диспетчеру","сотруднику"),
    Lexicon("Пётр","направил","направлено","направить","заявление","администратору","контролёру"),
    Lexicon("Валерия","доставила","доставлено","доставить","поручение","руководителю","посреднику"),
    Lexicon("Руслан","выслал","выслано","выслать","сообщение","смотрителю","консультанту"),
    Lexicon("Дарья","передала","передано","передать","приложение","докладчику","эксперту"),
)

@dataclass(frozen=True)
class TrainGraph:
    text: str
    family: int
    pattern: int
    source_predicate_token_index0: int
    potential_recipient_token_index0: int
    rejected_recipient: bool
    source_kind: str
    quoted_target: bool

def make_row(f:Lexicon,family:int,pattern:int)->TrainGraph:
    # Candidate annotated text is created ONLY on the TRAIN-data side.
    if pattern==0:
        text=f"{f.agent} {f.past} {f.patient} не {f.recipient}."
        source,target,positive,kind,quoted=f.past,f.recipient,True,"finite",False
    elif pattern==1:
        text=f"Не {f.recipient} {f.agent} {f.past} {f.patient}."
        source,target,positive,kind,quoted=f.past,f.recipient,True,"finite",False
    elif pattern==2:
        text=f"Никому из команды {f.agent} не {f.past} {f.patient}."
        source,target,positive,kind,quoted=f.past,"Никому",True,"finite",False
    elif pattern==3:
        text=f"{f.agent} {f.past} {f.patient} совсем не {f.recipient}."
        source,target,positive,kind,quoted=f.past,f.recipient,True,"finite",False
    elif pattern==4:
        text=f"{f.patient.capitalize()} {f.passive} не {f.recipient}."
        source,target,positive,kind,quoted=f.passive,f.recipient,True,"passive",False
    elif pattern==5:
        text=f"{f.agent} планировал {f.infinitive} {f.patient} не {f.recipient}."
        source,target,positive,kind,quoted=f.infinitive,f.recipient,True,"infinitive",False
    elif pattern==6:
        text=f"{f.agent} прочитал выражение «не {f.recipient}»."
        source,target,positive,kind,quoted="прочитал",f.recipient,False,"finite",True
    elif pattern==7:
        text=f"{f.agent} не {f.past} {f.patient} {f.recipient}."
        source,target,positive,kind,quoted=f.past,f.recipient,False,"finite",False
    elif pattern==8:
        text=f"В заметке {f.agent} процитировал слова «не {f.recipient}»."
        source,target,positive,kind,quoted="процитировал",f.recipient,False,"finite",True
    elif pattern==9:
        text=f"{f.agent} {f.past} {f.patient} {f.recipient}, но не {f.alternative}."
        source,target,positive,kind,quoted=f.past,f.alternative,True,"finite",False
    else:
        raise ValueError("S39 unsupported TRAIN pattern")
    tokens=raw_surface_grammar(text)
    src=[i for i,t in enumerate(tokens) if t.form==source]
    tgt=[i for i,t in enumerate(tokens) if t.form==target]
    if len(src)!=1 or len(tgt)!=1 or src[0]==tgt[0]:
        raise ValueError("S39 ambiguous TRAIN-side predicate/recipient annotation")
    return TrainGraph(text,family,pattern,src[0],tgt[0],positive,kind,quoted)

def build_train80():
    return tuple(make_row(f,i,p) for i,f in enumerate(FAMILIES)
                 for p in range(len(PATTERNS)))

def source_digest(rows=None):
    if rows is None: rows=build_train80()
    d=[{"text":r.text,"family":r.family,"pattern":r.pattern,
        "src":r.source_predicate_token_index0,
        "recipient":r.potential_recipient_token_index0,
        "rejected":r.rejected_recipient,"source_kind":r.source_kind,
        "quoted":r.quoted_target} for r in rows]
    return hashlib.sha256(json.dumps(
        d,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def partitions():
    rows=build_train80()
    folds=[]
    for f in range(8):
        held=tuple(i for i,r in enumerate(rows) if r.family==f)
        train=tuple(i for i in range(len(rows)) if i not in set(held))
        folds.append(("LEXICAL",f,train,held))
    for p in range(10):
        held=tuple(i for i,r in enumerate(rows) if r.pattern==p)
        train=tuple(i for i in range(len(rows)) if i not in set(held))
        folds.append(("CONSTRUCTION",p,train,held))
    return tuple(folds)

def validate_source():
    r=build_train80()
    assert len(r)==80 and len(set(t.text for t in r))==80
    assert len(FAMILIES)==8 and len(PATTERNS)==10
    assert len(text_only_inputs())==256
    assert not (set(t.text for t in r)&set(text_only_inputs()))
    for row in r:
        t=raw_surface_grammar(row.text)
        assert 2<=len(t)<=30
        assert row.source_predicate_token_index0<len(t)
        assert row.potential_recipient_token_index0<len(t)
        assert row.source_predicate_token_index0!=row.potential_recipient_token_index0
    folds=partitions()
    assert len(folds)==18
    for axis,k,train,held in folds:
        assert set(train).isdisjoint(held)
        assert len(set(train)|set(held))==80
        assert len(train)==(70 if axis=="LEXICAL" else 72)
        assert len(held)==(10 if axis=="LEXICAL" else 8)
    return {"stage":STAGE,"TRAIN_rows":80,"positive":sum(x.rejected_recipient for x in r),
            "negative":sum(not x.rejected_recipient for x in r),
            "families":8,"patterns":10,"internal_folds":len(folds),
            "sha256":source_digest(r),
            "old_TRAIN256_disjoint":True,
            "S38_challenge_source_not_imported_or_used":True}

if __name__=="__main__":
    print(json.dumps(validate_source(),sort_keys=True,ensure_ascii=False))
