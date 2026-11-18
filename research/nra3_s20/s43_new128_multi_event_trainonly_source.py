"""S43: separately committed NEW128 TRAIN-only multi-event Russian source graphs.

Predefined 8 lexical families x 16 constructions, including TWO distinct
predicate sources, multiple REJECTED recipients per predicate, passive,
infinitive, negation of event (not recipient), nonlocal recipient, and
quoted surface mentions. These are new synthetic TRAIN, NOT S38 control
or original TRAIN256/S39 TRAIN80. Offline annotations never enter forward.

S38 immutable adversarial challenge is never imported here. This data
source is deliberately curated in light of S42's shortcut failure, so
any held construction measurements on it are development, not independent
blind model evaluation. One source may have 2 recipients; matching one-to-
one is no longer a valid inference assumption.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from research.nra3_s18.s18c3_raw_russian_grammar_cues import raw_surface_grammar
from research.nra3_s20.s22_new256_counterfactual_train_graph_source import text_only_inputs
from research.nra3_s20.s39_new80_trainonly_scope_graph_source import build_train80

STAGE="NRA3_S43_NEW128_TRAIN_ONLY_MULTIEVENT_RUSSIAN_PREDICATE_SCOPE_GRAPH"

PATTERNS=(
    "finite_local_rejected",
    "finite_fronted_rejected",
    "short_passive_local",
    "short_passive_fronted",
    "infinitive_complement_local",
    "infinitive_fronted",
    "one_predicate_two_rejected_recipients",
    "two_events_two_recipients",
    "passive_and_finite_two_events",
    "infinitive_and_finite_two_events",
    "quoted_mention_plus_real_rejected",
    "only_quoted_negative_mention",
    "negative_pronoun_nonlocal",
    "negated_event_not_rejected_participant",
    "contrast_alternative_recipient",
    "quoted_mention_after_true_event",
)

@dataclass(frozen=True)
class Lex:
    agent:str
    past:str
    infinitive:str
    passive:str
    document:str
    person_a:str
    person_b:str
    other_agent:str
    other_past:str
    other_document:str

LEXICA=(
    Lex("Ирина","направила","направить","направлено","уведомление","проводнику","инспектору","Денис","передал","план"),
    Lex("Леонид","отправил","отправить","отправлено","заключение","бригадиру","редактору","Олег","вручил","чертёж"),
    Lex("Зоя","передала","передать","передано","сообщение","экскурсоводу","помощнику","Павел","отправил","отчёт"),
    Lex("Глеб","доставил","доставить","доставлено","предписание","ревизору","смотрителю","Игорь","передал","эскиз"),
    Lex("Катя","вручила","вручить","вручено","поручение","кладовщику","рецензенту","Матвей","направил","список"),
    Lex("Борис","показал","показать","показано","изображение","конструктору","наблюдателю","Никита","передал","график"),
    Lex("Вера","выслала","выслать","выслано","приглашение","методисту","оператору","Артур","отправил","договор"),
    Lex("Роман","представил","представить","представлено","пояснение","экспедитору","архивариусу","Фёдор","доставил","акт"),
)

@dataclass(frozen=True)
class TrainGraph:
    text:str
    family:int
    construction:int
    event_sources:tuple[int,...]
    rejected_edges:tuple[tuple[int,int],...]
    quoted_token_indices:tuple[int,...]
    semantic_events_count:int
    rejected_edges_count:int

def make_row(f:Lex,family:int,pattern:int)->TrainGraph:
    """Labels are explicit TRAIN annotations; no inference uses them."""
    events=[]
    edges=[]
    feminine=family in (0,2,4,6)
    decide="решила" if feminine else "решил"
    want="хотела" if feminine else "хотел"
    read="прочитала" if feminine else "прочитал"
    say="произнесла" if feminine else "произнёс"
    if pattern==0:
        text=f"{f.agent} {f.past} {f.document} не {f.person_a}."
        events=[f.past]
        edges=[(f.past,f.person_a)]
    elif pattern==1:
        text=f"Не {f.person_a} {f.agent} {f.past} {f.document}."
        events=[f.past]
        edges=[(f.past,f.person_a)]
    elif pattern==2:
        text=f"{f.document.capitalize()} {f.passive} не {f.person_a}."
        events=[f.passive]
        edges=[(f.passive,f.person_a)]
    elif pattern==3:
        text=f"Не {f.person_a} {f.document} {f.passive}."
        events=[f.passive]
        edges=[(f.passive,f.person_a)]
    elif pattern==4:
        text=f"{f.agent} {decide} {f.infinitive} {f.document} не {f.person_a}."
        events=[decide,f.infinitive]
        edges=[(f.infinitive,f.person_a)]
    elif pattern==5:
        text=f"Не {f.person_a} {f.agent} хотел {f.infinitive} {f.document}."
        events=[want,f.infinitive]
        edges=[(f.infinitive,f.person_a)]
    elif pattern==6:
        text=f"{f.agent} {f.past} {f.document} не {f.person_a} и не {f.person_b}."
        events=[f.past]
        edges=[(f.past,f.person_a),(f.past,f.person_b)]
    elif pattern==7:
        text=f"{f.agent} {f.past} {f.document} не {f.person_a}, а {f.other_agent} {f.other_past} {f.other_document} не {f.person_b}."
        events=[f.past,f.other_past]
        edges=[(f.past,f.person_a),(f.other_past,f.person_b)]
    elif pattern==8:
        text=f"{f.document.capitalize()} {f.passive} не {f.person_a}, а {f.other_agent} {f.other_past} {f.other_document} не {f.person_b}."
        events=[f.passive,f.other_past]
        edges=[(f.passive,f.person_a),(f.other_past,f.person_b)]
    elif pattern==9:
        text=f"{f.agent} {decide} {f.infinitive} {f.document} не {f.person_a}, а {f.other_agent} {f.other_past} {f.other_document} не {f.person_b}."
        events=["решил",f.infinitive,f.other_past]
        edges=[(f.infinitive,f.person_a),(f.other_past,f.person_b)]
    elif pattern==10:
        text=f"{f.agent} {read} фразу «не {f.person_b}» и {f.past} {f.document} не {f.person_a}."
        events=[read,f.past]
        edges=[(f.past,f.person_a)]
    elif pattern==11:
        text=f"{f.agent} {say} выражение «не {f.person_a}»."
        events=[say]
        edges=[]
    elif pattern==12:
        text=f"Никому из отдела {f.agent} не {f.past} {f.document}."
        events=[f.past]
        edges=[(f.past,"Никому")]
    elif pattern==13:
        text=f"{f.agent} не {f.past} {f.document} {f.person_a}."
        events=[f.past]
        edges=[]
    elif pattern==14:
        text=f"{f.agent} {f.past} {f.document} {f.person_a}, но не {f.person_b}."
        events=[f.past]
        edges=[(f.past,f.person_b)]
    elif pattern==15:
        text=f"{f.agent} {f.past} {f.document} не {f.person_a} и {say} фразу «не {f.person_b}»."
        events=[f.past,say]
        edges=[(f.past,f.person_a)]
    else: raise ValueError("unsupported S43 construction")

    tok=raw_surface_grammar(text)
    def locate(form):
        matches=[i for i,x in enumerate(tok) if x.form==form]
        if len(matches)!=1:raise ValueError(
            f"S43 TRAIN-only token alignment ambiguous: {form} {text} {matches}")
        return matches[0]
    source=tuple(locate(name) for name in events)
    pairs=tuple((locate(a),locate(b)) for a,b in edges)
    depth=0; quoted=[]
    for i,x in enumerate(tok):
        if x.form=="«":depth+=1
        elif x.form=="»":depth-=1
        elif depth>0:quoted.append(i)
    if depth!=0:raise ValueError("unbalanced quotation in S43 TRAIN")
    if not set(i for i,j in pairs).issubset(set(source)):
        raise ValueError("TRAIN link source not annotated event predicate")
    if len(set(source))!=len(source) or len(set(pairs))!=len(pairs):
        raise ValueError("duplicated event/pair TRAIN annotation")
    return TrainGraph(text,family,pattern,source,pairs,tuple(quoted),
                      len(source),len(pairs))

def build_train128():
    return tuple(make_row(f,i,p) for i,f in enumerate(LEXICA)
                 for p in range(len(PATTERNS)))

def train_source_sha(rows=None):
    if rows is None:rows=build_train128()
    data=[{"text":r.text,"family":r.family,"group":r.construction,
           "events":r.event_sources,"pairs":r.rejected_edges,
           "quoted":r.quoted_token_indices} for r in rows]
    return hashlib.sha256(json.dumps(data,sort_keys=True,
                 ensure_ascii=False,separators=(",",":")).encode("utf-8")).hexdigest()

def train_partitions():
    rows=build_train128()
    result=[]
    for family in range(8):
        held=tuple(i for i,r in enumerate(rows) if r.family==family)
        tr=tuple(i for i,r in enumerate(rows) if r.family!=family)
        result.append(("LEXICAL",family,tr,held))
    for group in range(16):
        held=tuple(i for i,r in enumerate(rows) if r.construction==group)
        tr=tuple(i for i,r in enumerate(rows) if r.construction!=group)
        result.append(("CONSTRUCTION",group,tr,held))
    return tuple(result)

def validate_source():
    rows=build_train128()
    assert len(rows)==128 and len({x.text for x in rows})==128
    old256=set(text_only_inputs())
    old80={r.text for r in build_train80()}
    assert not ({r.text for r in rows}&(old256|old80))
    assert len(PATTERNS)==16 and len(LEXICA)==8
    assert sum(len(r.rejected_edges)==2 for r in rows)==32
    assert sum(len(r.rejected_edges)==0 for r in rows)==16
    assert sum(len(r.rejected_edges)==1 for r in rows)==80
    assert sum(len(r.event_sources)>=2 for r in rows)>0
    for r in rows:
        assert len(raw_surface_grammar(r.text))<=40
    folds=train_partitions()
    assert len(folds)==24
    for axis,_,train,held in folds:
        assert not set(train)&set(held)
        assert set(train)|set(held)==set(range(128))
        assert len(train)==(112 if axis=="LEXICAL" else 120)
        assert len(held)==(16 if axis=="LEXICAL" else 8)
    return {"stage":STAGE,"count":128,"group_count":16,
        "lexical_families":8,"TRAIN_positive_directed_edges":
          sum(len(r.rejected_edges) for r in rows),
        "TRAIN_true_event_source_tokens":sum(len(r.event_sources) for r in rows),
        "TRAIN_two_rejected_edges_per_sentence":32,
        "TRAIN_zero_rejected_edges_per_sentence":16,
        "grouped_folds":24,"source_sha256":train_source_sha(rows),
        "disjoint_original_TRAIN256_and_S39_TRAIN80":True,
        "S38_reserved_not_imported_or_used":True}

if __name__=="__main__":
    print(json.dumps(validate_source(),sort_keys=True,ensure_ascii=False))
