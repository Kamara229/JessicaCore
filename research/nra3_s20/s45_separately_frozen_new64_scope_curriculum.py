"""S45 separately frozen 64 synthetic TRAIN-only new scope graph sentences.

All 64 examples are ADDITIONAL TRAIN examples (not evaluation) for
S44's already defined grammar shadow. Eight entirely new lexical
families × eight new constructions, chosen without using S38's protected
32 challenge. S43's 128 source and old TRAIN80 / TRAIN256 are fixed.

Their annotations enter the TRAIN loss ONLY; model/decoder forward
receives plain text-derived feature tensors. New64 results cannot
establish independent Russian language generalization or real 43M input.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json

from research.nra3_s18.s18c3_raw_russian_grammar_cues import raw_surface_grammar
from research.nra3_s20.s22_new256_counterfactual_train_graph_source import text_only_inputs
from research.nra3_s20.s39_new80_trainonly_scope_graph_source import build_train80
from research.nra3_s20.s43_new128_multi_event_trainonly_source import build_train128

STAGE="NRA3_S45_SEPARATELY_COMMITTED_NEW64_TRAIN_ONLY_NONLOCAL_QUOTED_EVENT_SCOPES"
STRUCTURES=(
    "double_rejected_same_event",
    "double_rejected_with_intervening_adverbs",
    "negated_event_plus_only_quoted_mention",
    "fronted_passive_time_adverb",
    "fronted_negative_pronoun_infinitive",
    "contrast_wrong_second_recipient_after_adverb",
    "two_predicates_both_rejected_front_and_local",
    "quoted_false_recipient_then_real_rejected",
)

@dataclass(frozen=True)
class Lexical:
    agent:str
    past:str
    infinitive:str
    passive:str
    document:str
    recipient_a:str
    recipient_b:str
    partner:str
    partner_past:str
    partner_document:str
    feminine:bool

LEXICAL_FAMILIES=(
    Lexical("Алёна","выписала","выписать","выписано","направление","технику","покупателю","Тимур","отправил","чек",True),
    Lexical("Филипп","переслал","переслать","переслано","разрешение","студенту","проводнику","Михаил","вручил","лист",False),
    Lexical("Светлана","отослала","отослать","отослано","предложение","охраннику","инженеру","Давид","выслал","рапорт",True),
    Lexical("Константин","передал","передать","передано","соглашение","стажёру","директору","Антон","направил","бланк",False),
    Lexical("Марина","предъявила","предъявить","предъявлено","свидетельство","механику","продавцу","Гриша","доставил","конверт",True),
    Lexical("Егор","отправил","отправить","отправлено","подтверждение","сотруднику","снабженцу","Семён","показал","билет",False),
    Lexical("Лидия","вручила","вручить","вручено","уведомление","курсанту","переводчику","Ефим","передал","проект",True),
    Lexical("Владислав","направил","направить","направлено","обращение","наблюдателю","исполнителю","Богдан","вручил","дневник",False),
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

def create_row(l:Lexical,family:int,pattern:int)->TrainGraph:
    decide="решила" if l.feminine else "решил"
    read="прочитала" if l.feminine else "прочитал"
    text=None
    event_forms=[]
    edge_forms=[]
    if pattern==0:
        text=f"{l.agent} {l.past} {l.document} не {l.recipient_a} и не {l.recipient_b}."
        event_forms=[l.past]
        edge_forms=[(l.past,l.recipient_a),(l.past,l.recipient_b)]
    elif pattern==1:
        text=f"{l.agent} {l.past} {l.document} совсем не {l.recipient_a} и вовсе не {l.recipient_b}."
        event_forms=[l.past]
        edge_forms=[(l.past,l.recipient_a),(l.past,l.recipient_b)]
    elif pattern==2:
        text=f"{l.agent} не {l.past} {l.document} {l.recipient_a} и {read} фразу «не {l.recipient_b}»."
        event_forms=[l.past,read]
    elif pattern==3:
        text=f"Не {l.recipient_a} {l.document} вчера {l.passive}."
        event_forms=[l.passive]
        edge_forms=[(l.passive,l.recipient_a)]
    elif pattern==4:
        text=f"Никому из группы {l.agent} {decide} {l.infinitive} {l.document}."
        event_forms=[decide,l.infinitive]
        edge_forms=[(l.infinitive,"Никому")]
    elif pattern==5:
        text=f"{l.agent} {l.past} {l.document} {l.recipient_a}, но совсем не {l.recipient_b}."
        event_forms=[l.past]
        edge_forms=[(l.past,l.recipient_b)]
    elif pattern==6:
        text=f"Не {l.recipient_a} {l.agent} {l.past} {l.document}, а {l.partner} {l.partner_past} {l.partner_document} не {l.recipient_b}."
        event_forms=[l.past,l.partner_past]
        edge_forms=[(l.past,l.recipient_a),(l.partner_past,l.recipient_b)]
    elif pattern==7:
        text=f"{l.agent} {read} текст «не {l.recipient_a}», затем {l.past} {l.document} не {l.recipient_b}."
        event_forms=[read,l.past]
        edge_forms=[(l.past,l.recipient_b)]
    else:
        raise ValueError("unknown new S45 TRAIN construction")
    tokens=raw_surface_grammar(text)
    def locate(term):
        idx=[i for i,t in enumerate(tokens) if t.form==term]
        if len(idx)!=1:
            raise ValueError(f"S45 TRAIN graph ambiguous lexical annotation {term!r}: {text!r} {idx}")
        return idx[0]
    sources=tuple(locate(x) for x in event_forms)
    links=tuple((locate(src),locate(dst)) for src,dst in edge_forms)
    quoted=[]
    depth=0
    for i,t in enumerate(tokens):
        if t.form=="«":depth+=1
        elif t.form=="»":depth-=1
        elif depth:
            quoted.append(i)
    if depth or len(set(sources))!=len(sources) or len(set(links))!=len(links):
        raise ValueError("S45 invalid quote or duplicate TRAIN graph index")
    if any(src not in sources for src,_ in links):
        raise ValueError("S45 TRAIN graph links from non-event source")
    return TrainGraph(text,family,pattern,sources,links,tuple(quoted),
                      len(sources),len(links))

def train64():
    return tuple(create_row(l,i,j) for i,l in enumerate(LEXICAL_FAMILIES)
                 for j in range(len(STRUCTURES)))

def source_digest(rows=None):
    rows=rows if rows is not None else train64()
    source=[{"text":r.text,"family":r.family,"group":r.construction,
             "sources":r.event_sources,"links":r.rejected_edges,
             "quoted":r.quoted_token_indices} for r in rows]
    return hashlib.sha256(json.dumps(
        source,ensure_ascii=False,sort_keys=True,separators=(",",":")
    ).encode("utf-8")).hexdigest()

def validate_source():
    rows=train64()
    assert len(rows)==64
    assert len(STRUCTURES)==8 and len(LEXICAL_FAMILIES)==8
    assert len({r.text for r in rows})==64
    prior=set(text_only_inputs())
    prior.update(r.text for r in build_train80())
    prior.update(r.text for r in build_train128())
    assert not {r.text for r in rows}&prior
    assert sum(len(r.rejected_edges)==2 for r in rows)==8*3
    assert sum(len(r.rejected_edges)==0 for r in rows)==8
    assert sum(len(r.rejected_edges)==1 for r in rows)==8*4
    assert all(len(raw_surface_grammar(r.text))<=40 for r in rows)
    assert all(r.semantic_events_count>=1 for r in rows)
    return {
       "stage":STAGE,
       "TRAIN_only_rows":64,
       "lexical_families":8,
       "syntactic_patterns":8,
       "multi_recipient_or_multievent_examples":sum(
           len(r.rejected_edges)>=2 or len(r.event_sources)>=2 for r in rows),
       "positive_directed_training_links":sum(len(r.rejected_edges) for r in rows),
       "negative_event_without_rejected_edges":8,
       "disjoint_S43_TRAIN128_S39_TRAIN80_S22_TRAIN256":True,
       "no_S38_examples_loaded":True,
       "source_sha256":source_digest(rows),
    }

if __name__=="__main__":
    print(json.dumps(validate_source(),sort_keys=True,ensure_ascii=False))
