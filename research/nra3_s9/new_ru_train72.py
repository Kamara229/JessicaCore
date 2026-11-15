"""NRA2 3.0 S9: 72 NEW Russian UD contrastive TRAIN sentences, six constructions.

This is additional *development TRAIN*, not an independent EVAL. All
annotations and lemma/case/role labels are optimizer-side only. No
original CALIBRI, model inference, J7, heldout test, or Jessica chat.

Generation uses 12 independent Russian entity/object lexical triplets x
six fixed UD constructions. This is a limited controlled generalization
study (lexical variation plus six transformations), NOT a broad free-form
Russian corpus, not coreference and not full natural-language competence.

S4 old TRAIN12 identities and original five consumed EVAL128 and reserved
sixth S3CI EVAL128 are never loaded as a supervised test.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

from research.nra2_bridge.stage3ck_russian_ud_role_bridge import (
    UDWord, extract_grammar,
)
from research.nra3_s0.neural_area import MORPH,SEMANTIC_ROLES,UPOS
from research.nra3_s4.russian_ud_train_only import (
    build_train as build_s4_train,
    TRAIN12_CANONICAL_SHA256,
)

S9_TRAIN_SPLIT="NRA3_S9_NEW_RU_UD_TRAIN72_ONLY"
# Independent SHA of complete canonical 72-row UD+semantic TRAIN annotations;
# after first successful offline CI, freeze to observed SHA for all follow-ups.
S9_TRAIN72_CANONICAL_SHA256="b09d60a77a17d30c2dbf846e40ab97860def8635e2bf31a5b0ab8981832b744d"
TEMPLATES=(
    "active_svo","fronted_object","recipient_subject",
    "passive_agent","fronted_adjective","temporal_adverb",
)


@dataclass(frozen=True)
class LexicalTriplet:
    source_nom:str
    source_gen:str
    source_ins:str
    recipient_nom:str
    recipient_dat:str
    object_nom:str
    object_lemma:str

# No real-world personal content: hand-specified fictional sentence examples.
_TRIPLETS=(
    LexicalTriplet("Артём","Артёма","Артёмом","Ольга","Ольге","пакет","пакет"),
    LexicalTriplet("Никита","Никиты","Никитой","Дарья","Дарье","ключ","ключ"),
    LexicalTriplet("Илья","Ильи","Ильёй","Вера","Вере","документ","документ"),
    LexicalTriplet("Максим","Максима","Максимом","Юлия","Юлии","билет","билет"),
    LexicalTriplet("Егор","Егора","Егором","Таня","Тане","чек","чек"),
    LexicalTriplet("Сергей","Сергея","Сергеем","Нина","Нине","подарок","подарок"),
    LexicalTriplet("Денис","Дениса","Денисом","Рита","Рите","отчёт","отчёт"),
    LexicalTriplet("Роман","Романа","Романом","Света","Свете","конверт","конверт"),
    LexicalTriplet("Пётр","Петра","Петром","Лиза","Лизе","счёт","счёт"),
    LexicalTriplet("Олег","Олега","Олегом","Катя","Кате","план","план"),
    LexicalTriplet("Глеб","Глеба","Глебом","Мила","Миле","жетон","жетон"),
    LexicalTriplet("Стас","Стаса","Стасом","Алиса","Алисе","сертификат","сертификат"),
)


@dataclass(frozen=True)
class NewRussianTrain:
    text:str
    words:tuple[UDWord,...]
    roles:tuple[str,...]
    construction:str
    lexical_family:int
    split:str=S9_TRAIN_SPLIT

    def sha256(self)->str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


# Surface, lemma, UPOS, HEAD (1-based, 0=root), deprel, morphology, semantic role.
Item=tuple[str,str,str,int,str,tuple[str,...],str]
_N=("Case=Nom","Number=Sing")
_A=("Case=Acc","Number=Sing")
_D=("Case=Dat","Number=Sing")
_G=("Case=Gen","Number=Sing")
_I=("Case=Ins","Number=Sing")
_P=("Tense=Past","Voice=Pass")
_V=("Tense=Past","Voice=Act")


def _make(parts:tuple[Item,...],*,construction:str,
          lexical_family:int)->NewRussianTrain:
    surfaces=[x[0] for x in parts]
    text=" ".join(surfaces[:-1])+surfaces[-1]
    if not text.endswith(".") or parts[-1][0]!=".":
        raise ValueError("S9 punctuation or sentence serializer drift")
    words=[]
    roles=[]
    cursor=0
    for index,(surface,lemma,upos,head,dep,feats,role) in enumerate(parts,1):
        start=text.find(surface,cursor)
        end=start+len(surface)
        if (start<cursor or not surface or upos not in UPOS
                or role not in SEMANTIC_ROLES
                or any(f not in MORPH for f in feats)):
            raise ValueError("S9 morph/role/surface TRAIN source mismatch")
        words.append(UDWord(index,surface,lemma,upos,head,dep,
                            start,end,feats))
        roles.append(role)
        cursor=end
    return NewRussianTrain(text,tuple(words),tuple(roles),construction,
                           lexical_family)


def _six(t:LexicalTriplet,family:int)->tuple[NewRussianTrain,...]:
    src=t.source_nom;dst=t.recipient_nom;d=t.recipient_dat
    item=t.object_nom;upper=item.capitalize()
    n=lambda word,case,role,head,dep,lemma: (
        word,lemma,"PROPN",head,dep,case,role)
    obj=lambda word,case,head,dep:(
        word,t.object_lemma,"NOUN",head,dep,case,"PATIENT")
    v=lambda word,head: (word,"передать","VERB",head,
                         "root" if head==0 else "conj",_V,"NONE")
    punct=lambda head:(".", ".", "PUNCT",head,"punct",(),"NONE")
    active=_make((
        n(src,_N,"AGENT",2,"nsubj",src),v("передал",0),
        n(d,_D,"RECIPIENT",2,"iobj",dst),
        obj(item,_A,2,"obj"),punct(2)),
        construction=TEMPLATES[0],lexical_family=family)
    fronted=_make((
        obj(upper,_A,3,"obj"),
        n(d,_D,"RECIPIENT",3,"iobj",dst),
        v("передал",0),n(src,_N,"AGENT",3,"nsubj",src),
        punct(3)),construction=TEMPLATES[1],lexical_family=family)
    received=_make((
        n(dst,_N,"RECIPIENT",2,"nsubj",dst),
        ("получила","получить","VERB",0,"root",_V,"NONE"),
        obj(item,_A,2,"obj"),
        ("от","от","ADP",5,"case",(),"NONE"),
        n(t.source_gen,_G,"SOURCE",2,"obl",src),
        punct(2)),construction=TEMPLATES[2],lexical_family=family)
    passive=_make((
        obj(upper,_N,3,"nsubj:pass"),
        ("был","быть","AUX",3,"aux",(),"NONE"),
        ("передан","передать","VERB",0,"root",_P,"NONE"),
        n(t.source_ins,_I,"AGENT",3,"obl:agent",src),
        n(d,_D,"RECIPIENT",3,"iobj",dst),
        punct(3)),construction=TEMPLATES[3],lexical_family=family)
    adjective=_make((
        ("Новый","новый","ADJ",2,"amod",_A,"NONE"),
        obj(item,_A,4,"obj"),
        n(src,_N,"AGENT",4,"nsubj",src),
        v("передал",0),n(d,_D,"RECIPIENT",4,"iobj",dst),
        punct(4)),construction=TEMPLATES[4],lexical_family=family)
    adverb=_make((
        ("Вчера","вчера","ADV",3,"advmod",(),"NONE"),
        n(src,_N,"AGENT",3,"nsubj",src),
        v("передал",0),n(d,_D,"RECIPIENT",3,"iobj",dst),
        obj(item,_A,3,"obj"),punct(3)),
        construction=TEMPLATES[5],lexical_family=family)
    return (active,fronted,received,passive,adjective,adverb)


def build_s9_train72()->tuple[NewRussianTrain,...]:
    out=tuple(x for i,t in enumerate(_TRIPLETS) for x in _six(t,i))
    validate_s9_train72(out)
    return out


def validate_s9_train72(rows:tuple[NewRussianTrain,...])->None:
    if (not isinstance(rows,tuple) or len(rows)!=72 or
            len(_TRIPLETS)!=12 or
            len({x.sha256() for x in rows})!=72 or
            len({(x.lexical_family,x.construction) for x in rows})!=72
            or any(x.split!=S9_TRAIN_SPLIT for x in rows)):
        raise ValueError("S9 TRAIN72 lexical-family or split integrity violated")
    # The original 12 are allowed only as earlier TRAIN identities. NONE
    # of their text can be reused as a supposedly new unseen example.
    old_s4={r.sha256() for r in build_s4_train()}
    if set(x.sha256() for x in rows)&old_s4:
        raise ValueError("S9 TRAIN72 overlaps S4 original TRAIN12")
    for r in rows:
        w=r.words
        if (r.construction not in TEMPLATES or
                not 0<=r.lexical_family<len(_TRIPLETS) or
                len(w)!=len(r.roles) or
                [i.word_id for i in w]!=list(range(1,len(w)+1)) or
                sum(i.head==0 for i in w)!=1 or
                w[-1].upos!="PUNCT" or
                any(i.head<0 or i.head>len(w) or i.head==i.word_id
                    for i in w) or
                any(i.upos not in UPOS for i in w) or
                any(a not in MORPH for i in w for a in i.feats) or
                not set(r.roles).issubset(SEMANTIC_ROLES) or
                "PATIENT" not in r.roles or
                "RECIPIENT" not in r.roles or
                not ("AGENT" in r.roles or "SOURCE" in r.roles)):
            raise ValueError("S9 invalid UD head morphology or semantic labels")
        if r.construction=="recipient_subject":
            if (r.words[0].deprel!="nsubj" or r.roles[0]!="RECIPIENT"
                    or r.roles[4]!="SOURCE" or "AGENT" in r.roles):
                raise ValueError("S9 recipient-subject vs agent semantic confusion")
        if r.construction=="passive_agent":
            if (r.words[0].deprel!="nsubj:pass" or r.roles[0]!="PATIENT"
                    or r.roles[3]!="AGENT"):
                raise ValueError("S9 passive syntax/semantic inversion lost")
        if r.construction=="fronted_object":
            if (r.words[0].deprel!="obj" or r.roles[0]!="PATIENT"):
                raise ValueError("S9 original object fronting lost")
        if r.construction=="fronted_adjective":
            if (r.words[0].deprel!="amod" or r.words[0].head!=2):
                raise ValueError("S9 adjective noun dependency lost")
        # Tests a small, partial subset of the existing S3CK grammar
        # contract only, NEVER a language-model prediction or evaluation.
        fr=extract_grammar(r.text,r.words)
        if fr.semantic_direction is not None or fr.graph_reachability is not None:
            raise ValueError("S9 forbidden graph-oracle truth")


def canonical_train72(rows:tuple[NewRussianTrain,...])->bytes:
    validate_s9_train72(rows)
    source=[{
        "split":r.split,"text":r.text,
        "text_sha256":r.sha256(),
        "construction":r.construction,
        "lexical_family":r.lexical_family,
        "tokens":[{
            "id":w.word_id,"form":w.text,"lemma":w.lemma,
            "upos":w.upos,"head":w.head,"deprel":w.deprel,
            "start":w.start,"end":w.end,"feats":list(w.feats),
            "semantic_role":r.roles[j],
        } for j,w in enumerate(r.words)],
        "evaluated":False,
        "coreference_gold_provided":False,
        "graph_gold_provided":False,
    } for r in rows]
    return json.dumps(source,sort_keys=True,ensure_ascii=False,
                      separators=(",",":")).encode("utf-8")


def seal_train72()->dict[str,Any]:
    rows=build_s9_train72()
    serialized=canonical_train72(rows)
    sha=hashlib.sha256(serialized).hexdigest()
    if sha!=S9_TRAIN72_CANONICAL_SHA256:
        raise ValueError("S9 immutable TRAIN72 gold/lexical manifest drift")
    return {
        "stage":"NRA3_S9_NEW_CONTRASTIVE_RUSSIAN_UD_TRAIN72_SOURCE_ONLY",
        "split":S9_TRAIN_SPLIT,
        "canonical_train72_sha256":sha,
        "source_rows":len(rows),
        "lexical_families":len(_TRIPLETS),
        "constructions":list(TEMPLATES),
        "rows_per_construction":{c:sum(r.construction==c for r in rows)
                                 for c in TEMPLATES},
        "text_sha256_ordered":[r.sha256() for r in rows],
        "s4_original_train12_canonical_sha256":TRAIN12_CANONICAL_SHA256,
        "s4_train12_not_reused_as_independent_evaluation":True,
        "additional_independent_eval_rows_used":0,
        "old_five_EVAL128_untouched":True,
        "reserved_sixth_S3CI_EVAL128_untouched":True,
        "future_independent_EVAL_requires_separate_creation_and_sealing":True,
        "real_frozen_43M_cpu_layer8_hidden320_captured":False,
        "physical_5bit_CB59_train_performed":False,
        "actual_grammatical_or_semantic_generalization_evaluated":False,
        "output_gate":0,
        "candidate_accepted":False,
        "no_LIVE_Render_Supabase_APK_or_Jessica_chat":True,
    }
