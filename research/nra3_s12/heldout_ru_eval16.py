"""NRA3 S12: one-time 16 new Russian heldout evaluation sentences.

Original S11 39-parameter FP16 router was frozen and physically SHA-locked
in CI #43734571081 BEFORE this text or any EVAL label was created.

This is a narrowly controlled lexical and construction transfer TEST, not
a representative natural Russian benchmark. It is not training material.
No prompt, role, head or answer from the previously consumed 5xEVAL128 or
reserved sixth S3CI EVAL128 is imported. Do not tune on this holdout.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from research.nra2_bridge.stage3ck_russian_ud_role_bridge import (
    UDWord,extract_grammar,
)
from research.nra3_s0.neural_area import MORPH,SEMANTIC_ROLES,UPOS
from research.nra3_s4.russian_ud_train_only import build_train as old_train
from research.nra3_s9.new_ru_train72 import build_s9_train72

SPLIT="NRA3_S12_NEW_RU_HELDOUT_EVAL16_LOCKED_NO_TRAIN"
PREVIOUS_S11_FROZEN_RUN=43734571081
PREVIOUS_S11_FROZEN_HEAD_SHA="8d7cbf7fd8ed2acc5d903117a8b943d15d98d791"
PREVIOUS_S11_PHYSICAL_ROUTER_SHA="05ce14e20c9f4e7e09f85dbd9bd46f0dc0a700d72f1a96ffa921b2e822f235c3"
S9_TRAIN72_SHA="b09d60a77a17d30c2dbf846e40ab97860def8635e2bf31a5b0ab8981832b744d"
# Obtained once from the first CI source-only audit; must be pinned before
# any original model forward or S11 scoring on these 16 EVAL texts.
EVAL16_FULL_CANONICAL_SHA="f2d2d5beba48dbe294a90f6918a3fd15870aa103abbd569e907c44729a43977c"


@dataclass(frozen=True)
class S12NovelEval:
    text:str
    words:tuple[UDWord,...]
    roles:tuple[str,...]
    construction:str
    split:str=SPLIT

    def sha256(self):
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


# (form,lemma,upos,head,deprel,features,semantic role).
Item=tuple[str,str,str,int,str,tuple[str,...],str]
N=("Case=Nom",)
A=("Case=Acc",)
D=("Case=Dat",)
G=("Case=Gen",)
I=("Case=Ins",)
ACT=("Tense=Past","Voice=Act")
PASS=("Voice=Pass",)


def _row(category:str, words:tuple[Item,...])->S12NovelEval:
    surface=[w[0] for w in words]
    assert surface[-1]=="."
    text=" ".join(surface[:-1])+"."
    cursor=0
    ud=[]
    roles=[]
    for i,(form,lemma,upos,head,rel,feats,role) in enumerate(words,1):
        left=text.find(form,cursor)
        right=left+len(form)
        if left<cursor or not form or upos not in UPOS or role not in SEMANTIC_ROLES:
            raise ValueError("S12 invalid source word or role")
        if any(f not in MORPH for f in feats):
            raise ValueError("S12 unknown morphological annotation")
        ud.append(UDWord(i,form,lemma,upos,head,rel,left,right,feats))
        roles.append(role)
        cursor=right
    return S12NovelEval(text,tuple(ud),tuple(roles),category)


def build_eval16()->tuple[S12NovelEval,...]:
    o=[]
    def add(category,*spec):
        o.append(_row(category,tuple(spec)))
    P=lambda h:(".", ".", "PUNCT",h,"punct",(),"NONE")
    add("gift_active",
      ("Марина","Марина","PROPN",2,"nsubj",N,"AGENT"),
      ("подарила","подарить","VERB",0,"root",ACT,"NONE"),
      ("Рустаму","Рустам","PROPN",2,"iobj",D,"RECIPIENT"),
      ("книгу","книга","NOUN",2,"obj",A,"PATIENT"),P(2))
    add("novel_verb_object_fronting",
      ("Записку","записка","NOUN",3,"obj",A,"PATIENT"),
      ("Ладе","Лада","PROPN",3,"iobj",D,"RECIPIENT"),
      ("отнёс","отнести","VERB",0,"root",ACT,"NONE"),
      ("Тимур","Тимур","PROPN",3,"nsubj",N,"AGENT"),P(3))
    add("received_source_distinct",
      ("Элина","Элина","PROPN",2,"nsubj",N,"RECIPIENT"),
      ("получила","получить","VERB",0,"root",ACT,"NONE"),
      ("посылку","посылка","NOUN",2,"obj",A,"PATIENT"),
      ("от","от","ADP",5,"case",(),"NONE"),
      ("Якова","Яков","PROPN",2,"obl",G,"SOURCE"),P(2))
    add("passive_gift",
      ("Каталог","каталог","NOUN",3,"nsubj:pass",N,"PATIENT"),
      ("был","быть","AUX",3,"aux",(),"NONE"),
      ("вручён","вручить","VERB",0,"root",PASS,"NONE"),
      ("Константином","Константин","PROPN",3,"obl:agent",I,"AGENT"),
      ("Василисе","Василиса","PROPN",3,"iobj",D,"RECIPIENT"),P(3))
    add("temporal_new_past_verb",
      ("Сегодня","сегодня","ADV",3,"advmod",(),"NONE"),
      ("Вадим","Вадим","PROPN",3,"nsubj",N,"AGENT"),
      ("принёс","принести","VERB",0,"root",ACT,"NONE"),
      ("Жанне","Жанна","PROPN",3,"iobj",D,"RECIPIENT"),
      ("коробку","коробка","NOUN",3,"obj",A,"PATIENT"),P(3))
    add("adjective_before_patient",
      ("Тонкую","тонкий","ADJ",2,"amod",A,"NONE"),
      ("тетрадь","тетрадь","NOUN",4,"obj",A,"PATIENT"),
      ("Инга","Инга","PROPN",4,"nsubj",N,"AGENT"),
      ("передала","передать","VERB",0,"root",ACT,"NONE"),
      ("Савелию","Савелий","PROPN",4,"iobj",D,"RECIPIENT"),P(4))
    add("time_adverb_deliver",
      ("Вечером","вечером","ADV",3,"advmod",(),"NONE"),
      ("Надежда","Надежда","PROPN",3,"nsubj",N,"AGENT"),
      ("переслала","переслать","VERB",0,"root",ACT,"NONE"),
      ("Милане","Милана","PROPN",3,"iobj",D,"RECIPIENT"),
      ("фотографию","фотография","NOUN",3,"obj",A,"PATIENT"),P(3))
    add("same_event_novel_verb_inverted",
      ("Фотографию","фотография","NOUN",3,"obj",A,"PATIENT"),
      ("Милане","Милана","PROPN",3,"iobj",D,"RECIPIENT"),
      ("переслала","переслать","VERB",0,"root",ACT,"NONE"),
      ("Надежда","Надежда","PROPN",3,"nsubj",N,"AGENT"),P(3))
    add("negative_not_actual_transfer",
      ("Кирилл","Кирилл","PROPN",3,"nsubj",N,"AGENT"),
      ("не","не","PART",3,"advmod",(),"NONE"),
      ("отправил","отправить","VERB",0,"root",ACT,"NONE"),
      ("Софье","Софья","PROPN",3,"iobj",D,"RECIPIENT"),
      ("открытку","открытка","NOUN",3,"obj",A,"PATIENT"),P(3))
    add("passive_return_agent",
      ("Рюкзак","рюкзак","NOUN",3,"nsubj:pass",N,"PATIENT"),
      ("был","быть","AUX",3,"aux",(),"NONE"),
      ("возвращён","возвратить","VERB",0,"root",PASS,"NONE"),
      ("Львом","Лев","PROPN",3,"obl:agent",I,"AGENT"),
      ("Полине","Полина","PROPN",3,"iobj",D,"RECIPIENT"),P(3))
    add("received_original_sender",
      ("Полина","Полина","PROPN",2,"nsubj",N,"RECIPIENT"),
      ("получила","получить","VERB",0,"root",ACT,"NONE"),
      ("рюкзак","рюкзак","NOUN",2,"obj",A,"PATIENT"),
      ("от","от","ADP",5,"case",(),"NONE"),
      ("Льва","Лев","PROPN",2,"obl",G,"SOURCE"),P(2))
    add("winter_adverb_novel_verb",
      ("Зимой","зимой","ADV",3,"advmod",(),"NONE"),
      ("Степан","Степан","PROPN",3,"nsubj",N,"AGENT"),
      ("отвёз","отвезти","VERB",0,"root",ACT,"NONE"),
      ("Тамаре","Тамара","PROPN",3,"iobj",D,"RECIPIENT"),
      ("лекарство","лекарство","NOUN",3,"obj",A,"PATIENT"),P(3))
    add("quality_object_fronted",
      ("Редкую","редкий","ADJ",2,"amod",A,"NONE"),
      ("монету","монета","NOUN",4,"obj",A,"PATIENT"),
      ("Матвей","Матвей","PROPN",4,"nsubj",N,"AGENT"),
      ("показал","показать","VERB",0,"root",ACT,"NONE"),
      ("Зое","Зоя","PROPN",4,"iobj",D,"RECIPIENT"),P(4))
    add("blueprint_new_stem",
      ("Серафима","Серафима","PROPN",2,"nsubj",N,"AGENT"),
      ("передала","передать","VERB",0,"root",ACT,"NONE"),
      ("Филиппу","Филипп","PROPN",2,"iobj",D,"RECIPIENT"),
      ("чертёж","чертёж","NOUN",2,"obj",A,"PATIENT"),P(2))
    add("blueprint_same_event_fronted",
      ("Чертёж","чертёж","NOUN",3,"obj",A,"PATIENT"),
      ("Филиппу","Филипп","PROPN",3,"iobj",D,"RECIPIENT"),
      ("передала","передать","VERB",0,"root",ACT,"NONE"),
      ("Серафима","Серафима","PROPN",3,"nsubj",N,"AGENT"),P(3))
    add("negation_and_temporal_two_modifiers",
      ("Утром","утром","ADV",4,"advmod",(),"NONE"),
      ("Ася","Ася","PROPN",4,"nsubj",N,"AGENT"),
      ("не","не","PART",4,"advmod",(),"NONE"),
      ("принесла","принести","VERB",0,"root",ACT,"NONE"),
      ("Григорию","Григорий","PROPN",4,"iobj",D,"RECIPIENT"),
      ("таблицу","таблица","NOUN",4,"obj",A,"PATIENT"),P(4))
    out=tuple(o)
    validate_eval16(out)
    return out


def validate_eval16(rows:tuple[S12NovelEval,...])->None:
    if (type(rows) is not tuple or len(rows)!=16 or
        any(type(r) is not S12NovelEval or r.split!=SPLIT for r in rows) or
        len({r.sha256() for r in rows})!=16):
        raise ValueError("S12 heldout fixed 16 sentence IDs required")
    disallowed={r.sha256() for r in old_train()}|{
        r.sha256() for r in build_s9_train72()}
    if disallowed&{r.sha256() for r in rows}:
        raise ValueError("S12 overlaps original S4 or S9 TRAIN")
    for row in rows:
        if (len(row.words)!=len(row.roles) or
            row.words[-1].upos!="PUNCT" or
            not {"PATIENT","RECIPIENT"}.issubset(row.roles) or
            not ({"AGENT","SOURCE"}&set(row.roles)) or
            not set(row.roles).issubset(SEMANTIC_ROLES) or
            any(w.upos not in UPOS or
                any(f not in MORPH for f in w.feats)
                for w in row.words)):
            raise ValueError("S12 unannotated or unsupported Russian EVAL")
        frame=extract_grammar(row.text,row.words)
        if frame.semantic_direction is not None or frame.graph_reachability is not None:
            raise ValueError("S12 prohibited graph/source oracle")
    categories=[r.construction for r in rows]
    if len(set(categories))!=len(categories):
        raise ValueError("S12 all 16 independent construction categories required")
    for r in rows:
        # Do not equate negative clause arguments with events that occurred.
        if "не" in [w.text for w in r.words]:
            if not r.construction.startswith(("negative","negation")):
                raise ValueError("S12 negation event truth is NOT source/target fact")


def canonical_eval16(rows:tuple[S12NovelEval,...])->bytes:
    validate_eval16(rows)
    data=[{"text":r.text,"text_sha256":r.sha256(),"split":r.split,
           "construction":r.construction,
           "words":[{"word_id":w.word_id,"text":w.text,"lemma":w.lemma,
                     "upos":w.upos,"head":w.head,"deprel":w.deprel,
                     "start":w.start,"end":w.end,"feats":list(w.feats),
                     "semantic_role":r.roles[i]}
                    for i,w in enumerate(r.words)]}
          for r in rows]
    return json.dumps(data,sort_keys=True,ensure_ascii=False,
                      separators=(",",":")).encode("utf-8")


def sealed_eval16()->dict:
    rows=build_eval16()
    raw=canonical_eval16(rows)
    sha=hashlib.sha256(raw).hexdigest()
    if sha!=EVAL16_FULL_CANONICAL_SHA:
        raise ValueError("S12 EVAL16 source changed after first source-only sealing")
    return {
      "stage":"NRA3_S12_PRE_REGISTERED_NEW_RU_HELDOUT16_NO_MODEL_EVAL_YET",
      "source_split":SPLIT,
      "complete_annotation_sha256":sha,
      "rows":len(rows),
      "row_text_sha256":[r.sha256() for r in rows],
      "s9_training_sha256":S9_TRAIN72_SHA,
      "prior_S11_locked_pre_EVAL_run":PREVIOUS_S11_FROZEN_RUN,
      "prior_S11_locked_pre_EVAL_head_sha":PREVIOUS_S11_FROZEN_HEAD_SHA,
      "prior_S11_router_fp16_physical_sha256":PREVIOUS_S11_PHYSICAL_ROUTER_SHA,
      "overlap_s4_s9_train_count":0,
      "new_eval_constructs_count":len({r.construction for r in rows}),
      "source_is_posthoc_to_model_frozen_but_pre_eval":True,
      "evaluation_started":False,
      "optimizer_steps_on_eval":0,
      "all_previous_five_consumed_EVAL128_untouched":True,
      "original_sixth_S3CI_EVAL128_untouched":True,
      "quality_or_generalization_gain_proven":False,
      "no_LIVE_Render_Supabase_APK_or_Jessica_chat":True,
      "output_gate":0,
      "candidate_accepted":False
    }
