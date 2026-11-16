"""NRA2 S17: NEW independent Russian directed predicate→argument EVAL24.

Source-only preregistration AFTER SHA-reproducible S16b 2,309-FP16
model frozen in successful GitHub Actions #43758892125.

DO NOT import/inspect/score the consumed S15c EVAL24 or S12c EVAL16.
Do not load any 43M/GT/CB59/S16b physical source in THIS stage.
This file defines *new* annotated Russian sentences with different
verbs, noun phrases, lexical families, negation, temporal modality,
word order, and some multiclausal syntax relative to S9 TRAIN72.

This corpus is a small hand-authored diagnostic, NOT a population
representative benchmark. Directed role task is only the S16b five
classes NONE/AGENT/PATIENT/RECIPIENT/SOURCE. 'Negated' and 'future'
are metadata, NOT truth prediction; reference pronouns are NOT tested
for coreference. No gold role/UD/verb filter enters inference, ever.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from research.nra3_s9.new_ru_train72 import build_s9_train72
from research.nra3_s4.russian_ud_train_only import build_train as old_s4_train
from research.nra3_s16.paid_directed_valence import ROLES

SPLIT="NRA3_S17_ONE_TIME_NEW_RUSSIAN24_DIRECTED_VALENCE_EVAL_ONLY"
S16B_FROZEN_SOURCE_RUN=43758892125
S16B_FROZEN_SOURCE_SHA="3fe4bbce718e53e4e812e94367ce25834f873cbed646479c59b1e3949d2ecf38"
S16B_FROZEN_FULL_HEAD_SHA="ea063bf1fa0e856111c54f3b045220ca7aa5319e"
S16B_ORIGINAL_TRAIN72_GOLD_SHA="9843c35ee9aa75b8f5c9c963e441589dd2f0292e41ec7933971d5aa3dcd365d6"
S17_EVAL24_COMPLETE_SHA="" # strictly set after FIRST successful source-only CI

# Surface, UD UPOS, 1-based head, UD relation, DIRECT semantic role.
# An optional 5th tuple field gives UD morphological features.
Spec=tuple[str,str,int,str,str]


@dataclass(frozen=True)
class EVALWord:
    index:int
    text:str
    upos:str
    head:int
    deprel:str
    role:str
    start:int
    end:int

@dataclass(frozen=True)
class EVALSentence:
    text:str
    words:tuple[EVALWord,...]
    construction:str
    event_modality:str
    split:str=SPLIT

    def text_sha256(self)->str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def _w(surface,upos,head,deprel,role="NONE"):
    return (surface,upos,head,deprel,role)


def _make(construction,modality,*spec):
    surface=[v[0] for v in spec]
    if len(spec)<4 or surface[-1]!=".":
        raise ValueError("S17 requires exact punctuation")
    text=" ".join(surface)
    for punc in (".",",","!","?"):
        text=text.replace(" "+punc,punc)
    words=[];cursor=0
    for i,(form,tag,head,rel,role) in enumerate(spec,1):
        start=text.find(form,cursor)
        if start<cursor:
            raise ValueError("S17 Unicode offsets not monotone")
        end=start+len(form)
        words.append(EVALWord(i,form,tag,head,rel,role,start,end))
        cursor=end
    return EVALSentence(text,tuple(words),construction,modality)


def build_new_eval24()->tuple[EVALSentence,...]:
    w=_w;p=lambda head:w(".","PUNCT",head,"punct")
    comma=lambda head:w(",","PUNCT",head,"punct")
    corpus=(
      _make("unseen_active_transfer_01","ASSERTED",
        w("Флора","PROPN",2,"nsubj","AGENT"),
        w("переслала","VERB",0,"root"),
        w("Аркадию","PROPN",2,"iobj","RECIPIENT"),
        w("чертёж","NOUN",2,"obj","PATIENT"),p(2)),
      _make("patient_before_agent_02","ASSERTED",
        w("Рецепт","NOUN",4,"obj","PATIENT"),
        w("Мира","PROPN",4,"nsubj","AGENT"),
        w("вчера","ADV",4,"advmod"),
        w("отдала","VERB",0,"root"),
        w("Савелию","PROPN",4,"iobj","RECIPIENT"),p(4)),
      _make("preposition_temporal_03","ASSERTED",
        w("На","ADP",2,"case"),
        w("рассвете","NOUN",4,"obl"),
        w("Ефим","PROPN",4,"nsubj","AGENT"),
        w("доставил","VERB",0,"root"),
        w("Глафире","PROPN",4,"iobj","RECIPIENT"),
        w("свёрток","NOUN",4,"obj","PATIENT"),p(4)),
      _make("fronted_adjective_object_04","ASSERTED",
        w("Небольшую","ADJ",2,"amod"),
        w("шкатулку","NOUN",4,"obj","PATIENT"),
        w("Варя","PROPN",4,"nsubj","AGENT"),
        w("вручила","VERB",0,"root"),
        w("Радмиру","PROPN",4,"iobj","RECIPIENT"),p(4)),
      _make("numeral_plural_patient_05","ASSERTED",
        w("Гаврила","PROPN",2,"nsubj","AGENT"),
        w("привёз","VERB",0,"root"),
        w("Астре","PROPN",2,"iobj","RECIPIENT"),
        w("две","NUM",5,"nummod"),
        w("папки","NOUN",2,"obj","PATIENT"),p(2)),
      _make("recipient_subject_from_source_06","ASSERTED",
        w("Оксана","PROPN",2,"nsubj","RECIPIENT"),
        w("получила","VERB",0,"root"),
        w("медаль","NOUN",2,"obj","PATIENT"),
        w("от","ADP",5,"case"),
        w("Юноны","PROPN",2,"obl","SOURCE"),p(2)),
      _make("fronted_patient_recipient_source_07","ASSERTED",
        w("Награду","NOUN",2,"obj","PATIENT"),
        w("получила","VERB",0,"root"),
        w("Ядвига","PROPN",2,"nsubj","RECIPIENT"),
        w("от","ADP",5,"case"),
        w("Зои","PROPN",2,"obl","SOURCE"),p(2)),
      _make("passive_pat_agent_08","ASSERTED",
        w("Посылка","NOUN",3,"nsubj:pass","PATIENT"),
        w("была","AUX",3,"aux"),
        w("доставлена","VERB",0,"root"),
        w("Мартой","PROPN",3,"obl:agent","AGENT"),
        w("Жанне","PROPN",3,"iobj","RECIPIENT"),p(3)),
      _make("passive_unseen_verb_09","ASSERTED",
        w("Чертёж","NOUN",3,"nsubj:pass","PATIENT"),
        w("был","AUX",3,"aux"),
        w("передан","VERB",0,"root"),
        w("Гектором","PROPN",3,"obl:agent","AGENT"),
        w("Марианне","PROPN",3,"iobj","RECIPIENT"),p(3)),
      _make("fronted_recipient_passive_10","ASSERTED",
        w("Марте","PROPN",3,"iobj","RECIPIENT"),
        w("был","AUX",3,"aux"),
        w("вручён","VERB",0,"root"),
        w("золотой","ADJ",5,"amod"),
        w("значок","NOUN",3,"nsubj:pass","PATIENT"),
        w("Тимуром","PROPN",3,"obl:agent","AGENT"),p(3)),
      _make("negated_agent_transfer_11","NEGATED",
        w("Фаина","PROPN",3,"nsubj","AGENT"),
        w("не","PART",3,"advmod"),
        w("отправила","VERB",0,"root"),
        w("Демьяну","PROPN",3,"iobj","RECIPIENT"),
        w("рукопись","NOUN",3,"obj","PATIENT"),p(3)),
      _make("negated_passive_12","NEGATED",
        w("Печать","NOUN",4,"nsubj:pass","PATIENT"),
        w("не","PART",4,"advmod"),
        w("была","AUX",4,"aux"),
        w("передана","VERB",0,"root"),
        w("Яном","PROPN",4,"obl:agent","AGENT"),
        w("Агате","PROPN",4,"iobj","RECIPIENT"),p(4)),
      _make("future_three_args_13","FUTURE_UNREALIZED",
        w("Завтра","ADV",3,"advmod"),
        w("Герман","PROPN",3,"nsubj","AGENT"),
        w("доставит","VERB",0,"root"),
        w("Элине","PROPN",3,"iobj","RECIPIENT"),
        w("семена","NOUN",3,"obj","PATIENT"),p(3)),
      _make("future_inverted_14","FUTURE_UNREALIZED",
        w("Чертёж","NOUN",3,"obj","PATIENT"),
        w("Аксинье","PROPN",3,"iobj","RECIPIENT"),
        w("принесёт","VERB",0,"root"),
        w("Тимур","PROPN",3,"nsubj","AGENT"),p(3)),
      _make("preposed_time_phrase_15","ASSERTED",
        w("После","ADP",2,"case"),
        w("дождя","NOUN",4,"obl"),
        w("Игнат","PROPN",4,"nsubj","AGENT"),
        w("показал","VERB",0,"root"),
        w("Евдокии","PROPN",4,"iobj","RECIPIENT"),
        w("модель","NOUN",4,"obj","PATIENT"),p(4)),
      _make("professional_noun_subject_16","ASSERTED",
        w("Смотритель","NOUN",2,"nsubj","AGENT"),
        w("передал","VERB",0,"root"),
        w("Людмиле","PROPN",2,"iobj","RECIPIENT"),
        w("уведомление","NOUN",2,"obj","PATIENT"),p(2)),
      _make("female_profession_agent_17","ASSERTED",
        w("Учительница","NOUN",2,"nsubj","AGENT"),
        w("вернула","VERB",0,"root"),
        w("Лавру","PROPN",2,"iobj","RECIPIENT"),
        w("тетрадь","NOUN",2,"obj","PATIENT"),p(2)),
      _make("passive_reversal_profession_18","ASSERTED",
        w("Тетрадь","NOUN",3,"nsubj:pass","PATIENT"),
        w("была","AUX",3,"aux"),
        w("возвращена","VERB",0,"root"),
        w("Лавру","PROPN",3,"iobj","RECIPIENT"),
        w("учительницей","NOUN",3,"obl:agent","AGENT"),p(3)),
      _make("locative_phrase_19","ASSERTED",
        w("В","ADP",2,"case"),
        w("мастерской","NOUN",4,"obl"),
        w("Злата","PROPN",4,"nsubj","AGENT"),
        w("вручила","VERB",0,"root"),
        w("Филиппу","PROPN",4,"iobj","RECIPIENT"),
        w("пропуск","NOUN",4,"obj","PATIENT"),p(4)),
      _make("verb_loan_new_valence_20","ASSERTED",
        w("Михаил","PROPN",2,"nsubj","AGENT"),
        w("одолжил","VERB",0,"root"),
        w("Фаине","PROPN",2,"iobj","RECIPIENT"),
        w("велосипед","NOUN",2,"obj","PATIENT"),p(2)),
      _make("verb_loan_contrast_fronted_21","ASSERTED",
        w("Велосипед","NOUN",4,"obj","PATIENT"),
        w("Фаине","PROPN",4,"iobj","RECIPIENT"),
        w("сегодня","ADV",4,"advmod"),
        w("одолжил","VERB",0,"root"),
        w("Михаил","PROPN",4,"nsubj","AGENT"),p(4)),
      _make("multiclause_conjunction_pronoun_22","ASSERTED",
        w("Пока","SCONJ",3,"mark"),
        w("Мира","PROPN",3,"nsubj","AGENT"),
        w("ждала","VERB",6,"advcl"),comma(3),
        w("Феликс","PROPN",6,"nsubj","AGENT"),
        w("передал","VERB",0,"root"),
        w("ей","PRON",6,"iobj","RECIPIENT"),
        w("карту","NOUN",6,"obj","PATIENT"),p(6)),
      _make("multiclause_motion_and_transfer_23","ASSERTED",
        w("Когда","SCONJ",3,"mark"),
        w("вагон","NOUN",3,"nsubj"),
        w("остановился","VERB",6,"advcl"),comma(3),
        w("Клавдия","PROPN",6,"nsubj","AGENT"),
        w("передала","VERB",0,"root"),
        w("Леону","PROPN",6,"iobj","RECIPIENT"),
        w("чемодан","NOUN",6,"obj","PATIENT"),p(6)),
      _make("multiclause_antecedent_unscored_24","ASSERTED",
        w("Когда","SCONJ",3,"mark"),
        w("Аксинья","PROPN",3,"nsubj","AGENT"),
        w("вернулась","VERB",6,"advcl"),comma(3),
        w("Тихон","PROPN",6,"nsubj","AGENT"),
        w("отдал","VERB",0,"root"),
        w("ей","PRON",6,"iobj","RECIPIENT"),
        w("альбом","NOUN",6,"obj","PATIENT"),p(6)),
    )
    validate_new_eval24(corpus)
    return corpus


def gold_links(row:EVALSentence)->tuple[tuple[int,int,str],...]:
    """Only at EVAL scoring, NEVER to enumerate model candidates.

    Exactly S16 TRAIN definition: direct head is a VERB, current word
    is role argument; gold is 1-based PREDICATE→ARGUMENT direction.
    Gold for root/aux/punctuation/indirect arcs is deliberately NONE.
    """
    verbs={w.index for w in row.words if w.upos=="VERB"}
    return tuple(sorted((w.head,w.index,w.role)
        for w in row.words
        if w.role in ROLES[1:] and w.index not in verbs and w.head in verbs))


def validate_new_eval24(rows:tuple[EVALSentence,...])->None:
    if (type(rows) is not tuple or len(rows)!=24
        or len({r.text_sha256() for r in rows})!=24
        or len({r.construction for r in rows})!=24
        or any(r.split!=SPLIT for r in rows)):
        raise ValueError("S17 requires 24 unique NEW sealed EVAL samples")
    old_text_sha={x.sha256() for x in build_s9_train72()} | {
                 x.sha256() for x in old_s4_train()}
    if {r.text_sha256() for r in rows}&old_text_sha:
        raise ValueError("S17 text overlaps original S9 or S4 TRAIN")
    role_count={z:0 for z in ROLES[1:]}
    orientation=[0,0];negated=0;future=0;multiclause=0;passive=0
    for r in rows:
        x=r.words
        if (len(x)<5 or x[-1].text!="." or not r.text.endswith(".")
            or [w.index for w in x]!=list(range(1,len(x)+1))
            or sum(w.head==0 for w in x)!=1
            or any(w.head<0 or w.head>len(x) or w.head==w.index
                or w.role not in ROLES or
                r.text[w.start:w.end]!=w.text or
                w.start>=w.end
                for w in x)):
            raise ValueError("S17 invalid UD syntax/semantic Unicode offset")
        if r.event_modality not in ("ASSERTED","NEGATED","FUTURE_UNREALIZED"):
            raise ValueError("S17 invalid event-status gold annotation")
        if ("не" in [w.text.casefold() for w in x])!=(r.event_modality=="NEGATED"):
            raise ValueError("S17 negation event metadata drift")
        if sum(w.upos=="VERB" for w in x)>1:multiclause+=1
        if any("pass" in w.deprel for w in x):passive+=1
        if r.event_modality=="NEGATED":negated+=1
        if r.event_modality=="FUTURE_UNREALIZED":future+=1
        for w in x:
            if w.head:
                cursor=w.head
                path={w.index}
                while cursor:
                    if cursor in path:raise ValueError("S17 cyclic UD parse")
                    path.add(cursor)
                    cursor=x[cursor-1].head
            if w.role!="NONE":
                if w.head<=0 or x[w.head-1].upos!="VERB":
                    raise ValueError("S17 semantic label must have direct verbal governor")
        edges=gold_links(r)
        if not 2<=len(edges)<=5:
            raise ValueError("S17 must have >=2 genuine directed verb argument links")
        for pred,arg,role in edges:
            role_count[role]+=1
            orientation[int(pred<arg)]+=1
    if (min(role_count.values())<2 or
        min(orientation)<20 or negated<2 or future<2 or
        passive<3 or multiclause<3):
        raise ValueError("S17 novel graph label coverage insufficient")


def canonical_eval24(rows:tuple[EVALSentence,...])->bytes:
    validate_new_eval24(rows)
    objects=[{
      "split":r.split,"text":r.text,"sha256":r.text_sha256(),
      "construction":r.construction,"event_modality":r.event_modality,
      "words":[{"index":w.index,"form":w.text,
                "upos":w.upos,"head":w.head,"deprel":w.deprel,
                "semantic_role":w.role,"start":w.start,"end":w.end}
               for w in r.words],
      "directed_gold_links":[
          {"predicate_word":p,"argument_word":a,"role":role}
          for p,a,role in gold_links(r)]
    } for r in rows]
    return json.dumps(objects,sort_keys=True,
            separators=(",",":"),ensure_ascii=False).encode("utf-8")


def preregister_new_eval24()->dict:
    rows=build_new_eval24()
    encoded=canonical_eval24(rows)
    sha=hashlib.sha256(encoded).hexdigest()
    if S17_EVAL24_COMPLETE_SHA and sha!=S17_EVAL24_COMPLETE_SHA:
        raise ValueError("S17 EVAL24 mutated after first source-only SHA")
    gold=[link for r in rows for link in gold_links(r)]
    rcounts={role:sum(z[2]==role for z in gold) for role in ROLES[1:]}
    return {
      "stage":"NRA3_S17_ONE_TIME_NEW_DIRECTED_RUSSIAN24_SOURCE_ONLY_SHA_PRECOMMIT",
      "S17_new_Russian_EVAL_sentence_count":len(rows),
      "S17_new_Russian_EVAL_complete_UD_and_directed_link_SHA256":sha,
      "S17_new_Russian_EVAL_text_SHA_order":[r.text_sha256() for r in rows],
      "S17_new_gold_predicate_argument_links":len(gold),
      "S17_new_gold_directed_role_counts":rcounts,
      "S16b_frozen_physical_source_run_BEFORE_EVAL":S16B_FROZEN_SOURCE_RUN,
      "S16b_frozen_physical_SHA_BEFORE_EVAL":S16B_FROZEN_SOURCE_SHA,
      "S16b_frozen_head_SHA_BEFORE_EVAL":S16B_FROZEN_FULL_HEAD_SHA,
      "S16b_directed_train_gold_complete_SHA":S16B_ORIGINAL_TRAIN72_GOLD_SHA,
      "S17_one_time_holdout_gold_loaded_into_model":False,
      "S17_model_forward_calls":0,
      "S17_model_evaluation_scored":False,
      "S17_optimizer_steps":0,
      "S17_old_consumed_S15c_EVAL24_never_loaded":True,
      "S17_old_consumed_S12c_EVAL16_never_loaded":True,
      "S17_old_five_EVAL128_and_reserved_sixth_S3CI_never_loaded":True,
      "S17_small_handauthored_source_not_general_Russian_benchmark":True,
      "S17_event_truth_and_pronoun_coreference_NOT_scored_by_valence_head":True,
      "S17_original_Calibri43M_GT_nativeC_NRA_not_loaded":True,
      "S17_no_LIVE_Jessica_chat_Render_Supabase_APK":True,
      "output_gate":0,
      "candidate_accepted":False,
      "activation_allowed":False
    }
