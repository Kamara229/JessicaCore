"""NRA3 S18 source-only new 96 Russian TRAIN: multi-verb role graph curriculum.

WHY: independently frozen original43M+GT/S16b full native CPU S17c
failed on a NEW 24-sentence EVAL (heldout exact directed-role F1 15.93%
vs S16b original 72 TRAIN F1 86.60%). These 24 S17c EVAL (and other
earlier consumed EVALs) are NEVER used as supervision, data lookup,
lexical choice source or trial-by-error tuning here.

S18 creates *additional TRAIN* from explicitly prewritten 12 lexical
families and eight morphosyntactic constructions (including passive,
object-fronting, negation, future, receipt/source and independent
two-predicate subordinate clauses). 96 TRAIN texts, not independent
generalization tests. This is a limited controlled synthetic teaching
curriculum; a larger externally sourced human reference corpus would
be necessary to claim natural Russian language generalization.

During inference, original 43M/GT, CB59 native C and new rank8 may
only access raw texts and real wordpiece features; UD/roles remain on
optimizer-side labels. No training, model calls, or score here.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from research.nra3_s9.new_ru_train72 import build_s9_train72
from research.nra3_s4.russian_ud_train_only import build_train as build_s4_train
from research.nra3_s16.paid_directed_valence import ROLES

S18_SPLIT="NRA3_S18_12_FAMILIES_8_MORPHOSYNTACTIC_CONSTRUCTIONS_NEW96_TRAIN_ONLY"
S18_CANONICAL_96_FULL_UD_AND_ROLES_SHA256=""  # pin after FIRST SOURCE-ONLY CI
S18_PARENT_FROZEN_S16B_MODEL_SHA256="3fe4bbce718e53e4e812e94367ce25834f873cbed646479c59b1e3949d2ecf38"
S18_PARENT_S9_DIRECTED_72_TRAIN_GOLD_SHA256="9843c35ee9aa75b8f5c9c963e441589dd2f0292e41ec7933971d5aa3dcd365d6"
CONSTRUCTIONS=(
 "active_distinct_verb",
 "fronted_patient_distinct_verb",
 "passive_agent_patient_inversion",
 "recipient_subject_with_source",
 "negated_direct_roles_only",
 "future_direct_roles_only",
 "adjective_and_adverb_modifiers",
 "two_finite_predicates_distinct_agents",
)
EVENT_MODALITY={
 CONSTRUCTIONS[4]:"NEGATED",
 CONSTRUCTIONS[5]:"FUTURE_UNREALIZED",
}


@dataclass(frozen=True)
class LexicalFamily:
    agent_nom:str
    agent_gen:str
    agent_ins:str
    recipient_nom:str
    recipient_dat:str
    object_acc:str
    object_nom:str
    verb_past:str
    verb_passive:str
    verb_future:str
    verb_lemma:str
    object_lemma:str

FAMILIES=(
 LexicalFamily("Антон","Антона","Антоном","Валерия","Валерии","схему","схема","выслал","выслана","вышлет","выслать","схема"),
 LexicalFamily("Богдан","Богдана","Богданом","Софья","Софье","книгу","книга","подарил","подарена","подарит","подарить","книга"),
 LexicalFamily("Владимир","Владимира","Владимиром","Елена","Елене","картину","картина","принёс","принесена","принесёт","принести","картина"),
 LexicalFamily("Давид","Давида","Давидом","Надежда","Надежде","медаль","медаль","вручил","вручена","вручит","вручить","медаль"),
 LexicalFamily("Кирилл","Кирилла","Кириллом","Любовь","Любови","записку","записка","показал","показана","покажет","показать","записка"),
 LexicalFamily("Леонид","Леонида","Леонидом","Маргарита","Маргарите","телеграмму","телеграмма","отправил","отправлена","отправит","отправить","телеграмма"),
 LexicalFamily("Матвей","Матвея","Матвеем","Полина","Полине","заявку","заявка","подал","подана","подаст","подать","заявка"),
 LexicalFamily("Назар","Назара","Назаром","Ксения","Ксении","ведомость","ведомость","оформил","оформлена","оформит","оформить","ведомость"),
 LexicalFamily("Платон","Платона","Платоном","Роза","Розе","открытку","открытка","прислал","прислана","пришлёт","прислать","открытка"),
 LexicalFamily("Руслан","Руслана","Русланом","Нелли","Нелли","фотографию","фотография","переслал","переслана","перешлёт","переслать","фотография"),
 LexicalFamily("Степан","Степана","Степаном","Эмма","Эмме","коробку","коробка","доставил","доставлена","доставит","доставить","коробка"),
 LexicalFamily("Фёдор","Фёдора","Фёдором","Инга","Инге","посуду","посуда","отнёс","отнесена","отнесёт","отнести","посуда"),
)

@dataclass(frozen=True)
class TrainWord:
    index:int
    form:str
    lemma:str
    upos:str
    head:int
    deprel:str
    semantic_role:str
    start:int
    end:int

@dataclass(frozen=True)
class TrainSentence:
    text:str
    words:tuple[TrainWord,...]
    construction:str
    lexical_family:int
    event_modality:str
    split:str=S18_SPLIT

    def text_sha256(self):
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def _w(form,lemma,upos,head,dep,role="NONE"):
    return (form,lemma,upos,head,dep,role)


def _sentence(fid:int,mode:str,*spec)->TrainSentence:
    forms=[z[0] for z in spec]
    if forms[-1]!=".":
        raise ValueError("S18 source cannot omit period")
    text=" ".join(forms).replace(" .",".").replace(" ,",",")
    words=[];cursor=0
    for n,(form,lemma,upos,head,dep,role) in enumerate(spec,1):
        start=text.find(form,cursor)
        if start<cursor:
            raise ValueError("S18 source Unicode offsets invalid")
        end=start+len(form)
        words.append(TrainWord(n,form,lemma,upos,head,dep,role,start,end))
        cursor=end
    return TrainSentence(text,tuple(words),mode,fid,EVENT_MODALITY.get(mode,"ASSERTED"))


def _eight(f:LexicalFamily,fid:int)->tuple[TrainSentence,...]:
    w=_w
    N=lambda s,lemma,head,dep,role:w(s,lemma,"PROPN",head,dep,role)
    O=lambda form,head,dep:w(form,f.object_lemma,"NOUN",head,dep,"PATIENT")
    V=lambda form,head,dep:w(form,f.verb_lemma,"VERB",head,dep)
    P=lambda head:w(".",".","PUNCT",head,"punct")
    a=f.agent_nom
    d=f.recipient_dat
    o=f.object_acc
    c=f.object_nom.capitalize()
    ret=(
      _sentence(fid,CONSTRUCTIONS[0],
        N(a,a,2,"nsubj","AGENT"),V(f.verb_past,0,"root"),
        N(d,f.recipient_nom,2,"iobj","RECIPIENT"),
        O(o,2,"obj"),P(2)),
      _sentence(fid,CONSTRUCTIONS[1],
        O(o.capitalize(),3,"obj"),
        N(d,f.recipient_nom,3,"iobj","RECIPIENT"),
        V(f.verb_past,0,"root"),
        N(a,a,3,"nsubj","AGENT"),P(3)),
      _sentence(fid,CONSTRUCTIONS[2],
        O(c,3,"nsubj:pass"),
        w("была","быть","AUX",3,"aux"),
        V(f.verb_passive,0,"root"),
        N(f.agent_ins,a,3,"obl:agent","AGENT"),
        N(d,f.recipient_nom,3,"iobj","RECIPIENT"),P(3)),
      _sentence(fid,CONSTRUCTIONS[3],
        N(f.recipient_nom,f.recipient_nom,2,"nsubj","RECIPIENT"),
        w("получила","получить","VERB",0,"root"),
        O(o,2,"obj"),
        w("от","от","ADP",5,"case"),
        N(f.agent_gen,a,2,"obl","SOURCE"),P(2)),
      _sentence(fid,CONSTRUCTIONS[4],
        N(a,a,3,"nsubj","AGENT"),
        w("не","не","PART",3,"advmod"),
        V(f.verb_past,0,"root"),
        O(o,3,"obj"),
        N(d,f.recipient_nom,3,"iobj","RECIPIENT"),P(3)),
      _sentence(fid,CONSTRUCTIONS[5],
        w("Завтра","завтра","ADV",3,"advmod"),
        N(a,a,3,"nsubj","AGENT"),
        V(f.verb_future,0,"root"),
        N(d,f.recipient_nom,3,"iobj","RECIPIENT"),
        O(o,3,"obj"),P(3)),
      _sentence(fid,CONSTRUCTIONS[6],
        w("Небольшую","небольшой","ADJ",2,"amod"),
        O(o,4,"obj"),
        N(a,a,4,"nsubj","AGENT"),
        V(f.verb_past,0,"root"),
        N(d,f.recipient_nom,4,"iobj","RECIPIENT"),
        w("сегодня","сегодня","ADV",4,"advmod"),P(4)),
      _sentence(fid,CONSTRUCTIONS[7],
        w("Пока","пока","SCONJ",3,"mark"),
        w("сосед","сосед","NOUN",3,"nsubj","AGENT"),
        w("работал","работать","VERB",6,"advcl"),
        w(",",",","PUNCT",3,"punct"),
        N(a,a,6,"nsubj","AGENT"),
        V(f.verb_past,0,"root"),
        N(d,f.recipient_nom,6,"iobj","RECIPIENT"),
        O(o,6,"obj"),P(6)),
    )
    return ret


def build_s18_new_train96()->tuple[TrainSentence,...]:
    rows=tuple(row for i,f in enumerate(FAMILIES) for row in _eight(f,i))
    validate_s18_train96(rows)
    return rows


def direct_predicate_argument_gold(row:TrainSentence)->tuple[tuple[int,int,str],...]:
    verbs={w.index for w in row.words if w.upos=="VERB"}
    return tuple(sorted(
        (w.head,w.index,w.semantic_role) for w in row.words
        if w.semantic_role in ROLES[1:]
        and w.index not in verbs and w.head in verbs))


def validate_s18_train96(rows:tuple[TrainSentence,...])->None:
    if (type(rows) is not tuple or len(rows)!=96 or len(FAMILIES)!=12 or
        len(CONSTRUCTIONS)!=8 or len({r.text_sha256() for r in rows})!=96 or
        {(r.lexical_family,r.construction) for r in rows}!={
            (i,c) for i in range(12) for c in CONSTRUCTIONS} or
        any(r.split!=S18_SPLIT for r in rows)):
        raise ValueError("S18 old TRAIN96 source changed or split is not TRAIN")
    known={r.sha256() for r in build_s9_train72()} | {
        r.sha256() for r in build_s4_train()}
    if known & {r.text_sha256() for r in rows}:
        raise ValueError("S18 source reused original TRAIN texts")
    # Across 12 families: 8 AGENT, 8 PATIENT, 8 RECIPIENT, 1 SOURCE
    # per family. Multi-clause has 2 agents, recipient-subject 0 agents.
    expected={"AGENT":96,"PATIENT":96,"RECIPIENT":96,"SOURCE":12}
    edges=[z for r in rows for z in direct_predicate_argument_gold(r)]
    observed={role:sum(z[2]==role for z in edges) for role in ROLES[1:]}
    if observed!=expected or len(edges)!=300:
        raise ValueError(f"S18 full verb→argument TRAIN labels drift: {observed}")
    for r in rows:
        w=r.words
        if (r.construction not in CONSTRUCTIONS or
            not 0<=r.lexical_family<12 or
            len(w)<5 or w[-1].form!="." or not r.text.endswith(".") or
            [x.index for x in w]!=list(range(1,len(w)+1)) or
            sum(x.head==0 for x in w)!=1 or
            any(x.head<0 or x.head>len(w) or x.head==x.index or
                x.semantic_role not in ROLES or
                r.text[x.start:x.end]!=x.form or x.start>=x.end for x in w)):
            raise ValueError("S18 malformed U(D), offsets or semantic roles")
        for x in w:
            chain={x.index}
            h=x.head
            while h:
                if h in chain:raise ValueError("S18 cyclic UD source")
                chain.add(h)
                h=w[h-1].head
            if x.semantic_role!="NONE":
                if x.head==0 or w[x.head-1].upos!="VERB":
                    raise ValueError("S18 direct semantic role lacks actual verb head")
        n=sum(z.semantic_role=="AGENT" for z in w)
        if r.construction==CONSTRUCTIONS[7]:
            if n!=2 or sum(x.upos=="VERB" for x in w)!=2:
                raise ValueError("S18 two-predicate multiclause grammar lost")
        elif r.construction==CONSTRUCTIONS[3]:
            if n!=0 or sum(x.semantic_role=="SOURCE" for x in w)!=1:
                raise ValueError("S18 recipient vs agent conflation")
        elif n!=1:
            raise ValueError("S18 single source AGENT link lost")
        if r.construction==CONSTRUCTIONS[2]:
            if not any(x.deprel=="nsubj:pass" and x.semantic_role=="PATIENT" for x in w):
                raise ValueError("S18 passive patient inverted to AGENT")
        if r.construction==CONSTRUCTIONS[4] and not any(z.form=="не" for z in w):
            raise ValueError("S18 negative marker lost")
        if r.construction==CONSTRUCTIONS[5] and r.event_modality!="FUTURE_UNREALIZED":
            raise ValueError("S18 grammatical future modality not registered")
        if (r.construction==CONSTRUCTIONS[4])!=(r.event_modality=="NEGATED"):
            raise ValueError("S18 negated event metadata changed")


def canonical_s18_new96_train(rows:tuple[TrainSentence,...])->bytes:
    validate_s18_train96(rows)
    record=[{
        "split":r.split,
        "construction":r.construction,
        "lexical_family":r.lexical_family,
        "event_modality":r.event_modality,
        "text":r.text,"text_sha256":r.text_sha256(),
        "words":[
          {"index":w.index,"form":w.form,"lemma":w.lemma,"upos":w.upos,
           "head":w.head,"deprel":w.deprel,"semantic_role":w.semantic_role,
           "start":w.start,"end":w.end}
          for w in r.words],
        "gold_directed_predicate_argument_links":[
           {"predicate_word":p,"argument_word":a,"semantic_role":role}
           for p,a,role in direct_predicate_argument_gold(r)]
    } for r in rows]
    return json.dumps(record,sort_keys=True,separators=(",",":"),
                      ensure_ascii=False).encode("utf-8")


def seal_s18_new96_train()->dict:
    source=canonical_s18_new96_train(build_s18_new_train96())
    sha=hashlib.sha256(source).hexdigest()
    if S18_CANONICAL_96_FULL_UD_AND_ROLES_SHA256 and sha!=S18_CANONICAL_96_FULL_UD_AND_ROLES_SHA256:
        raise ValueError("S18 TRAIN-only source SHA changed after seal")
    rows=build_s18_new_train96()
    edges=[z for r in rows for z in direct_predicate_argument_gold(r)]
    return {
        "stage":"NRA3_S18_NEW96_RU_12VERB_8STRUCTURES_300_DIRECTIONAL_LINKS_SOURCE_ONLY_TRAIN",
        "S18_complete_original_96_new_TRAIN_full_annotation_SHA256":sha,
        "S18_original_source_rows":96,
        "S18_new_lexical_families":12,
        "S18_new_distinct_verb_lemmas":len({f.verb_lemma for f in FAMILIES}),
        "S18_new_syntactic_constructions":8,
        "S18_new_directed_verb_to_argument_links":len(edges),
        "S18_grammatical_role_link_counts":{role:sum(z[2]==role for z in edges)
          for role in ROLES[1:]},
        "S18_frozen_original_S16b_2309FP16_SHA":S18_PARENT_FROZEN_S16B_MODEL_SHA256,
        "S18_original_S9_TRAIN72_gold_SHA":S18_PARENT_S9_DIRECTED_72_TRAIN_GOLD_SHA256,
        "S18_uses_new_additional_TRAIN_only":True,
        "S18_old_S17c_24_independent_EVAL_now_consumed_not_read":True,
        "S18_old_S15c_24_and_S12c_16_consumed_EVAL_never_loaded":True,
        "S18_old_five_EVAL128_and_reserved_sixth_S3CI_never_loaded":True,
        "S18_original_43M_GT_NRA_C_or_2309FP16_model_loaded":False,
        "S18_original_S16b_model_trained_or_updated":False,
        "S18_new_optimizer_steps":0,
        "S18_separate_future_unseen_EVAL_generated_or_scored":False,
        "S18_original_parent_616_J7_UNCHANGED_no_active_candidate_test":True,
        "S18_no_LIVE_Jessica_chat_Render_Supabase_APK":True,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False
    }
