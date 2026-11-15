"""NRA2 S15: preregister exactly 24 NEW Russian heldout linguistic EVAL prompts.

S14 2,581-byte rank8 UPOS-locked FP16 physical candidate (SHA
5b96e2174e33da5232fcc2a4a397b6fab4370b6879adb684d75fea8a503f1279)
and its pre-committed 48 TRAIN + 24 INTERNAL selection were frozen in
successful GitHub Actions #43740415883 *before* S15 source exists.

This is source/label pre-registration ONLY. NO Calibri43M/GT/NRA
inference, S14 scoring, training, hidden capture or model selection.
No S12c consumed EVAL16, old five EVAL128 or reserved sixth S3CI
data are loaded, imported, compared or opened. No Jessica chat tests.

The examples are constructed by hand and not a representative corpus.
They purposely introduce unseen lexical families, order, passive,
negation, future tense, pronominal anaphora, instruments, destination,
experiencers and multi-clause structures. Some underlying grammatical
constructions overlap prior TRAIN; correctness on this small dataset
would NOT demonstrate broad language understanding.

Gold roles are grammatical argument roles, NOT proof an event occurred.
Coreference links are gold annotation, NEVER supplied to model input.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json

from research.nra2_bridge.stage3ck_russian_ud_role_bridge import UDWord
from research.nra3_s0.neural_area import UPOS,MORPH,SEMANTIC_ROLES
from research.nra3_s4.russian_ud_train_only import build_train as prior_s4_train
from research.nra3_s9.new_ru_train72 import build_s9_train72,S9_TRAIN72_CANONICAL_SHA256

SPLIT="NRA3_S15_NEW_RU24_ONE_TIME_HUMAN_ANNOTATED_EVAL_NEVER_TRAIN"
S14_FROZEN_CI=43740415883
S14_FROZEN_HEAD_SHA="ab95483ef5dc985b8a32941163fb248bc9640ef3"
S14_SELECTED_FP16_SHA="5b96e2174e33da5232fcc2a4a397b6fab4370b6879adb684d75fea8a503f1279"
S14_INPUT_S10_REAL43M_HIDDEN_SHA="ed262984c0fd48b60987308e7b135be7d05d3e5434fe9923f373aec178ff2b46"
S8B_FROZEN_NATIVE_43_CB59_SHA="d710aed710a44719b6003434a2dd83150861928ce167364cb92329beabf7be97"
S11_FROZEN_39FP16_SHA="05ce14e20c9f4e7e09f85dbd9bd46f0dc0a700d72f1a96ffa921b2e822f235c3"
# Hard-lock AFTER the first source-only CI and BEFORE ANY model scoring.
EVAL24_CANONICAL_SHA=""


@dataclass(frozen=True)
class S15NovelEval:
    text:str
    words:tuple[UDWord,...]
    roles:tuple[str,...]
    construction:str
    event_truth:str
    antecedents:tuple[tuple[int,int],...]
    split:str=SPLIT

    def sha256(self)->str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


# (form, lemma, UPOS, 1-based head, UD relation, tuple(morph), semantic role)
Item=tuple[str,str,str,int,str,tuple[str,...],str]
N=("Case=Nom","Number=Sing")
A=("Case=Acc","Number=Sing")
D=("Case=Dat","Number=Sing")
G=("Case=Gen","Number=Sing")
I=("Case=Ins","Number=Sing")
LOC=("Case=Loc","Number=Sing")
P=("Tense=Past","Voice=Act")
PASS=("Tense=Past","Voice=Pass")
FUT=("Tense=Fut","Voice=Act")
PL=("Case=Acc","Number=Plur")
PU=(".", ".", "PUNCT",0,"punct",(),"NONE")


def _make(category:str,truth:str,spec:tuple[Item,...],
          coref:tuple[tuple[int,int],...]=())->S15NovelEval:
    if not spec or spec[-1][0]!=".":
        raise ValueError("S15 must have full stop terminal token")
    text=" ".join(w[0] for w in spec)
    for punc in (".",",","!","?"):
        text=text.replace(" "+punc,punc)
    cursor=0
    words=[];roles=[]
    for i,(form,lemma,tag,head,rel,morph,role) in enumerate(spec,1):
        start=text.find(form,cursor)
        end=start+len(form)
        if start<cursor or text[start:end]!=form:
            raise ValueError("S15 exact per-word Unicode offsets required")
        words.append(UDWord(i,form,lemma,tag,head,rel,start,end,morph))
        roles.append(role);cursor=end
    return S15NovelEval(text,tuple(words),tuple(roles),category,truth,coref)


def new_russian_eval24()->tuple[S15NovelEval,...]:
    items=[]
    def add(category,truth,*spec,coref=()):
        items.append(_make(category,truth,tuple(spec),tuple(coref)))
    def e(h):return (".",".","PUNCT",h,"punct",(),"NONE")
    def comma(h):return (",",",","PUNCT",h,"punct",(),"NONE")
    def w(s,l,u,h,r,m=(),v="NONE"):return (s,l,u,h,r,m,v)

    add("01_unseen_gift_agent_recipient_patient","ASSERTED",
        w("Лариса","Лариса","PROPN",2,"nsubj",N,"AGENT"),
        w("отправила","отправить","VERB",0,"root",P),
        w("Виталию","Виталий","PROPN",2,"iobj",D,"RECIPIENT"),
        w("рукопись","рукопись","NOUN",2,"obj",A,"PATIENT"),e(2))

    add("02_passive_reverse_roles_unseen_stem","ASSERTED",
        w("Рукопись","рукопись","NOUN",3,"nsubj:pass",N,"PATIENT"),
        w("была","быть","AUX",3,"aux"),
        w("передана","передать","VERB",0,"root",PASS),
        w("Ларисой","Лариса","PROPN",3,"obl:agent",I,"AGENT"),
        w("Виталию","Виталий","PROPN",3,"iobj",D,"RECIPIENT"),e(3))

    add("03_receive_from_source_not_agent","ASSERTED",
        w("Виталий","Виталий","PROPN",2,"nsubj",N,"RECIPIENT"),
        w("получил","получить","VERB",0,"root",P),
        w("рукопись","рукопись","NOUN",2,"obj",A,"PATIENT"),
        w("от","от","ADP",5,"case"),
        w("Ларисы","Лариса","PROPN",2,"obl",G,"SOURCE"),e(2))

    add("04_negated_transitive_event_did_not_occur","NEGATED",
        w("Марфа","Марфа","PROPN",3,"nsubj",N,"AGENT"),
        w("не","не","PART",3,"advmod"),
        w("передала","передать","VERB",0,"root",P),
        w("Дмитрию","Дмитрий","PROPN",3,"iobj",D,"RECIPIENT"),
        w("схему","схема","NOUN",3,"obj",A,"PATIENT"),e(3))

    add("05_future_delivery_no_occurred_event","FUTURE_UNREALIZED",
        w("Завтра","завтра","ADV",3,"advmod"),
        w("Клим","Клим","PROPN",3,"nsubj",N,"AGENT"),
        w("доставит","доставить","VERB",0,"root",FUT),
        w("Аделине","Аделина","PROPN",3,"iobj",D,"RECIPIENT"),
        w("посылку","посылка","NOUN",3,"obj",A,"PATIENT"),e(3))

    add("06_fronted_accusative_future_sender_after_verb","FUTURE_UNREALIZED",
        w("Посылку","посылка","NOUN",3,"obj",A,"PATIENT"),
        w("Аделине","Аделина","PROPN",3,"iobj",D,"RECIPIENT"),
        w("доставит","доставить","VERB",0,"root",FUT),
        w("Клим","Клим","PROPN",3,"nsubj",N,"AGENT"),e(3))

    add("07_adjective_object_attachment","ASSERTED",
        w("Снежана","Снежана","PROPN",2,"nsubj",N,"AGENT"),
        w("вручила","вручить","VERB",0,"root",P),
        w("Макару","Макар","PROPN",2,"iobj",D,"RECIPIENT"),
        w("тяжёлую","тяжёлый","ADJ",5,"amod",A),
        w("папку","папка","NOUN",2,"obj",A,"PATIENT"),e(2))

    add("08_adjective_fronted_patient_and_reverse_subject","ASSERTED",
        w("Тяжёлую","тяжёлый","ADJ",2,"amod",A),
        w("папку","папка","NOUN",4,"obj",A,"PATIENT"),
        w("Макару","Макар","PROPN",4,"iobj",D,"RECIPIENT"),
        w("вручила","вручить","VERB",0,"root",P),
        w("Снежана","Снежана","PROPN",4,"nsubj",N,"AGENT"),e(4))

    add("09_two_temporal_adverbials_plural_patient","ASSERTED",
        w("Поздно","поздно","ADV",2,"advmod"),
        w("вечером","вечером","ADV",4,"advmod"),
        w("Федот","Федот","PROPN",4,"nsubj",N,"AGENT"),
        w("передал","передать","VERB",0,"root",P),
        w("Иветте","Иветта","PROPN",4,"iobj",D,"RECIPIENT"),
        w("ключи","ключ","NOUN",4,"obj",PL,"PATIENT"),e(4))

    add("10_location_prepositional_goal","ASSERTED",
        w("Роберт","Роберт","PROPN",2,"nsubj",N,"AGENT"),
        w("положил","положить","VERB",0,"root",P),
        w("карту","карта","NOUN",2,"obj",A,"PATIENT"),
        w("на","на","ADP",5,"case"),
        w("стол","стол","NOUN",2,"obl",A,"DESTINATION"),e(2))

    add("11_object_taken_from_source","ASSERTED",
        w("Татьяна","Татьяна","PROPN",2,"nsubj",N,"AGENT"),
        w("забрала","забрать","VERB",0,"root",P),
        w("сумку","сумка","NOUN",2,"obj",A,"PATIENT"),
        w("у","у","ADP",5,"case"),
        w("Бориса","Борис","PROPN",2,"obl",G,"SOURCE"),e(2))

    add("12_transport_of_object_from_location","ASSERTED",
        w("Алла","Алла","PROPN",2,"nsubj",N,"AGENT"),
        w("принесла","принести","VERB",0,"root",P),
        w("лекарства","лекарство","NOUN",2,"obj",PL,"PATIENT"),
        w("со","со","ADP",5,"case"),
        w("склада","склад","NOUN",2,"obl",G,"SOURCE"),e(2))

    add("13_tool_is_instrument_not_agent","ASSERTED",
        w("Назар","Назар","PROPN",2,"nsubj",N,"AGENT"),
        w("открыл","открыть","VERB",0,"root",P),
        w("ящик","ящик","NOUN",2,"obj",A,"PATIENT"),
        w("отвёрткой","отвёртка","NOUN",2,"obl",I,"INSTRUMENT"),e(2))

    add("14_perception_experiencer_distinct_from_agent","ASSERTED",
        w("Валерия","Валерия","PROPN",2,"nsubj",N,"EXPERIENCER"),
        w("заметила","заметить","VERB",0,"root",P),
        w("ошибку","ошибка","NOUN",2,"obj",A,"PATIENT"),e(2))

    add("15_stative_adjective_location","ASSERTED",
        w("Старая","старый","ADJ",2,"amod",N),
        w("карта","карта","NOUN",3,"nsubj",N,"PATIENT"),
        w("лежала","лежать","VERB",0,"root",P),
        w("на","на","ADP",5,"case"),
        w("полке","полка","NOUN",3,"obl",LOC,"LOCATION"),e(3))

    add("16_coref_subject_and_object_after_conjunction","ASSERTED",
        w("Кира","Кира","PROPN",2,"nsubj",N,"AGENT"),
        w("нашла","найти","VERB",0,"root",P),
        w("письмо","письмо","NOUN",2,"obj",A,"PATIENT"),
        w("и","и","CCONJ",6,"cc"),
        w("она","она","PRON",6,"nsubj",N,"AGENT"),
        w("передала","передать","VERB",2,"conj",P),
        w("его","он","PRON",6,"obj",A,"PATIENT"),
        w("Тимофею","Тимофей","PROPN",6,"iobj",D,"RECIPIENT"),e(6),
        coref=((5,1),(7,3)))

    add("17_two_coref_pronouns_one_omitted_recipient","ASSERTED",
        w("Софья","Софья","PROPN",2,"nsubj",N,"AGENT"),
        w("взяла","взять","VERB",0,"root",P),
        w("тетрадь","тетрадь","NOUN",2,"obj",A,"PATIENT"),
        w("и","и","CCONJ",6,"cc"),
        w("она","она","PRON",6,"nsubj",N,"AGENT"),
        w("спрятала","спрятать","VERB",2,"conj",P),
        w("её","она","PRON",6,"obj",A,"PATIENT"),e(6),
        coref=((5,1),(7,3)))

    add("18_unresolved_pronoun_no_fake_antecedent","ASSERTED",
        w("Он","он","PRON",2,"nsubj",N,"AGENT"),
        w("вернул","вернуть","VERB",0,"root",P),
        w("Оксане","Оксана","PROPN",2,"iobj",D,"RECIPIENT"),
        w("журнал","журнал","NOUN",2,"obj",A,"PATIENT"),e(2))

    add("19_conjunction_recipient_pronoun_refers_to_previous_subject","ASSERTED",
        w("Лида","Лида","PROPN",2,"nsubj",N,"EXPERIENCER"),
        w("нашла","найти","VERB",0,"root",P),
        w("маршрут","маршрут","NOUN",2,"obj",A,"PATIENT"),comma(2),
        w("а","а","CCONJ",7,"cc"),
        w("Семён","Семён","PROPN",7,"nsubj",N,"AGENT"),
        w("показал","показать","VERB",2,"conj",P),
        w("ей","она","PRON",7,"iobj",D,"RECIPIENT"),
        w("карту","карта","NOUN",7,"obj",A,"PATIENT"),e(7),
        coref=((8,1),))

    add("20_passive_fronted_recipient","ASSERTED",
        w("Ирине","Ирина","PROPN",3,"iobj",D,"RECIPIENT"),
        w("был","быть","AUX",3,"aux"),
        w("отправлен","отправить","VERB",0,"root",PASS),
        w("отчёт","отчёт","NOUN",3,"nsubj:pass",N,"PATIENT"),
        w("Павлом","Павел","PROPN",3,"obl:agent",I,"AGENT"),e(3))

    add("21_future_passive_not_yet_actualized","FUTURE_UNREALIZED",
        w("Конверт","конверт","NOUN",3,"nsubj:pass",N,"PATIENT"),
        w("будет","быть","AUX",3,"aux"),
        w("доставлен","доставить","VERB",0,"root",PASS),
        w("Ариной","Арина","PROPN",3,"obl:agent",I,"AGENT"),
        w("Ренату","Ренат","PROPN",3,"iobj",D,"RECIPIENT"),e(3))

    add("22_negated_passive_without_actual_transfer","NEGATED",
        w("Документ","документ","NOUN",4,"nsubj:pass",N,"PATIENT"),
        w("не","не","PART",4,"advmod"),
        w("был","быть","AUX",4,"aux"),
        w("передан","передать","VERB",0,"root",PASS),
        w("Фаддеем","Фаддей","PROPN",4,"obl:agent",I,"AGENT"),
        w("Марине","Марина","PROPN",4,"iobj",D,"RECIPIENT"),e(4))

    add("23_when_clause_pronoun_anaphora_and_argument","ASSERTED",
        w("Когда","когда","SCONJ",3,"mark"),
        w("Ника","Ника","PROPN",3,"nsubj",N,"AGENT"),
        w("вернулась","вернуться","VERB",6,"advcl",P),comma(3),
        w("Арсений","Арсений","PROPN",6,"nsubj",N,"AGENT"),
        w("отдал","отдать","VERB",0,"root",P),
        w("ей","она","PRON",6,"iobj",D,"RECIPIENT"),
        w("альбом","альбом","NOUN",6,"obj",A,"PATIENT"),e(6),
        coref=((7,2),))

    add("24_numeral_plural_fronted_patient","ASSERTED",
        w("Две","два","NUM",2,"nummod",PL),
        w("открытки","открытка","NOUN",4,"obj",PL,"PATIENT"),
        w("Вероника","Вероника","PROPN",4,"nsubj",N,"AGENT"),
        w("отправила","отправить","VERB",0,"root",P),
        w("Юрию","Юрий","PROPN",4,"iobj",D,"RECIPIENT"),e(4))

    out=tuple(items)
    validate_new_eval24(out)
    return out


def validate_new_eval24(rows:tuple[S15NovelEval,...])->None:
    if (type(rows) is not tuple or len(rows)!=24 or
        len({r.sha256() for r in rows})!=24 or
        len({r.construction for r in rows})!=24 or
        any(type(r) is not S15NovelEval or r.split!=SPLIT for r in rows)):
        raise ValueError("S15 must have exactly 24 fixed EVAL-only semantic cases")
    # Only prior TRAIN is loaded; no prior heldout EVAL source ever accessed.
    original_train={r.sha256() for r in prior_s4_train()}|{
        r.sha256() for r in build_s9_train72()}
    if set(r.sha256() for r in rows)&original_train:
        raise ValueError("S15 novel EVAL24 accidentally contains prior TRAIN text")
    negated=0; future=0; anaphora=0; pron=0;othercases=set()
    for row in rows:
        words=row.words
        if (not row.text.endswith(".") or
            not words or words[-1].text!="." or
            len(words)!=len(row.roles) or
            [w.word_id for w in words]!=list(range(1,len(words)+1)) or
            sum(w.head==0 for w in words)!=1 or
            row.event_truth not in ("ASSERTED","NEGATED","FUTURE_UNREALIZED") or
            not set(row.roles).issubset(SEMANTIC_ROLES) or
            any(w.upos not in UPOS or w.deprel=="" or w.head<0 or
                w.head>len(words) or w.head==w.word_id or
                any(m not in MORPH for m in w.feats) or
                row.text[w.start:w.end]!=w.text for w in words)):
            raise ValueError("S15 invalid Unicode, morph, UD or event role source")
        if "PATIENT" in row.roles or "RECIPIENT" in row.roles:
            othercases.add(row.construction)
        for w in words:
            seen={w.word_id}
            head=w.head
            while head:
                if head in seen:
                    raise ValueError("S15 UD must be a rooted acyclic graph")
                seen.add(head)
                head=words[head-1].head
        token_forms=[w.text.lower() for w in words]
        has_neg="не" in token_forms
        if has_neg != (row.event_truth=="NEGATED"):
            raise ValueError("S15 negated predicate truth must NOT be assumed")
        if has_neg:negated+=1
        if row.event_truth=="FUTURE_UNREALIZED":future+=1
        if any(w.upos=="PRON" for w in words):pron+=1
        for pronoun,antecedent in row.antecedents:
            if (not all(type(x) is int for x in (pronoun,antecedent)) or
                not 1<=antecedent<pronoun<=len(words) or
                words[pronoun-1].upos!="PRON" or
                words[antecedent-1].upos not in ("NOUN","PROPN","PRON")):
                raise ValueError("S15 coreference gold must point to earlier noun")
            anaphora+=1
    if (negated<2 or future<2 or pron<4 or anaphora<4 or
        len(othercases)!=24):
        raise ValueError("S15 heldout must cover negation/future/pronoun/coref and semantics")


def canonical_new_eval24(rows:tuple[S15NovelEval,...])->bytes:
    validate_new_eval24(rows)
    payload=[{
       "text":r.text,"text_sha256":r.sha256(),
       "construction":r.construction,"split":r.split,
       "event_truth":r.event_truth,
       "antecedents":[list(pair) for pair in r.antecedents],
       "words":[{"id":w.word_id,"text":w.text,"lemma":w.lemma,
                 "upos":w.upos,"head":w.head,"deprel":w.deprel,
                 "start":w.start,"end":w.end,"feats":list(w.feats),
                 "role":r.roles[j]} for j,w in enumerate(r.words)]
      } for r in rows]
    return json.dumps(payload,ensure_ascii=False,sort_keys=True,
                      separators=(",",":")).encode("utf-8")


def sealed_new_eval24()->dict:
    rows=new_russian_eval24()
    digest=hashlib.sha256(canonical_new_eval24(rows)).hexdigest()
    if EVAL24_CANONICAL_SHA and digest!=EVAL24_CANONICAL_SHA:
        raise ValueError("S15 full EVAL24 source changed after original prereg")
    return {
       "stage":"NRA3_S15_SOURCE_ONLY_NEW_RU24_ONE_TIME_PRE_REGISTER_BEFORE_ANY_SCORE",
       "complete_UD_semantic_role_coreference_event_gold_sha256":digest,
       "number_of_new_EVAL24_cases":len(rows),
       "row_text_sha256":[x.sha256() for x in rows],
       "S14_model_frozen_run_BEFORE_S15_source":S14_FROZEN_CI,
       "S14_model_frozen_head_BEFORE_S15_source":S14_FROZEN_HEAD_SHA,
       "S14_physical_fp16_selected_sha_BEFORE_S15_source":S14_SELECTED_FP16_SHA,
       "S8b_original_43_native_C_CB59_sha":S8B_FROZEN_NATIVE_43_CB59_SHA,
       "S11_original_39FP16_sha":S11_FROZEN_39FP16_SHA,
       "S9_prior_training_canonical_sha":S9_TRAIN72_CANONICAL_SHA256,
       "S15_no_overlapping_prior_S4_or_S9_TRAIN_texts":True,
       "S15_prior_S12c_consumed_EVAL16_NOT_LOADED":True,
       "S15_prior_five_consumed_EVAL128_NOT_LOADED":True,
       "S15_reserved_sixth_S3CI_EVAL128_NOT_LOADED":True,
       "S15_new_EVAL24_scored":False,
       "S15_new_EVAL24_model_prefills":0,
       "S15_eval24_optimizer_steps":0,
       "S15_new_EVAL24_is_small_hand_authored_not_broad_natural_language":True,
       "S15_event_truth_is_not_implied_by_grammar_roles":True,
       "S15_anaphora_labels_never_passed_to_model":True,
       "S15_no_original43M_GT_NRA_S14_inference_or_weights_loaded":True,
       "S15_no_LIVE_Jessica_chat_Render_Supabase_APK":True,
       "output_gate":0,
       "candidate_accepted":False,
       "activation_allowed":False
    }
