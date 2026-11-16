"""NRA3 S18c6: ONE-TIME pre-registration of 32 new Russian EVAL sentences.

This file is created AFTER S18c4 actual 2,597-FP16 final rank8 source
and AFTER the S18c5 12-fold TRAIN-internal report. Both candidate models
for this future one-time comparison were frozen BEFORE these 32 text/roles:
  baseline S18c1 2,309 FP16:
    ec598cc7b5cb98fc8945101caee11c8ca4e589abca4500325ec7ac6eddc831df
  candidate S18c4 2,597 FP16:
    ea47451b8d34a8842d75529af4332717599fbd9862dc0cad97c4d58c1803d761

This is an author-written diagnostic HOLDOUT of new raw Russian texts,
not a naturally sampled external corpus. It is *new and reserved* only
if training/tuning does not use ANY labels/hidden/metrics from this file.
Morphology design was informed by earlier TRAIN internal group holdouts,
not by this new source. New examples are deliberately not copies of
S4, S9, S18 TRAIN or prior S12c/S15c/S17c consumed evaluations.
We do not import/open consumed prior EVALs, EVAL128 or reserved S3CI.

S18c6 is SOURCE + gold annotation + cryptographic one-time registration
ONLY. NEVER call models, tokenizer, native C, original 43M, new
candidate, pretrained model, or measured F1 in this stage.
No one-time model scoring until source complete SHA is independently
pinned in source and approved by source-only CI. This stage must NOT
be trained, selected on, rerun for better model results or silently edited.

Metric prereg: exact directed role from EACH observed word pair in both
directions, five labels NONE/AGENT/PATIENT/RECIPIENT/SOURCE, original
wordpiece last-token alignment, exact micro P/R/F1 and per-construction
F1, compare both frozen candidates on SAME EVAL32. Gold used solely in
scoring AFTER all physically frozen predictions, never model inputs.
Full-sentence frozen S95 read-port does NOT establish causal online
word-by-word processing. Neither 616 J7 new candidate nor all-in bpp/
CPU >=90pct compliance is certified here. Output gate zero.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from research.nra3_s18.s18c3_raw_russian_grammar_cues import raw_surface_grammar
from research.nra3_s18.ru96_multiverb_train_source import (
    build_s18_new_train96, S18_CANONICAL_96_FULL_UD_AND_ROLES_SHA256,
)
from research.nra3_s9.new_ru_train72 import build_s9_train72
from research.nra3_s4.russian_ud_train_only import build_train as build_s4_train

SPLIT = "NRA3_S18C6_PREREG_NEW_RUSSIAN_32_ONE_TIME_EVAL_NEVER_TRAIN"
NEW_EVAL_SOURCE_CI = 43919440242  # FIRST SOURCE-ONLY successful CI; no model inference
EVAL32_COMPLETE_CANONICAL_SHA256 = "16cb3964a0c4220afc82ff67756534a180623f0303c92aede6eff48e0416a5fe"  # first SOURCE-ONLY CI #43919440242, frozen before any EVAL inference
S18_TRAIN96_SHA = "abde63c34534d90e00a6ac61808d2453ce0cfc966c83d45377c1fb620999cca9"
PRE_FROZEN_CANDIDATE_TRAIN_CI = 43916875445
PRE_FROZEN_CANDIDATE_2597_FP16_SHA = (
    "ea47451b8d34a8842d75529af4332717599fbd9862dc0cad97c4d58c1803d761"
)
PRE_FROZEN_CONTROL_TRAIN_CI = 43913448392
PRE_FROZEN_CONTROL_2309_FP16_SHA = (
    "ec598cc7b5cb98fc8945101caee11c8ca4e589abca4500325ec7ac6eddc831df"
)
PREVIOUS_12FOLD_TRAIN_ONLY_CI = 43918044123
CATEGORIES = (
    "01_active_new_entities",
    "02_fronted_patient_new_entities",
    "03_passive_with_instrumental_agent",
    "04_recipient_subject_with_ot_source",
    "05_negated_transfer_not_occurred",
    "06_future_unrealized",
    "07_adjective_adverb_word_order",
    "08_two_independent_finite_predicates",
)
ROLES = ("NONE", "AGENT", "PATIENT", "RECIPIENT", "SOURCE")
EXPECTED_ROLES = {"AGENT": 32, "PATIENT": 36, "RECIPIENT": 32, "SOURCE": 4}
EXPECTED_EDGES = 104


@dataclass(frozen=True)
class DirectedGold:
    predicate_word: int   # 1-based RAW word index, including punctuation in count
    argument_word: int    # 1-based RAW word index, including punctuation in count
    role: str


@dataclass(frozen=True)
class EVALRow:
    text: str
    category: str
    event_modality: str
    predicate_word_indices: tuple[int, ...]
    directed_gold: tuple[DirectedGold, ...]
    split: str = SPLIT

    def text_sha256(self):
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def _mk(category, truth, text, predicates, links):
    return EVALRow(
        text, category, truth, tuple(predicates),
        tuple(DirectedGold(pred, arg, role) for pred, arg, role in links))


def build_new_eval32() -> tuple[EVALRow, ...]:
    """32 *manually specified* NEW texts and directed roles, no template grid.

    Reused names INSIDE EVAL highlight contrastive passive vs recipient-subject;
    none of these are training instructions. Return order/indices are sealed.
    """
    out = []
    def add(kind, truth, text, verbs, links):
        out.append(_mk(kind, truth, text, verbs, links))

    active,fronted,passive,receive,negated,future,modified,two = CATEGORIES
    # ACTIVE: 4 independently authored sentences, three semantic edges each.
    add(active,"ASSERTED","Зинаида вручила Петру диплом.",(2,),
        ((2,1,"AGENT"),(2,3,"RECIPIENT"),(2,4,"PATIENT")))
    add(active,"ASSERTED","Гавриил передал Оксане пропуск.",(2,),
        ((2,1,"AGENT"),(2,3,"RECIPIENT"),(2,4,"PATIENT")))
    add(active,"ASSERTED","Регина выдала Аркадию квитанцию.",(2,),
        ((2,1,"AGENT"),(2,3,"RECIPIENT"),(2,4,"PATIENT")))
    add(active,"ASSERTED","Ильдар направил Веронике извещение.",(2,),
        ((2,1,"AGENT"),(2,3,"RECIPIENT"),(2,4,"PATIENT")))

    # Fronted PATIENT, person AGENT follows the verb.
    add(fronted,"ASSERTED","Грамоту Марине вручил Юлиан.",(3,),
        ((3,4,"AGENT"),(3,1,"PATIENT"),(3,2,"RECIPIENT")))
    add(fronted,"ASSERTED","Бандероль Зое отправила Агата.",(3,),
        ((3,4,"AGENT"),(3,1,"PATIENT"),(3,2,"RECIPIENT")))
    add(fronted,"ASSERTED","Справку Тимуру передала Эльвира.",(3,),
        ((3,4,"AGENT"),(3,1,"PATIENT"),(3,2,"RECIPIENT")))
    add(fronted,"ASSERTED","Билет Раисе привёз Савва.",(3,),
        ((3,4,"AGENT"),(3,1,"PATIENT"),(3,2,"RECIPIENT")))

    # Passive morphology: predicate is short passive participle (word 3).
    add(passive,"ASSERTED","Грамота была вручена Юлианом Марине.",(3,),
        ((3,4,"AGENT"),(3,1,"PATIENT"),(3,5,"RECIPIENT")))
    add(passive,"ASSERTED","Квитанция была выписана Региной Аркадию.",(3,),
        ((3,4,"AGENT"),(3,1,"PATIENT"),(3,5,"RECIPIENT")))
    add(passive,"ASSERTED","Бандероль была отправлена Агатой Зое.",(3,),
        ((3,4,"AGENT"),(3,1,"PATIENT"),(3,5,"RECIPIENT")))
    add(passive,"ASSERTED","Извещение было передано Ильдаром Веронике.",(3,),
        ((3,4,"AGENT"),(3,1,"PATIENT"),(3,5,"RECIPIENT")))

    # 'получил(а)' recipient is subject, 'от' phrase = SOURCE not AGENT.
    add(receive,"ASSERTED","Марина получила грамоту от Юлиана.",(2,),
        ((2,1,"RECIPIENT"),(2,3,"PATIENT"),(2,5,"SOURCE")))
    add(receive,"ASSERTED","Аркадий получил квитанцию от Регины.",(2,),
        ((2,1,"RECIPIENT"),(2,3,"PATIENT"),(2,5,"SOURCE")))
    add(receive,"ASSERTED","Зоя получила бандероль от Агаты.",(2,),
        ((2,1,"RECIPIENT"),(2,3,"PATIENT"),(2,5,"SOURCE")))
    add(receive,"ASSERTED","Вероника получила извещение от Ильдара.",(2,),
        ((2,1,"RECIPIENT"),(2,3,"PATIENT"),(2,5,"SOURCE")))

    # Negated: argument structure is still gold, but event NEVER occurred.
    add(negated,"NEGATED","Доминика не показала врачу рентгенограмму.",(3,),
        ((3,1,"AGENT"),(3,4,"RECIPIENT"),(3,5,"PATIENT")))
    add(negated,"NEGATED","Костя не отправил редактору заметку.",(3,),
        ((3,1,"AGENT"),(3,4,"RECIPIENT"),(3,5,"PATIENT")))
    add(negated,"NEGATED","Евдокия не выдала студенту справку.",(3,),
        ((3,1,"AGENT"),(3,4,"RECIPIENT"),(3,5,"PATIENT")))
    add(negated,"NEGATED","Профессор не передал лаборанту образец.",(3,),
        ((3,1,"AGENT"),(3,4,"RECIPIENT"),(3,5,"PATIENT")))

    # Future: argument structure without asserting transfer already occurred.
    add(future,"FUTURE_UNREALIZED","Вечером Марселла привезёт директору акт.",(3,),
        ((3,2,"AGENT"),(3,4,"RECIPIENT"),(3,5,"PATIENT")))
    add(future,"FUTURE_UNREALIZED","Завтра Аким предоставит комиссии отчёт.",(3,),
        ((3,2,"AGENT"),(3,4,"RECIPIENT"),(3,5,"PATIENT")))
    add(future,"FUTURE_UNREALIZED","Утром Мила сообщит диспетчеру адрес.",(3,),
        ((3,2,"AGENT"),(3,4,"RECIPIENT"),(3,5,"PATIENT")))
    add(future,"FUTURE_UNREALIZED","Позже Герман доставит заказчику макет.",(3,),
        ((3,2,"AGENT"),(3,4,"RECIPIENT"),(3,5,"PATIENT")))

    # Noise modifiers may be wordpair candidates, but MUST NOT be oracle-filtered.
    add(modified,"ASSERTED","Тяжёлый контейнер Снежана передала инженеру.",(4,),
        ((4,3,"AGENT"),(4,2,"PATIENT"),(4,5,"RECIPIENT")))
    add(modified,"ASSERTED","Почтальон аккуратно вручил соседке уведомление.",(3,),
        ((3,1,"AGENT"),(3,4,"RECIPIENT"),(3,5,"PATIENT")))
    add(modified,"ASSERTED","Новый сертификат Валентин отправил аудитору.",(4,),
        ((4,3,"AGENT"),(4,2,"PATIENT"),(4,5,"RECIPIENT")))
    add(modified,"ASSERTED","Курьер быстро передал секретарю конверт.",(3,),
        ((3,1,"AGENT"),(3,4,"RECIPIENT"),(3,5,"PATIENT")))

    # Two separately annotated predicate heads: first has two arguments,
    # second has three. 'а' and comma are distractors, not model hints.
    add(two,"ASSERTED",
        "Технолог измерил деталь, а мастер передал контролёру протокол.",
        (2,7), ((2,1,"AGENT"),(2,3,"PATIENT"),
                (7,6,"AGENT"),(7,8,"RECIPIENT"),(7,9,"PATIENT")))
    add(two,"ASSERTED",
        "Бухгалтер сверил суммы, а кассир выдал клиенту чек.",
        (2,7), ((2,1,"AGENT"),(2,3,"PATIENT"),
                (7,6,"AGENT"),(7,8,"RECIPIENT"),(7,9,"PATIENT")))
    add(two,"ASSERTED",
        "Инженер подписал журнал, а техник показал стажёру схему.",
        (2,7), ((2,1,"AGENT"),(2,3,"PATIENT"),
                (7,6,"AGENT"),(7,8,"RECIPIENT"),(7,9,"PATIENT")))
    add(two,"ASSERTED",
        "Врач проверил снимок, а медсестра передала пациенту заключение.",
        (2,7), ((2,1,"AGENT"),(2,3,"PATIENT"),
                (7,6,"AGENT"),(7,8,"RECIPIENT"),(7,9,"PATIENT")))

    rows=tuple(out)
    validate_new_eval32(rows)
    return rows


def validate_new_eval32(rows: tuple[EVALRow, ...]) -> None:
    """Deterministic source-only constraints. No candidate inference allowed."""
    def check(ok,why):
        if not bool(ok):
            raise ValueError("S18c6 NEW32 one-time source prereg: "+why)
    check(type(rows) is tuple and len(rows)==32
          and all(type(r) is EVALRow and r.split==SPLIT for r in rows)
          and len({r.text_sha256() for r in rows})==32,
          "32 unique previously-unseen text-only rows required")
    check(S18_CANONICAL_96_FULL_UD_AND_ROLES_SHA256==S18_TRAIN96_SHA
          and PRE_FROZEN_CANDIDATE_TRAIN_CI==43916875445
          and PRE_FROZEN_CONTROL_TRAIN_CI==43913448392,
          "prior original model and TRAIN-only frozen provenance drift")
    prior_train={r.text_sha256() for r in build_s18_new_train96()}
    prior_train|={r.sha256() for r in build_s9_train72()}
    prior_train|={r.sha256() for r in build_s4_train()}
    check(not (prior_train & {r.text_sha256() for r in rows}),
          "a source-only new EVAL row duplicates old actual TRAIN text")
    check(all(sum(r.category==cat for r in rows)==4 for cat in CATEGORIES)
          and [r.category for r in rows]==[
              cat for cat in CATEGORIES for _ in range(4)],
          "precommitted 8 construction groups of 4 each required")
    seen_roles={}
    total=0
    for idx,row in enumerate(rows):
        tokens=raw_surface_grammar(row.text)
        forms=[z.form for z in tokens]
        check(bool(row.text) and row.text.endswith(".") and forms[-1]=="."
              and len(tokens)>=5 and len(tokens)<=12
              and all(0<=t.start<t.end<=len(row.text) for t in tokens),
              "new32 Unicode source and punctuation invalid")
        check(row.event_modality in ("ASSERTED","NEGATED","FUTURE_UNREALIZED")
              and (row.event_modality=="NEGATED")==
                  (row.category==CATEGORIES[4])
              and (row.event_modality=="FUTURE_UNREALIZED")==
                  (row.category==CATEGORIES[5]),
              "negated/future grammatical argument not event occurrence")
        if row.category==CATEGORIES[4]:
            check("не" in [t.casefold() for t in forms],
                  "negation event unmarked in source text")
        if row.category==CATEGORIES[3]:
            check("от" in [t.casefold() for t in forms],
                  "SOURCE preposition missing from receive group")
        need_count=2 if row.category==CATEGORIES[7] else 1
        check(len(row.predicate_word_indices)==need_count
              and len(set(row.predicate_word_indices))==need_count
              and all(type(k) is int and 1<=k<len(tokens)
                      for k in row.predicate_word_indices),
              "manually pre-registered predicate positions invalid")
        edges=set()
        counts={}
        for edge in row.directed_gold:
            check(type(edge) is DirectedGold
                  and type(edge.predicate_word) is int
                  and edge.predicate_word in row.predicate_word_indices
                  and type(edge.argument_word) is int
                  and 1<=edge.argument_word<len(tokens)
                  and edge.argument_word!=edge.predicate_word
                  and edge.role in ROLES[1:],
                  "gold edge predicate / argument / role invalid")
            t=(edge.predicate_word,edge.argument_word)
            check(t not in edges,"duplicate gold edge for same ordered wordpair")
            edges.add(t)
            counts[edge.role]=counts.get(edge.role,0)+1
            seen_roles[edge.role]=seen_roles.get(edge.role,0)+1
            total+=1
        if row.category==CATEGORIES[3]:
            check(counts=={"RECIPIENT":1,"PATIENT":1,"SOURCE":1},
                  "recipient-subject must NOT be labeled AGENT")
        elif row.category==CATEGORIES[7]:
            check(counts=={"AGENT":2,"PATIENT":2,"RECIPIENT":1},
                  "two predicates require two AGENT and PATIENT roles")
        else:
            check(counts=={"AGENT":1,"PATIENT":1,"RECIPIENT":1},
                  "one active/passive/negated/future head needs 3 directed roles")
        check(len(row.directed_gold)==(5 if row.category==CATEGORIES[7] else 3),
              "every new EVAL row directed link count fixed")
    check(total==EXPECTED_EDGES==104 and seen_roles==EXPECTED_ROLES,
          "complete 104 exact directed roles or role distribution mismatch")


def canonical_eval32_bytes(rows: tuple[EVALRow, ...]) -> bytes:
    validate_new_eval32(rows)
    corpus=[]
    for row in rows:
        tokens=raw_surface_grammar(row.text)
        corpus.append({
            "split":row.split,
            "category":row.category,
            "event_modality":row.event_modality,
            "text":row.text,
            "text_sha256":row.text_sha256(),
            "token_offsets":[[t.form,t.start,t.end] for t in tokens],
            "predicate_word_indices":list(row.predicate_word_indices),
            "directed_gold":[[e.predicate_word,e.argument_word,e.role]
                             for e in row.directed_gold],
        })
    return json.dumps(corpus,ensure_ascii=False,
                      sort_keys=True,separators=(",",":")).encode("utf-8")


def seal_new_eval32_source():
    rows=build_new_eval32()
    canonical=canonical_eval32_bytes(rows)
    digest=hashlib.sha256(canonical).hexdigest()
    if EVAL32_COMPLETE_CANONICAL_SHA256 is not None and (
            digest!=EVAL32_COMPLETE_CANONICAL_SHA256):
        raise ValueError("S18c6 immutable new one-time EVAL32 source SHA changed")
    return {
        "stage":"NRA3_S18C6_NEW32_ONE_TIME_EVAL_SOURCE_ONLY_PREREG_NOT_SCORED",
        "new_eval_source_complete_sha256":digest,
        "first_prereg_source_only_CI":NEW_EVAL_SOURCE_CI,
        "sha_locked_in_source":EVAL32_COMPLETE_CANONICAL_SHA256 is not None,
        "source_prior_candidate_train_run":PRE_FROZEN_CANDIDATE_TRAIN_CI,
        "source_prior_candidate_2597_FP16_sha":PRE_FROZEN_CANDIDATE_2597_FP16_SHA,
        "source_prior_control_train_run":PRE_FROZEN_CONTROL_TRAIN_CI,
        "source_prior_control_2309_FP16_sha":PRE_FROZEN_CONTROL_2309_FP16_SHA,
        "prior_S18c5_12fold_TRAIN_CV_run":PREVIOUS_12FOLD_TRAIN_ONLY_CI,
        "precommitted_exact_new_eval_sentences":len(rows),
        "new_eval_sentences_each_8_groups":[4]*8,
        "new_eval_gold_predicate_to_argument_links":EXPECTED_EDGES,
        "gold_roles_counts":EXPECTED_ROLES,
        "new_EVAL_categories":list(CATEGORIES),
        "manual_one_time_heldout_not_representative_natural_corpus":True,
        "previous_consumed_EVAL16_EVAL24_EVAL128_and_S3CI_not_loaded":True,
        "old_training_texts_disjoint":True,
        "gold_never_passed_to_candidate_or_original43M":True,
        "candidate_scored_on_eval32":False,
        "original43M_GT_inference_executed_on_eval32":False,
        "new_independent_eval_used_for_training_or_model_selection":False,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
        "LIVE_Jessica_chat_Render_Supabase_APK_modified":False,
    }
