"""S119q2c1: fresh paired author-written Russian relation counterfactuals.

Train-only original hand-authored SMOKE dataset for CALIBRI, not a general QA
benchmark. Predeclared 2 contrast-pairs per kind used by optimizer; the
third lexical contrast-pair is a SOURCE-only diagnostic probe, NEVER gradient.
No sealed GOLD, spent S119n author prompts, older EVAL128, J7 answers, Jessica
chats or external copyrighted dataset. No final independent QA.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter

STAGE = "CALIBRI_S119Q2C1_ORIGINAL_AUTHOR_PAIRED_RU_RELATIONS_TRAIN_ONLY"
VERSION = "S119Q2C1_AUTHOR_PAIRS_20261122"
KINDS = ("roles_word_order","negation","transitive_comparison",
         "event_time","arithmetic","paraphrase")
USES = ("TRAIN","SOURCE_PROBE")
PAIR_BANK = {
    "roles_word_order": (
        (("Книгу Лене передал Артём. Кто передал книгу?", "Артём."),
         ("Книгу Артёму передала Лена. Кто передал книгу?", "Лена.")),
        (("Папку Оле передал Максим. Кто передал папку?", "Максим."),
         ("Папку Максиму передала Оля. Кто передал папку?", "Оля.")),
        (("Схему Вере показал Руслан. Кто показал схему?", "Руслан."),
         ("Схему Руслану показала Вера. Кто показал схему?", "Вера.")),
    ),
    "negation": (
        (("Никита отправил посылку Кире. Состоялась ли отправка?", "Да."),
         ("Никита не отправил посылку Кире. Состоялась ли отправка?", "Нет.")),
        (("Вадим передал ключ Миле. Состоялась ли передача?", "Да."),
         ("Вадим не передал ключ Миле. Состоялась ли передача?", "Нет.")),
        (("Егор открыл дверь. Состоялось ли открытие?", "Да."),
         ("Егор не открыл дверь. Состоялось ли открытие?", "Нет.")),
    ),
    "transitive_comparison": (
        (("Лев выше Павла. Кто выше?", "Лев."),
         ("Павел выше Льва. Кто выше?", "Павел.")),
        (("Борис старше Степана. Кто старше?", "Борис."),
         ("Степан старше Бориса. Кто старше?", "Степан.")),
        (("Глеб быстрее Юрия. Кто быстрее?", "Глеб."),
         ("Юрий быстрее Глеба. Кто быстрее?", "Юрий.")),
    ),
    "event_time": (
        (("Сначала погас свет, затем открылась дверь. Что произошло раньше?",
          "Погас свет."),
         ("Сначала открылась дверь, затем погас свет. Что произошло раньше?",
          "Открылась дверь.")),
        (("Сначала зазвонил телефон, затем начался дождь. Что произошло раньше?",
          "Зазвонил телефон."),
         ("Сначала начался дождь, затем зазвонил телефон. Что произошло раньше?",
          "Начался дождь.")),
        (("Сначала вышел поезд, затем закрылся магазин. Что произошло раньше?",
          "Вышел поезд."),
         ("Сначала закрылся магазин, затем вышел поезд. Что произошло раньше?",
          "Закрылся магазин.")),
    ),
    "arithmetic": (
        (("В коробке было 9 кубиков. Добавили 2 кубика. Сколько стало?", "11."),
         ("В коробке было 9 кубиков. Убрали 2 кубика. Сколько осталось?",
          "7.")),
        (("Было 12 билетов. Добавили 8 билетов. Сколько стало?", "20."),
         ("Было 12 билетов. Убрали 8 билетов. Сколько осталось?", "4.")),
        (("Было 23 карточки. Добавили 6 карточек. Сколько стало?", "29."),
         ("Было 23 карточки. Убрали 6 карточек. Сколько осталось?", "17.")),
    ),
    "paraphrase": (
        (("Перефразируй глагол одним словом: «Павел начал работу».",
          "Приступил."),
         ("Перефразируй глагол одним словом: «Павел закончил работу».",
          "Завершил.")),
        (("Перефразируй глагол одним словом: «Вадим приобрёл билет».",
          "Купил."),
         ("Перефразируй глагол одним словом: «Вадим потерял билет».",
          "Утратил.")),
        (("Перефразируй глагол одним словом: «Лев увидел ошибку».",
          "Заметил."),
         ("Перефразируй глагол одним словом: «Лев исправил ошибку».",
          "Устранил.")),
    ),
}
EXPECTED_SOURCE_SHA256 = "7052b503e4c734ccc05d9f97fd572913449975f755b6cd18815becc501c7fc60"  # pinned physical Actions #45012677814


def demand(ok: bool, reason: str):
    if not ok:
        raise ValueError("S119q2c1 FAIL CLOSED: " + reason)


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False)+"\n").encode("utf8")


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def rows():
    output=[]
    demand(tuple(PAIR_BANK)==KINDS and len(KINDS)==6,
           "six predeclared original relational domains only")
    for kind in KINDS:
        group=PAIR_BANK[kind]
        demand(len(group)==3, "two TRAIN pairs plus one SOURCE_PROBE pair")
        for pair_index,pair in enumerate(group):
            demand(len(pair)==2,
                   "each semantic contrast must have exactly two sides")
            for side,(prompt,answer) in enumerate(pair):
                output.append({
                    "id":f"{VERSION}_{kind}_{pair_index}_{side}",
                    "kind":kind,
                    "use":"TRAIN" if pair_index<2 else "SOURCE_PROBE",
                    "pair_index":pair_index,
                    "side":side,
                    "prompt":prompt,
                    "completion":answer,
                    "rule_kind":"authored_semantic_counterfactual_not_independent_gold",
                    "rights_scope":"self_authored_research_only",
                    "no_external_data_or_s118_gold":True,
                })
    return output


def validate(records):
    demand(type(records) is list and len(records)==36,
           "exactly 24 TRAIN + 12 untouched SOURCE_PROBE")
    demand(records==rows(), "no changed original source prompts/labels")
    ids=set();prompts=set();by_use=Counter()
    from research.calibri_nra3.s119q2a_train_source_manifest import (
        SPENT_S119N_PROMPTS,build_rows,EXPECTED_MANIFEST_SHA256,build_manifest
    )
    demand(build_manifest(build_rows())["synthetic_rows_SHA256"]==
           EXPECTED_MANIFEST_SHA256,
           "original q2a source digest unchanged")
    spent=set(SPENT_S119N_PROMPTS)|{r["prompt"] for r in build_rows()}
    for r in records:
        demand(r["id"] not in ids and r["prompt"] not in prompts and
               r["prompt"] not in spent, "duplicate or previously exposed source")
        demand(r["completion"].endswith(".") and
               r["prompt"]==r["prompt"].strip() and
               r["completion"]==r["completion"].strip(),
               "canonical literal completion and prompt")
        ids.add(r["id"]);prompts.add(r["prompt"]);by_use[r["use"]]+=1
    demand(dict(by_use)=={"TRAIN":24,"SOURCE_PROBE":12},
           "fixed source-only TRAIN and SOURCE_PROBE partition")
    for kind in KINDS:
        category=[r for r in records if r["kind"]==kind]
        demand(len(category)==6 and
               len([r for r in category if r["use"]=="TRAIN"])==4 and
               len([r for r in category if r["use"]=="SOURCE_PROBE"])==2,
               "exact balanced paired source representation")
        for pair_index in range(3):
            pair=[r for r in category if r["pair_index"]==pair_index]
            demand(len(pair)==2 and {r["side"] for r in pair}=={0,1}
                   and pair[0]["completion"]!=pair[1]["completion"],
                   "counterfactual pair MUST flip correct output")
    raw=b"".join(canonical_bytes(r) for r in records)
    digest=sha(raw)
    if EXPECTED_SOURCE_SHA256 is not None:
        demand(digest==EXPECTED_SOURCE_SHA256,
               "physical preregistered authored source SHA changed")
    return {"stage":STAGE,"version":VERSION,
            "source_sha256":digest,
            "source_ROWS_n":len(records),
            "TRAIN_rows_n":24,"SOURCE_PROBE_rows_n":12,
            "authored_counterfactual_pairs_n":18,
            "TRAIN_pairs_n":12,"SOURCE_PROBE_pairs_n":6,
            "six_domains":list(KINDS),
            "not_independent_unseen_generalization":True,
            "no_original_S118_B_GOLD_EVAL128_J7":True,
            "no_Jessica_chats_or_external_corpus":True,
            "no_training_in_this_source_builder":True}


def split(records):
    evidence=validate(records)
    a=[r for r in records if r["use"]=="TRAIN"]
    b=[r for r in records if r["use"]=="SOURCE_PROBE"]
    demand(len(a)==24 and len(b)==12 and
           {r["prompt"] for r in a}.isdisjoint(
               r["prompt"] for r in b),
           "train/probe must remain strictly separate")
    return a,b,evidence
