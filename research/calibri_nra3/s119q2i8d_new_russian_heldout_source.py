"""S119q2i8d held-out Russian full-answer DEV for teacherKL-vs-CE.
Explicitly NEW 14 original authored prompts not seen in Q2I8a TRAIN/DEV,
q2i6, q2i1 or previous source experiments. Two per seven semantic areas.
The answers are source-owned ground truth, NEVER copied from source135M:
the upstream 135M has been proven to continue the prompt and is not
a semantic answer oracle. Internal inspectable DEV is not external blind.
"""
from __future__ import annotations
import hashlib
import json
from collections import Counter

from research.calibri_nra3.s119q2i8a_broader_russian_synthetic_source import (
    KINDS,rows as q2i8a_rows,
)
from research.calibri_nra3.s119q2i6_fresh_ru_utf8_curriculum import rows as i6_rows
from research.calibri_nra3.s119q2i1_authored_ru_train_source import rows as i1_rows
from research.calibri_nra3.s119q2g_fresh_own_token_source import rows as q2g_rows
from research.calibri_nra3.s119q2c1_author_counterfactual_pairs import rows as c1_rows
from research.calibri_nra3.s119q2a_train_source_manifest import build_rows as old_rows

STAGE="CALIBRI_S119Q2I8D_FRESH_ORIGINAL_AUTHORED_14_NEVER_TRAINED_INTERNAL_RUSSIAN_DEV"
ORIGIN="research_original_author_newly_created_for_q2i8d_heldout_no_external_text"
ITEMS={
 "unicode_word_copy":(
   ("Скопируйте лишь слово, стоящее внутри фигурных скобок: {Клён}","Клён"),
   ("Напишите ровно слово после длинного тире: — Блик","Блик"),
 ),
 "reversed_agent_roles":(
   ("Кристине телеграмму отправил Роман. Кто отправитель? Одно имя.","Роман"),
   ("Андрею ключи передала Нина. Кто передал ключи? Одно имя.","Нина"),
 ),
 "scope_negation":(
   ("Валентина не полила кактус. Полила ли она кактус? Одно слово.","Нет"),
   ("Родион дописал рассказ. Дописал ли Родион рассказ? Одно слово.","Да"),
 ),
 "numeric_arithmetic":(
   ("У Лары было 14 пуговиц, она нашла ещё 9. Сколько стало? Числом.","23"),
   ("В коробке были 31 карандаш, 18 достали. Сколько осталось? Числом.","13"),
 ),
 "event_chronology":(
   ("Парус подняли перед отходом лодки. Что сделали раньше? Одно слово.","Парус"),
   ("Звонок прозвучал после прихода поезда. Что произошло раньше? Одно слово.","Поезд"),
 ),
 "comparative_references":(
   ("Бочонок из глины тяжелее бочонка из дерева. Какой легче? Одно слово.","Деревянный"),
   ("Зелёная сумка меньше оранжевой. Какая больше? Одно слово.","Оранжевая"),
 ),
 "lexical_paraphrase":(
   ("Подберите синоним к слову «необычный». Одно слово.","Странный"),
   ("Как сказать одним словом «окончить»?","Завершить"),
 ),
}

def rows(split="DEV"):
    if split!="DEV":
        raise ValueError("Q2I8d exposes only a NEW internal DEV; never Q2I8d TRAIN or EVAL")
    return [{
        "id":f"S119Q2I8D_{k}_{i:02d}",
        "kind":k,"split":"DEV",
        "prompt":question,"completion":answer,"origin":ORIGIN,
    } for k in KINDS for i,(question,answer) in enumerate(ITEMS[k])]


def _digest(items):
    return hashlib.sha256(b"".join((
       json.dumps(r,ensure_ascii=False,sort_keys=True,
                  separators=(",",":"),allow_nan=False)+"\n"
       ).encode("utf8") for r in items)).hexdigest()


def verify():
    dev=rows()
    previously_used=(
        {x["prompt"] for x in q2i8a_rows("TRAIN")+q2i8a_rows("DEV")} |
        {x["prompt"] for x in i6_rows("TRAIN")+i6_rows("DEV")} |
        {x["prompt"] for x in i1_rows("TRAIN")+i1_rows("DEV")} |
        {x["question"] for x in q2g_rows()} |
        {x["prompt"] for x in c1_rows()} |
        {x["prompt"] for x in old_rows()}
    )
    if (len(dev)!=14 or len({x["prompt"] for x in dev})!=14
        or Counter(x["kind"] for x in dev)!={k:2 for k in KINDS}
        or any(len(x["completion"])>25 or not x["completion"] or
               len(x["prompt"])>160 for x in dev)
        or {x["prompt"] for x in dev}&previously_used):
        raise ValueError("14 brand-new original heldout DEV, zero previous overlap")
    return {
        "stage":STAGE,"origin":ORIGIN,
        "original_new_author_heldout_DEV_count":14,
        "source_new_DEV_sha256":_digest(dev),
        "same_7_prior_source_domains":True,
        "no_prompt_overlap_with_any_previous_training_or_scored_DEV":True,
        "synthetic_author_written_NOT_external_blind":True,
        "no_saved_unopened_J7_GOLD_EVAL128_access":True,
    }
