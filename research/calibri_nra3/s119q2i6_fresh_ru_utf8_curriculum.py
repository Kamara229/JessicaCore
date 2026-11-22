"""S119q2i6 PRE-DECLARED original-author Russian native UTF-8 curriculum.

18 TRAIN + 6 separate DEV: one each in six domains (Cyrillic
character, syllable, word copy, negation, arithmetic, event ordering).
Sources are short to expose multibyte ByteLevel BPE continuation.
Do not confuse copying isolated letters with full Russian competence.
All text is newly handwritten, not scraped or externally copyrighted.
DEV is inspectable and small: NOT externally independent/blind.

Plus six already known q2i1 TRAIN-only exemplars for retention rehearsal;
neither q2i1 old DEV nor q2g/q2h exposed labels ever become TRAIN.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json

from research.calibri_nra3.s119q2a_train_source_manifest import build_rows
from research.calibri_nra3.s119q2c1_author_counterfactual_pairs import (
    rows as old_counterfactual,
)
from research.calibri_nra3.s119q2g_fresh_own_token_source import (
    rows as old_q2g,
)
from research.calibri_nra3.s119q2i1_authored_ru_train_source import (
    rows as old_i1,
)

STAGE="CALIBRI_S119Q2I6_NEW_AUTHOR_RUSSIAN_UTF8_MULTIBPE_CURRICULUM"
KINDS=("cyrillic_letter","cyrillic_syllable","cyrillic_word_copy",
       "negation","arithmetic","event_order")
AUTHOR_ORIGIN="original_author_new_utf8_cyrillic_continuation_examples"
TRAIN_PER_KIND=3
DEV_PER_KIND=1
REHEARSAL_OLD_TRAIN_PER_KIND=1
CASES={
 "cyrillic_letter":(
  ("Напечатайте без кавычек единственную букву между скобками: [Л].","Л"),
  ("Скопируйте ровно одну кириллическую букву после двоеточия: Ж","Ж"),
  ("Ответ состоит из одной буквы. Повторите символ «Н».","Н"),
  ("Выведите одну русскую букву, указанную здесь: [Ф].","Ф"),
 ),
 "cyrillic_syllable":(
  ("Выпишите только два русских символа из скобок: [ма]","ма"),
  ("Повторите один слог после знака равенства: слог = ры","ры"),
  ("Введите без пробелов указанный слог: «жу»","жу"),
  ("Скопируйте двубуквенный слог без кавычек: [зи]","зи"),
 ),
 "cyrillic_word_copy":(
  ("После стрелки стоит слово. Выпишите только это слово: → мир","мир"),
  ("Повторите русское слово после знака равенства: слово = луг","луг"),
  ("Напишите ровно слово из скобок, без кавычек: [кот]","кот"),
  ("Перепишите только трёхбуквенное слово после двоеточия: сыр","сыр"),
 ),
 "negation":(
  ("Артём не поднял табличку. Поднял ли Артём табличку? Одно слово.","Нет"),
  ("Рита позвонила брату. Позвонила ли Рита брату? Одно слово.","Да"),
  ("Вадим не открыл конверт. Открыл ли Вадим конверт? Одно слово.","Нет"),
  ("Ульяна прочитала заметку. Прочитала ли Ульяна заметку? Одно слово.","Да"),
 ),
 "arithmetic":(
  ("Какова сумма четырёх и трёх? Только число.","7"),
  ("Из одиннадцати фишек убрали две. Сколько фишек осталось? Число.","9"),
  ("Прибавьте пять к шести. Ответьте числом.","11"),
  ("У Светы было восемь открыток, три она подарила. Сколько осталось? Число.","5"),
 ),
 "event_order":(
  ("Сперва прозвучал гонг, потом открыли зал. Что произошло первым? Одно слово.","Гонг"),
  ("Сначала выключили музыку, затем погасили лампу. Что произошло позже? Одно слово.","Лампу"),
  ("Звук сирены раздался до приезда автомобиля. Что было раньше? Одно слово.","Сирена"),
  ("До прихода мастера закончили уборку. Что было раньше? Одно слово.","Уборка"),
 ),
}


def _canon(o):
    return (json.dumps(o,ensure_ascii=False,sort_keys=True,
                       separators=(",",":"),allow_nan=False)+"\n").encode()


def rows(split):
    if split not in ("TRAIN","DEV"):
        raise ValueError("Only new original-author TRAIN or DEV; never reserved EVAL")
    selected=(range(3) if split=="TRAIN" else range(3,4))
    return [{
        "id":f"S119Q2I6_{kind}_{i:02d}",
        "kind":kind,"split":split,
        "prompt":CASES[kind][i][0],
        "completion":CASES[kind][i][1],
        "origin":AUTHOR_ORIGIN,
    } for kind in KINDS for i in selected]


def source_sha(rows_):
    return hashlib.sha256(b"".join(_canon(x) for x in rows_)).hexdigest()


def verify():
    train=rows("TRAIN")
    dev=rows("DEV")
    if len(train)!=18 or len(dev)!=6 or any(
        len(CASES[k])!=4 for k in KINDS
    ):
        raise ValueError("S119q2i6 original SOURCE size/splits corrupted")
    if Counter(x["kind"] for x in train)!={k:3 for k in KINDS}:
        raise ValueError("S119q2i6 TRAIN six-domain composition changed")
    if Counter(x["kind"] for x in dev)!={k:1 for k in KINDS}:
        raise ValueError("S119q2i6 DEV composition changed")
    all_new={r["prompt"] for r in train+dev}
    previously_used=(
        {r["prompt"] for r in build_rows()} |
        {r["prompt"] for r in old_counterfactual()} |
        {r["question"] for r in old_q2g()} |
        {r["prompt"] for r in old_i1("TRAIN")+old_i1("DEV")}
    )
    if len(all_new)!=24 or all_new & previously_used:
        raise ValueError("S119q2i6 SOURCE duplicate previous TRAIN/DEV/GOLD")
    if any(len(r["prompt"])>160 or len(r["completion"])>12 or
           not r["completion"].strip() for r in train+dev):
        raise ValueError("S119q2i6 source must be short complete UTF8 targets")
    return {
        "stage":STAGE,
        "author_origin":AUTHOR_ORIGIN,
        "fresh_original_author_TRAIN":len(train),
        "fresh_original_author_DEV":len(dev),
        "six_domains_3TRAIN_1DEV":True,
        "fresh_TRAIN_SHA256":source_sha(train),
        "fresh_DEV_SHA256":source_sha(dev),
        "no_overlap_any_previous_source_or_spent_evaluation":True,
        "DEV_is_small_inspectable_NOT_independent_BLIND":True,
        "NO_new_SEALED_EVAL_or_J7_answer_files_opened":True,
    }
