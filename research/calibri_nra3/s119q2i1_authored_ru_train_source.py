"""S119q2i1: original-author Russian full-answer native-token supervision.

Completely newly authored, not scraped/copyrighted third-party text.
For each of six task domains, EXACT 6 TRAIN, 2 DEV, 2 SEALED_EVAL
pairs. The SEALED_EVAL block is source-provenance only: never opened by
physical learning or diagnostic job, never use it to tune the model.
This is a pre-registered small synthetic corpus, NOT a rights-verified
large natural language distribution or an independent external blind test.

Every answer must be generated from native multi-BPE targets plus the
actual tokenizer EOS; no heuristic "Russian" vocab masks or hard-coded
word bonuses. No old S118 GOLD, spent EVAL128, protected J7 answer labels.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import re

from research.calibri_nra3.s119q2a_train_source_manifest import build_rows
from research.calibri_nra3.s119q2c1_author_counterfactual_pairs import (
    rows as old_q2c1_rows,
)
from research.calibri_nra3.s119q2g_fresh_own_token_source import (
    rows as spent_q2g_rows,
)

STAGE="CALIBRI_S119Q2I1_AUTHOR_ORIGINAL_RU_FULL_NATIVE_TRAIN_AND_DEV"
KINDS=("roles_word_order","negation","transitive_comparison",
       "event_time","arithmetic","paraphrase")
TRAIN_PER_KIND=6
DEV_PER_KIND=2
SEALED_PER_KIND=2
SOURCE_LICENSE="original_newly_authored_examples_no_third_party_corpus"
# Invented language examples; answers follow the written facts, no RLHF API.
CASES={
"roles_word_order":(
 ("Часы Кириллу вернула Марина. Кто вернул часы? Ответьте одним именем.","Марина."),
 ("Документ Алёне показал Тимофей. Кто показал документ? Одно имя.","Тимофей."),
 ("Тетрадь Николаю отдала Вера. Кто отдал тетрадь? Одно имя.","Вера."),
 ("Зонт Полине принёс Максим. Кто принёс зонт? Одно имя.","Максим."),
 ("Газету Роману передала Юлия. Кто передал газету? Одно имя.","Юлия."),
 ("Букет Инге подарил Антон. Кто подарил букет? Одно имя.","Антон."),
 ("Ключи Владиславу вручил Артём. Кто вручил ключи? Одно имя.","Артём."),
 ("Подарок Ксении отправила Нелли. Кто отправил подарок? Одно имя.","Нелли."),
 ("Фонарь Василию передала Зоя. Кто передал фонарь? Одно имя.","Зоя."),
 ("Отчёт Тамаре представил Денис. Кто представил отчёт? Одно имя.","Денис."),
),
"negation":(
 ("Пётр не включил свет. Включил ли Пётр свет? Ответьте да или нет.","Нет."),
 ("Зина закрыла шкатулку. Закрыла ли Зина шкатулку? Ответьте да или нет.","Да."),
 ("Марта не написала записку. Написала ли Марта записку? Да или нет.","Нет."),
 ("Лев открыл шкаф. Открыл ли Лев шкаф? Да или нет.","Да."),
 ("Юрий не поднял флаг. Поднял ли Юрий флаг? Да или нет.","Нет."),
 ("Наталья поставила вазу. Поставила ли Наталья вазу? Да или нет.","Да."),
 ("Дина не прочитала письмо. Прочитала ли Дина письмо? Да или нет.","Нет."),
 ("Валерий выключил радио. Выключил ли Валерий радио? Да или нет.","Да."),
 ("Ася не почистила ботинки. Почистила ли Ася ботинки? Да или нет.","Нет."),
 ("Ильдар открыл футляр. Открыл ли Ильдар футляр? Да или нет.","Да."),
),
"transitive_comparison":(
 ("Озеро Северное глубже озера Тихого. Какое озеро глубже? Кратко.","Северное."),
 ("Сосна выше берёзы. Какое дерево ниже? Одним словом.","Берёза."),
 ("Сумка Ирины тяжелее сумки Оксаны. Чья сумка легче? Одно имя.","Оксаны."),
 ("Дорога до леса длиннее дороги до поля. Какая дорога короче? Кратко.","До поля."),
 ("Корабль Заря быстрее корабля Волна. Какой корабль быстрее? Одно название.","Заря."),
 ("Кувшин шире стакана. Что уже? Одно слово.","Стакан."),
 ("Мост Южный уже моста Восточного. Какой мост шире? Одно название.","Восточный."),
 ("Альбом толще блокнота. Что тоньше? Одно слово.","Блокнот."),
 ("Зал А светлее зала Б. Какой зал темнее? Одно обозначение.","Б."),
 ("Верёвка красная короче верёвки синей. Какая верёвка длиннее? Одно слово.","Синяя."),
),
"event_time":(
 ("После того как погас экран, включился вентилятор. Что случилось раньше? Кратко.","Погас экран."),
 ("Сначала открыли ворота, затем завели двигатель. Что сделали позже? Кратко.","Завели двигатель."),
 ("Прежде чем закипел чайник, прозвенел будильник. Что было раньше? Кратко.","Прозвенел будильник."),
 ("Сначала выгрузили ящики, потом закрыли багажник. Что сделали позже? Кратко.","Закрыли багажник."),
 ("После окончания лекции начался дождь. Что случилось раньше? Кратко.","Окончилась лекция."),
 ("Перед отправкой письма подписали конверт. Что было позже? Кратко.","Отправили письмо."),
 ("Сперва вымыли стол, затем расставили тарелки. Что сделали позже? Кратко.","Расставили тарелки."),
 ("Когда зажгли фонарь, велосипед уже остановился. Что было раньше? Кратко.","Остановился велосипед."),
 ("После звонка вышли из кабинета. Что было раньше? Кратко.","Звонок."),
 ("Сначала растопили печь, после этого открыли форточку. Что сделали позже? Кратко.","Открыли форточку."),
),
"arithmetic":(
 ("У Миши было 14 орехов, пять он съел. Сколько орехов осталось? Число.","9."),
 ("К девяти карандашам добавили восемь. Сколько карандашей стало? Число.","17."),
 ("На полке стояли 23 книги, забрали семь. Сколько книг осталось? Число.","16."),
 ("В ящике было четыре яблока, добавили одиннадцать. Сколько стало? Число.","15."),
 ("У продавца осталось 30 свечей, девять продали. Сколько осталось? Число.","21."),
 ("К шести чашкам добавили тринадцать. Сколько чашек теперь? Число.","19."),
 ("В кувшине 25 ягод, десять пересыпали. Сколько ягод осталось? Число.","15."),
 ("В саду 12 кустов, посадили ещё девять. Сколько всего кустов? Число.","21."),
 ("Из 18 листов использовали четыре. Сколько листов осталось? Число.","14."),
 ("В банке восемь конфет, положили ещё шесть. Сколько стало? Число.","14."),
),
"paraphrase":(
 ("Подберите один синоним к слову «смелый». Только слово.","Храбрый."),
 ("Как одним словом назвать того, кто «размышляет»? Синоним глагола.","Думает."),
 ("Замените глагол «мчался» одним синонимом.","Несся."),
 ("Назовите синоним прилагательного «огромный». Одно слово.","Гигантский."),
 ("Подберите один синоним к слову «тихий».","Негромкий."),
 ("Замени глагол «улыбнулся» близким по смыслу словом.","Усмехнулся."),
 ("Назови синоним слова «быстрый». Только слово.","Скорый."),
 ("Назовите синоним к слову «печальный». Одно слово.","Грустный."),
 ("Замените слово «радость» синонимом. Одно слово.","Веселье."),
 ("Подберите синоним для слова «красивый». Одно слово.","Прекрасный."),
),
}


def canonical_bytes(obj):
    return (json.dumps(obj,ensure_ascii=False,sort_keys=True,
                       separators=(",",":"),allow_nan=False)+"\n").encode()


def hash_rows(rows):
    return hashlib.sha256(b"".join(canonical_bytes(x) for x in rows)).hexdigest()


def rows(split):
    """TRAIN or DEV only; sealed EVAL never returned to training API."""
    if split not in ("TRAIN","DEV"):
        raise ValueError("SEALED_EVAL intentionally inaccessible through training API")
    start=0 if split=="TRAIN" else TRAIN_PER_KIND
    stop=TRAIN_PER_KIND if split=="TRAIN" else TRAIN_PER_KIND+DEV_PER_KIND
    return [
        {"id":f"S119Q2I1_{kind}_{i:02d}",
         "kind":kind,"split":split,"prompt":CASES[kind][i][0],
         "completion":CASES[kind][i][1],
         "origin":SOURCE_LICENSE}
        for kind in KINDS for i in range(start,stop)
    ]


def protected_eval_digest():
    """No EVAL labels in any downstream persisted training/DEV report."""
    return hashlib.sha256(b"".join(
        canonical_bytes([kind,i,*CASES[kind][i]])
        for kind in KINDS
        for i in range(TRAIN_PER_KIND+DEV_PER_KIND,
                       TRAIN_PER_KIND+DEV_PER_KIND+SEALED_PER_KIND)
    )).hexdigest()


def provenance():
    train=rows("TRAIN")
    dev=rows("DEV")
    if not (len(train)==36 and len(dev)==12 and
            Counter(x["kind"] for x in train)=={k:6 for k in KINDS} and
            Counter(x["kind"] for x in dev)=={k:2 for k in KINDS}):
        raise ValueError("S119q2i1 failclosed wrong authored source shape")
    if any(len(CASES[kind])!=10 for kind in KINDS):
        raise ValueError("S119q2i1 needs 6 train, 2 DEV, 2 sealed per domain")
    old={x["prompt"] for x in build_rows()}|{
        x["prompt"] for x in old_q2c1_rows()}|{
        x["question"] for x in spent_q2g_rows()}
    new={x["prompt"] for x in train+dev}
    if len(new)!=48 or new&old:
        raise ValueError("S119q2i1 TRAIN/DEV duplicates or spent old sources")
    return {
        "stage":STAGE,
        "origin":SOURCE_LICENSE,
        "TRAIN_rows":len(train),"DEV_rows":len(dev),
        "TRAIN_sha256":hash_rows(train),
        "DEV_sha256":hash_rows(dev),
        "SEALED_EVAL_count_NEVER_OPENED":12,
        "SEALED_EVAL_provenance_SHA256_NOT_LABELS":protected_eval_digest(),
        "old_sources_and_spent_q2g_disjoint":True,
        "no_external_independent_blind_QA_claim":True,
        "original_author_training_examples_only":True,
    }
