"""S119q2i8a: larger, independently authored SYNTHETIC Russian full-text source.

Source is 70 NEW TRAIN / 14 distinct internal DEV from 7 broad tasks.
Every example was explicitly created for this research stage; not scraped
from third-party text, nor a licensed natural Russian corpus. Internal DEV
is VISIBLE in source, NOT independent external blind evaluation. Every DEV
prompt is new and never optimized; never use DEV labels for model choice.

Responses include Cyrillic and native EOS, no hardcoded output tokens.
New corpus is 70 (vs 18 previous), but still too small to prove general
linguistic ability. Full 616 J7/EOS etc remain separately protected.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json

from research.calibri_nra3.s119q2a_train_source_manifest import build_rows
from research.calibri_nra3.s119q2c1_author_counterfactual_pairs import rows as old_cf
from research.calibri_nra3.s119q2g_fresh_own_token_source import rows as old_q2g
from research.calibri_nra3.s119q2i1_authored_ru_train_source import rows as old_i1
from research.calibri_nra3.s119q2i6_fresh_ru_utf8_curriculum import rows as old_i6

STAGE="CALIBRI_S119Q2I8A_NEW_ORIGINAL_SYNTHETIC_RUSSIAN_70TRAIN_14DEV"
ORIGIN="created_for_this_experiment_7_domains_synthetic_not_natural_corpus"
KINDS=(
    "unicode_word_copy",
    "reversed_agent_roles",
    "scope_negation",
    "numeric_arithmetic",
    "event_chronology",
    "comparative_references",
    "lexical_paraphrase",
)
TRAIN_PER_KIND=10
DEV_PER_KIND=2
CASES={
    "unicode_word_copy":(
        ("Перепишите только слово в квадратных скобках: [Роза]","Роза"),
        ("Напечатайте лишь слово после стрелки: → Мёд","Мёд"),
        ("Повторите точно слово без кавычек: «Бобр»","Бобр"),
        ("Выпишите слово из записи слово=Лиса, больше ничего.","Лиса"),
        ("Что написано в скобках? В ответе только слово: (Груша)","Груша"),
        ("Введите только указанное название: [Слон]","Слон"),
        ("Копируйте русское слово после двоеточия: Лист","Лист"),
        ("Напишите без пробелов и знаков только слово: Парус","Парус"),
        ("Продублируйте текст между скобками: (Ключ)","Ключ"),
        ("Повторите короткое слово после знака равно: Ветка","Ветка"),
        ("Выпишите точно слово из квадратных скобок: [Мышь]","Мышь"),
        ("Напечатайте только русское слово после стрелки: → Мороз","Мороз"),
    ),
    "reversed_agent_roles":(
        ("Рисунок Грише передала Лидия. Назовите отправителя.","Лидия"),
        ("Письмо Ване принёс Игнат. Кто принёс письмо?","Игнат"),
        ("Шарик Феде отдала Милана. Кто отдал шарик?","Милана"),
        ("Секрет Вере сообщил Платон. Кто сообщил секрет?","Платон"),
        ("Карту Борису показала Регина. Кто показывал карту?","Регина"),
        ("Ведро Любе передал Глеб. Кто передал ведро?","Глеб"),
        ("Пакет Егору принесла Лариса. Кто принёс пакет?","Лариса"),
        ("Мячик Диме подбросила Олеся. Кто подбросил мячик?","Олеся"),
        ("Книгу Алисе вручил Виталий. Кто вручил книгу?","Виталий"),
        ("Посылку Илье доставила Алина. Кто доставил посылку?","Алина"),
        ("Билет Гале передал Ростислав. Кто передал билет?","Ростислав"),
        ("Записку Соне принесла Жанна. Кто принёс записку?","Жанна"),
    ),
    "scope_negation":(
        ("Богдан не зажёг свечу. Зажёг ли он её? Одно слово.","Нет"),
        ("Софья прочла рецепт. Прочла ли она рецепт? Одно слово.","Да"),
        ("Филипп не открыл окно. Открыл ли он окно? Одно слово.","Нет"),
        ("Лада нашла билет. Нашла ли она билет? Одно слово.","Да"),
        ("Никита не опустил занавес. Опустил ли он его? Одно слово.","Нет"),
        ("Эмма включила фонарик. Включила ли она его? Одно слово.","Да"),
        ("Степан не закрасил квадрат. Закрасил ли он квадрат?","Нет"),
        ("Эля собрала конструктор. Собрала ли Эля конструктор?","Да"),
        ("Оскар не вытер стол. Вытер ли Оскар стол?","Нет"),
        ("Агата поставила чайник. Поставила ли Агата чайник?","Да"),
        ("Семён не завязал шнурок. Завязал ли Семён шнурок?","Нет"),
        ("Ева закрыла папку. Закрыла ли Ева папку?","Да"),
    ),
    "numeric_arithmetic":(
        ("В корзине 7 слив, добавили 4. Сколько слив теперь? Только число.","11"),
        ("На тарелке 16 ягод, 6 съели. Сколько осталось? Только число.","10"),
        ("У Сергея 5 марок, купил ещё 8. Сколько марок стало?","13"),
        ("В пачке 19 листов, 9 взяли. Сколько листов осталось?","10"),
        ("В коробке 3 кубика, добавили 6. Сколько всего?","9"),
        ("На полке 22 тетради, убрали 5. Сколько осталось?","17"),
        ("В саду 11 ромашек, распустились ещё 7. Сколько стало?","18"),
        ("У Юли 20 наклеек, подарила 12. Сколько осталось?","8"),
        ("Два мешка содержат по 4 мяча. Сколько мячей всего?","8"),
        ("Было 24 билета, купили 11. Сколько билетов осталось?","13"),
        ("У Вадима 8 свечей и ещё 9. Сколько свечей всего?","17"),
        ("Из 27 фишек потеряли 8. Сколько фишек осталось?","19"),
    ),
    "event_chronology":(
        ("Сначала замигал индикатор, затем запустился мотор. Что раньше? Одно слово.","Индикатор"),
        ("Раньше открылся шлюз, позже зашёл катер. Что было позже?","Катер"),
        ("Вначале прозвучала песня, потом выступил хор. Что было первым?","Песня"),
        ("Перед грозой усилился ветер. Что было раньше?","Ветер"),
        ("Сначала выключили свет, после зажгли свечу. Что позже?","Свечу"),
        ("Перед посадкой самолёта подали сигнал. Что раньше?","Сигнал"),
        ("Сначала распустилась сирень, позже выросла трава. Что позже?","Трава"),
        ("Вначале зазвонил телефон, потом открыли дверь. Что раньше?","Телефон"),
        ("Прежде чем зажёгся костёр, сложили дрова. Что раньше?","Дрова"),
        ("После старта ракеты включили сирену. Что случилось позже?","Сирена"),
        ("Сначала появилась радуга, затем потемнело. Что было раньше?","Радуга"),
        ("Перед обедом пришли гости. Что произошло раньше?","Гости"),
    ),
    "comparative_references":(
        ("Шар красный больше шара синего. Какой шар меньше? Одно слово.","Синий"),
        ("Ваза узкая ниже вазы широкой. Какая ваза выше? Одно слово.","Широкая"),
        ("Тропа к реке длиннее тропы к лесу. Куда путь короче?","Лес"),
        ("Дом номер два выше дома номер пять. Какой выше? Число.","2"),
        ("Рюкзак Лёши тяжелее рюкзака Гены. Чей легче? Одно имя.","Гены"),
        ("Плитка А холоднее плитки Б. Какая холоднее? Буква.","А"),
        ("Палка берёзовая длиннее палки дубовой. Какая короче?","Дубовая"),
        ("Коридор правый светлее левого. Какой темнее?","Левый"),
        ("Пирог с яблоком слаще пирога с грушей. Какой слаще?","Яблочный"),
        ("Поезд Зарница медленнее поезда Север. Какой быстрее?","Север"),
        ("Карандаш зелёный толще красного. Какой тоньше?","Красный"),
        ("Кольцо серебряное меньше кольца медного. Какое больше?","Медное"),
    ),
    "lexical_paraphrase":(
        ("Замените слово «крошечный» близким по значению. Одно слово.","Маленький"),
        ("Подберите синоним слова «холодный». Одно слово.","Студёный"),
        ("Как иначе одним словом сказать «разговаривать»?","Беседовать"),
        ("Подберите синоним к глаголу «наблюдать».","Смотреть"),
        ("Замените слово «начало» близким по смыслу существительным.","Старт"),
        ("Одним словом замените «беззвучный».","Тихий"),
        ("Как иначе сказать «старинный» одним словом?","Древний"),
        ("Укажите синоним глагола «сберечь».","Сохранить"),
        ("Замените прилагательное «ловкий» синонимом.","Проворный"),
        ("Как одним словом сказать «торопиться»?","Спешить"),
        ("Подберите близкое по смыслу слово к «умный».","Разумный"),
        ("Назовите синоним слова «яркий».","Светлый"),
    ),
}


def rows(split):
    if split not in ("TRAIN","DEV"):
        raise ValueError("Only registered original synthetic TRAIN/DEV; no EVAL labels")
    wanted=range(TRAIN_PER_KIND) if split=="TRAIN" else range(
        TRAIN_PER_KIND,TRAIN_PER_KIND+DEV_PER_KIND)
    return [{
        "id":f"S119Q2I8A_{kind}_{i:02d}",
        "kind":kind,
        "split":split,
        "prompt":CASES[kind][i][0],
        "completion":CASES[kind][i][1],
        "origin":ORIGIN,
    } for kind in KINDS for i in wanted]


def source_sha(rows_):
    encoded=b"".join((
        json.dumps(x,sort_keys=True,ensure_ascii=False,
                   separators=(",",":"),allow_nan=False)+"\n").encode("utf8")
        for x in rows_)
    return hashlib.sha256(encoded).hexdigest()


def verify():
    tr=rows("TRAIN");dv=rows("DEV")
    if (len(KINDS)!=7 or len(tr)!=70 or len(dv)!=14 or
            any(len(CASES[k])!=12 for k in KINDS) or
            Counter(x["kind"] for x in tr)!={k:10 for k in KINDS} or
            Counter(x["kind"] for x in dv)!={k:2 for k in KINDS}):
        raise ValueError("S119q2i8a source contract must be exactly 70/14 in 7 tasks")
    previous=(
        {x["prompt"] for x in build_rows()} |
        {x["prompt"] for x in old_cf()} |
        {x["question"] for x in old_q2g()} |
        {x["prompt"] for x in old_i1("TRAIN")+old_i1("DEV")} |
        {x["prompt"] for x in old_i6("TRAIN")+old_i6("DEV")}
    )
    current={x["prompt"] for x in tr+dv}
    if len(current)!=84 or current&previous:
        raise ValueError("S119q2i8a must be source-disjoint from all prior spent DEV")
    if any(not x["completion"].strip() or len(x["completion"])>25
           or len(x["prompt"])>160 for x in tr+dv):
        raise ValueError("S119q2i8a short native EOS source completion contract")
    return {
        "stage":STAGE,
        "origin":ORIGIN,
        "original_synthetic_created_for_experiment_NOT_licensed_natural":True,
        "TRAIN":70,
        "DEV":14,
        "TRAIN_SHA256":source_sha(tr),
        "DEV_SHA256":source_sha(dv),
        "domain_counts_TRAIN":dict(Counter(x["kind"] for x in tr)),
        "domain_counts_DEV":dict(Counter(x["kind"] for x in dv)),
        "no_overlap_prior_or_spent_Q2G_I1_I6":True,
        "DEV_internal_visible_small_NOT_external_independent_blind":True,
        "no_external_reserved_EVAL_loaded":True,
    }
