"""S119q2i8w: *new* precommitted original 21 RU TRAIN + 14 RU DEV.

7 domains x 3 TRAIN and 2 fresh DEV (original author 2026-11-22).
Prompt and completion disjoint by exact strings from ALL previous
E70, G, K, R and T, as well as from each other. Previously spent
DEV prompts and results are NEVER allowed into training. Source is
synthetic small-scale internal and MUST NOT be presented as blind.
"""
from __future__ import annotations
from collections import Counter
import json
import hashlib
from research.calibri_nra3.s119q2i8e_new_ru_multiword_train_source import (
    KINDS,rows as e_rows,
)
from research.calibri_nra3.s119q2i8g_new_ru_multiword_dev_source import rows as g_rows
from research.calibri_nra3.s119q2i8k_new_ru_multiword_dev_source import rows as k_rows
from research.calibri_nra3.s119q2i8r_new_ru_DEV_eos_source import rows as r_rows
from research.calibri_nra3.s119q2i8t_new_ru_train_dev_source import rows as t_rows

STAGE="CALIBRI_S119Q2I8W_AUTHORED_NEW21_TRAIN_NEW14_DEV_EARLY23_NON_EOS_CE"
ORIGIN="research_original_author_S119Q2I8W_20261122_precommitted"
CASES={
  "sentence_formulation": [
    [
      "Составьте предложение о смотрителе зоопарка, который кормит жирафа.",
      "Смотритель кормит высокого жирафа."
    ],
    [
      "Составьте предложение о портном, проверяющем длину рукава.",
      "Портной измеряет длину рукава."
    ],
    [
      "Составьте предложение о машинисте перед отправлением поезда.",
      "Машинист проверяет приборы перед отправлением."
    ],
    [
      "Напишите предложение о садовнике, подстригающем кустарник.",
      "Садовник аккуратно подстригает кустарник."
    ],
    [
      "Напишите предложение о курьере, доставляющем цветы.",
      "Курьер привозит красивый букет цветов."
    ]
  ],
  "case_agreement": [
    [
      "Исправьте падеж: «Повар подошёл к горячая плите».",
      "Повар подошёл к горячей плите."
    ],
    [
      "Исправьте согласование: «Дети гуляли возле высокий деревьев».",
      "Дети гуляли возле высоких деревьев."
    ],
    [
      "Исправьте падеж: «Письма хранились в старый сундуке».",
      "Письма хранились в старом сундуке."
    ],
    [
      "Исправьте согласование: «Мама принесла свежий клубнику».",
      "Мама принесла свежую клубнику."
    ],
    [
      "Исправьте падеж: «Туристы шли по узкая тропе».",
      "Туристы шли по узкой тропе."
    ]
  ],
  "agent_action_object": [
    [
      "Галине помогла перенести сумку соседка. Кто помог Галине?",
      "Соседка помогла Галине перенести сумку."
    ],
    [
      "Подарок преподавателю вручил студент. Кто вручил подарок?",
      "Студент вручил подарок преподавателю."
    ],
    [
      "Кошку погладил мальчик Артём. Кто погладил кошку?",
      "Артём погладил домашнюю кошку."
    ],
    [
      "Гостю налил чай хозяин дома. Кто налил чай?",
      "Хозяин дома налил гостю чай."
    ],
    [
      "Рюкзак брату передала сестра. Кто передал рюкзак?",
      "Сестра передала рюкзак брату."
    ]
  ],
  "time_causality": [
    [
      "Сначала мастер установил дверь, затем покрасил её. Что было раньше?",
      "Мастер сначала установил новую дверь."
    ],
    [
      "Перед началом концерта зрители заняли места. Что произошло сначала?",
      "Сначала зрители заняли свои места."
    ],
    [
      "Когда поднялось солнце, путешественники вышли из палатки. Что было позже?",
      "Позже путешественники вышли из палатки."
    ],
    [
      "После чтения книги школьник написал отзыв. Что было последним действием?",
      "Последним школьник написал короткий отзыв."
    ],
    [
      "Сначала поднялся ветер, затем пошёл дождь. Что было позже?",
      "Позже начался сильный дождь."
    ]
  ],
  "negation_scope": [
    [
      "Секретарь не распечатал договор, а отправил его электронной почтой. Что сделал секретарь?",
      "Секретарь отправил договор электронной почтой."
    ],
    [
      "Илья не купил мяч, а взял его напрокат. Что сделал Илья?",
      "Илья взял мяч напрокат."
    ],
    [
      "Дежурный не открыл склад, а закрыл его. Что сделал дежурный?",
      "Дежурный закрыл дверь склада."
    ],
    [
      "Анна заказала не пиццу, а салат. Что заказала Анна?",
      "Анна заказала овощной салат."
    ],
    [
      "Рабочие не разобрали мост, а укрепили его. Что сделали рабочие?",
      "Рабочие укрепили старый мост."
    ]
  ],
  "arithmetic_units": [
    [
      "У почтальона было четырнадцать писем, шесть он доставил. Сколько осталось? Ответьте предложением.",
      "Осталось восемь недоставленных писем."
    ],
    [
      "На стоянке было три велосипеда, привезли ещё шесть. Сколько стало? Ответьте предложением.",
      "Стало девять исправных велосипедов."
    ],
    [
      "В саду росли одиннадцать тюльпанов, четыре срезали. Сколько осталось? Ответьте предложением.",
      "Осталось семь красивых тюльпанов."
    ],
    [
      "В коробке лежали семь лампочек, добавили ещё три. Сколько стало? Ответьте предложением.",
      "Стало десять новых лампочек."
    ],
    [
      "На столе было девять тарелок, две убрали. Сколько осталось? Ответьте предложением.",
      "Осталось семь чистых тарелок."
    ]
  ],
  "semantic_paraphrase": [
    [
      "Перефразируйте проще: «Монтажник осуществил установку оборудования».",
      "Монтажник установил новое оборудование."
    ],
    [
      "Скажите проще: «Сотрудник выполнил регистрацию посетителей».",
      "Сотрудник зарегистрировал новых посетителей."
    ],
    [
      "Перефразируйте проще: «Механик произвёл проверку тормозной системы».",
      "Механик проверил исправность тормозов."
    ],
    [
      "Скажите проще: «Исследователь осуществил измерение температуры».",
      "Исследователь измерил температуру воды."
    ],
    [
      "Перефразируйте проще: «Волонтёры осуществили сбор необходимых вещей».",
      "Волонтёры собрали необходимые вещи."
    ]
  ]
}

def rows(split):
    if split not in ("TRAIN","DEV"):
        raise ValueError("only fresh W TRAIN or internal DEV; no J7 EVAL128 GOLD")
    return [
        {"id":f"S119Q2I8W_{kind}_{j:02d}",
         "kind":kind,"split":split,"origin":ORIGIN,
         "prompt":prompt,"completion":completion}
        for kind in KINDS for j,(prompt,completion) in enumerate(CASES[kind])
        if (j<3)==(split=="TRAIN")
    ]

def digest(rows_):
    return hashlib.sha256("".join(json.dumps(
        x,ensure_ascii=False,sort_keys=True,separators=(",",":"))+"\n"
        for x in rows_).encode("utf8")).hexdigest()

def verify():
    train=rows("TRAIN")
    dev=rows("DEV")
    previous=(e_rows()+g_rows()+k_rows()+r_rows()+
              t_rows("TRAIN")+t_rows("DEV"))
    allrows=train+dev
    source={x["prompt"].strip().casefold() for x in allrows}
    old={x["prompt"].strip().casefold() for x in previous}
    if not (
        len(train)==21 and len(dev)==14 and len(source)==35 and
        source.isdisjoint(old) and len({x["id"] for x in allrows})==35 and
        Counter(x["kind"] for x in train)=={k:3 for k in KINDS} and
        Counter(x["kind"] for x in dev)=={k:2 for k in KINDS} and
        all(x["origin"]==ORIGIN and len(x["completion"].split())>=3
            and x["completion"].endswith(".") and
            len(x["prompt"])>=15 for x in allrows)
    ):
        raise ValueError("W original author 21 TRAIN / 14 DEV exact disjointness")
    return {
        "stage":STAGE,"new_21_TRAIN_sha256":digest(train),
        "new_14_DEV_sha256":digest(dev),
        "fresh_TRAIN":21,"fresh_internal_DEV":14,
        "source_disjoint_E70_G_K_R_T":True,
        "protected_616J7_EVAL128_GOLD_not_loaded":True,
        "small_visible_synthetic_DEV_NOT_blind":True,
    }
