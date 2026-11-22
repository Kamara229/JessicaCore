"""S119q2i8z2 author preregistered new21 TRAIN / 14 internal DEV.

Exact 7 domains × 3 TRAIN, 2 DEV (NEW); all prompts source-disjoint
from E70 + G K R T W synthetic datasets, with SHA. Never use source
DEV / hidden 616 J7 / EVAL128 / GOLD during gradient, training,
hyperparameter selection. Small synthetic DEV becomes spent on first
physical scoring; cannot claim independent blinded generalization.
"""
from __future__ import annotations
from collections import Counter
import json
import hashlib
from research.calibri_nra3.s119q2i8e_new_ru_multiword_train_source import (
    KINDS, rows as e_rows,
)
from research.calibri_nra3.s119q2i8g_new_ru_multiword_dev_source import rows as g_rows
from research.calibri_nra3.s119q2i8k_new_ru_multiword_dev_source import rows as k_rows
from research.calibri_nra3.s119q2i8r_new_ru_DEV_eos_source import rows as r_rows
from research.calibri_nra3.s119q2i8t_new_ru_train_dev_source import rows as t_rows
from research.calibri_nra3.s119q2i8w_new_ru_train_dev_source import rows as w_rows
from research.calibri_nra3.s119q2i8x_new_ru_train_dev_source import rows as x_rows

STAGE="CALIBRI_S119Q2I8Z2_NEW_AUTHORED_21TRAIN_AND_14_DEV_FOR_CORE_GATE_TRAINONLY"
ORIGIN="research_original_author_S119Q2I8Z2_new_before_43M_joint_20261122"
CASES={
  "sentence_formulation": [
    [
      "Напишите предложение о водолазе, который проверяет якорь корабля.",
      "Водолаз осматривает тяжёлый корабельный якорь."
    ],
    [
      "Составьте предложение о часовщике, который чинит наручные часы.",
      "Часовщик ремонтирует старые наручные часы."
    ],
    [
      "Напишите предложение о ветеринаре, осматривающем заболевшую лошадь.",
      "Ветеринар внимательно осматривает больную лошадь."
    ],
    [
      "Составьте предложение о библиотекаре, протирающем книжные полки.",
      "Библиотекарь протирает высокие книжные полки."
    ],
    [
      "Напишите предложение о кондитере, украшающем праздничный торт.",
      "Кондитер украшает большой праздничный торт."
    ]
  ],
  "case_agreement": [
    [
      "Исправьте согласование: «Гость любовался яркий картиной».",
      "Гость любовался яркой картиной."
    ],
    [
      "Исправьте падеж: «Студент работал над сложная задачей».",
      "Студент работал над сложной задачей."
    ],
    [
      "Исправьте предложение: «Туристы подошли к высокий башне».",
      "Туристы подошли к высокой башне."
    ],
    [
      "Исправьте согласование: «Моряк рассказал о дальний путешествии».",
      "Моряк рассказал о дальнем путешествии."
    ],
    [
      "Исправьте падеж: «Ребята отдыхали возле зелёный поляны».",
      "Ребята отдыхали возле зелёной поляны."
    ]
  ],
  "agent_action_object": [
    [
      "План архитектору передала помощница Ольга. Кто передал план?",
      "Помощница Ольга передала план архитектору."
    ],
    [
      "Портфель Павлу вернул охранник здания. Кто вернул портфель?",
      "Охранник здания вернул Павлу портфель."
    ],
    [
      "Дверь соседу открыл молодой дворник. Кто открыл дверь?",
      "Молодой дворник открыл соседу дверь."
    ],
    [
      "Счетоводу документы передал кассир магазина. Кто передал документы?",
      "Кассир магазина передал счетоводу документы."
    ],
    [
      "Корзину Марине принесла её подруга. Кто принёс корзину?",
      "Подруга принесла Марине корзину."
    ]
  ],
  "time_causality": [
    [
      "До начала экзамена преподаватель раздал задания. Что было сначала?",
      "Сначала преподаватель раздал письменные задания."
    ],
    [
      "Когда закончился снегопад, дорожники расчистили тротуар. Что было позже?",
      "Позже дорожники расчистили городской тротуар."
    ],
    [
      "Перед посадкой в автобус турист купил билет. Что произошло раньше?",
      "Раньше турист купил автобусный билет."
    ],
    [
      "После обеда мастер настроил измерительный прибор. Что было после обеда?",
      "После обеда мастер настроил прибор."
    ],
    [
      "Когда зал открыли, зрители вошли внутрь. Что было сначала?",
      "Сначала сотрудники открыли концертный зал."
    ]
  ],
  "negation_scope": [
    [
      "Егор не выбросил книгу, а подарил её соседу. Что сделал Егор?",
      "Егор подарил книгу своему соседу."
    ],
    [
      "Инженер не заменил двигатель, а проверил проводку. Что проверил инженер?",
      "Инженер проверил электрическую проводку."
    ],
    [
      "Светлана заказала не сок, а компот. Что заказала Светлана?",
      "Светлана заказала ягодный компот."
    ],
    [
      "Плотник не красил окно, а заменил ручку. Что сделал плотник?",
      "Плотник заменил оконную ручку."
    ],
    [
      "Пассажир не забыл чемодан, а оставил зонт. Что оставил пассажир?",
      "Пассажир оставил чёрный зонт."
    ]
  ],
  "arithmetic_units": [
    [
      "В мастерской было семнадцать гаек, девять использовали. Сколько осталось? Ответьте фразой.",
      "Осталось восемь металлических гаек."
    ],
    [
      "В вагоне находились восемь пассажиров, вошли ещё четыре. Сколько стало? Ответьте фразой.",
      "Стало двенадцать новых пассажиров."
    ],
    [
      "В упаковке было двенадцать карандашей, семь раздали. Сколько осталось? Ответьте фразой.",
      "Осталось пять цветных карандашей."
    ],
    [
      "На складе было шесть мешков, привезли ещё семь. Сколько стало? Ответьте фразой.",
      "Стало тринадцать тяжёлых мешков."
    ],
    [
      "На стенде висели девять плакатов, четыре сняли. Сколько осталось? Ответьте фразой.",
      "Осталось пять учебных плакатов."
    ]
  ],
  "semantic_paraphrase": [
    [
      "Перефразируйте проще: «Специалист произвёл замену повреждённого кабеля».",
      "Специалист заменил повреждённый кабель."
    ],
    [
      "Скажите проще: «Работник осуществил сортировку поступивших посылок».",
      "Работник рассортировал новые посылки."
    ],
    [
      "Перефразируйте: «Мастер выполнил проверку надёжности крепления».",
      "Мастер проверил прочность крепления."
    ],
    [
      "Скажите проще: «Путешественник осуществил бронирование номера».",
      "Путешественник забронировал удобный номер."
    ],
    [
      "Перефразируйте: «Механик произвёл очистку деталей двигателя».",
      "Механик очистил детали двигателя."
    ]
  ]
}

def rows(split):
    if split not in ("TRAIN","DEV"):
        raise ValueError("Only original X TRAIN and internal synthetic DEV, not EVAL/J7")
    return [
        {"id":f"S119Q2I8Z2_{kind}_{j:02d}",
         "kind":kind,"split":split,"origin":ORIGIN,
         "prompt":question,"completion":answer}
        for kind in KINDS for j,(question,answer) in enumerate(CASES[kind])
        if (j<3)==(split=="TRAIN")
    ]

def digest(sources):
    return hashlib.sha256("".join(
        json.dumps(v,sort_keys=True,ensure_ascii=False,
                   separators=(",",":"))+"\n"
        for v in sources).encode("utf8")).hexdigest()

def verify():
    train,dev=rows("TRAIN"),rows("DEV")
    old=(e_rows()+g_rows()+k_rows()+r_rows()+
         t_rows("TRAIN")+t_rows("DEV")+
         w_rows("TRAIN")+w_rows("DEV")+
         x_rows("TRAIN")+x_rows("DEV"))
    seen={r["prompt"].strip().casefold() for r in train+dev}
    old_seen={r["prompt"].strip().casefold() for r in old}
    if not (len(train)==21 and len(dev)==14 and len(seen)==35
            and seen.isdisjoint(old_seen)
            and len({r["id"] for r in train+dev})==35
            and Counter(r["kind"] for r in train)=={k:3 for k in KINDS}
            and Counter(r["kind"] for r in dev)=={k:2 for k in KINDS}
            and all(r["origin"]==ORIGIN and r["completion"].endswith(".")
                    and len(r["completion"].split())>=3
                    and len(r["prompt"])>=18 for r in train+dev)):
        raise ValueError("X exact NEW author 21 TRAIN & 14 synthetic DEV disjointness")
    return {
        "stage":STAGE,
        "fresh_Z2_TRAIN_rows":21,
        "fresh_Z2_DEV_rows":14,
        "Z2_21_TRAIN_sha256":digest(train),
        "Z2_14_DEV_sha256":digest(dev),
        "all_X_prompts_disjoint_prior_E_G_K_R_T_W_X":True,
        "DEV_is_small_synthetic_AND_not_independent_blind":True,
        "protected_616J7_EVAL128_GOLD_inaccessible":True,
    }
