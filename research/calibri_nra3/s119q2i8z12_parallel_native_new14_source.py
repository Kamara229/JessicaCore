"""S119q2i8z12: immutable previously unscored synthetic 14 RU lookahead DEV.

Registered BEFORE any Z12 true-original43M branch expansion. 7 domains
x2 author prompts distinct across all earlier E/G/K/R/T/W/X/Z2/Z9.
Only read-only evaluation; never gradients, hyperparameter tuning, or
blind language benchmark. Preregister fixed beam width 8, third width2,
2 native BPE lookahead from an own-generated first BPE. Gold is used
only to audit candidate coverage AFTER decoding, never to select a beam.
Original 616 J7/EVAL128/GOLD remain untouched.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import json
from research.calibri_nra3.s119q2i8e_new_ru_multiword_train_source import (
    KINDS, rows as e_rows,
)
from research.calibri_nra3.s119q2i8g_new_ru_multiword_dev_source import rows as g_rows
from research.calibri_nra3.s119q2i8k_new_ru_multiword_dev_source import rows as k_rows
from research.calibri_nra3.s119q2i8r_new_ru_DEV_eos_source import rows as r_rows
from research.calibri_nra3.s119q2i8t_new_ru_train_dev_source import rows as t_rows
from research.calibri_nra3.s119q2i8w_new_ru_train_dev_source import rows as w_rows
from research.calibri_nra3.s119q2i8x_new_ru_train_dev_source import rows as x_rows
from research.calibri_nra3.s119q2i8z2_new_ru_train_dev_source import rows as z2_rows
from research.calibri_nra3.s119q2i8z9_new_ru_train_dev_source import rows as z9_rows

STAGE="CALIBRI_S119Q2I8Z12_FRESH14_SOURCE_ONLY_PARALLEL_NATIVE_BPE2_BPE3"
ORIGIN="research_original_author_20261123_Z12_parallel_native_continuation"
BRANCH_WIDTH=8
THIRD_WIDTH=2
HORIZON=2
CASES={
  "sentence_formulation": [
    [
      "Составьте предложение о биологе, который наблюдает за морскими звёздами.",
      "Биолог наблюдает за морскими звёздами."
    ],
    [
      "Напишите предложение о метеорологе, изучающем облака над степью.",
      "Метеоролог изучает облака над степью."
    ]
  ],
  "case_agreement": [
    [
      "Исправьте ошибку в словосочетании: «Корабль подошёл к дальний пристани».",
      "Корабль подошёл к дальней пристани."
    ],
    [
      "Исправьте ошибку: «Подарок лежал под деревянная скамейкой».",
      "Подарок лежал под деревянной скамейкой."
    ]
  ],
  "agent_action_object": [
    [
      "Открытку соседу передала библиотекарь Ирина. Кто передал открытку?",
      "Библиотекарь Ирина передала открытку соседу."
    ],
    [
      "Модель самолёта ученику показал конструктор Павел. Кто показал модель?",
      "Конструктор Павел показал модель самолёта."
    ]
  ],
  "time_causality": [
    [
      "Сначала лаборант промыл пробирку, затем налил раствор. Что он сделал первым?",
      "Сначала лаборант промыл стеклянную пробирку."
    ],
    [
      "Перед посадкой семян Виктор разрыхлил землю. Что произошло раньше?",
      "Раньше Виктор разрыхлил садовую землю."
    ]
  ],
  "negation_scope": [
    [
      "Повар не сварил компот, а приготовил кисель. Что приготовил повар?",
      "Повар приготовил сладкий ягодный кисель."
    ],
    [
      "Пассажир не нашёл зонт, но обнаружил шарф. Что он обнаружил?",
      "Пассажир обнаружил свой тёплый шарф."
    ]
  ],
  "arithmetic_units": [
    [
      "В лавке было семнадцать блокнотов, восемь купили. Сколько осталось? Ответьте предложением.",
      "Осталось девять бумажных блокнотов."
    ],
    [
      "Мастер собрал двенадцать стульев, затем ещё пять. Сколько всего? Ответьте предложением.",
      "Мастер собрал семнадцать деревянных стульев."
    ]
  ],
  "semantic_paraphrase": [
    [
      "Скажите проще: «Редактор осуществил проверку рукописи».",
      "Редактор проверил новую рукопись."
    ],
    [
      "Перефразируйте: «Агроном произвёл оценку состояния почвы».",
      "Агроном оценил состояние почвы."
    ]
  ]
}

def rows(split="DEV"):
    if split!="DEV":
        raise ValueError("Z12 ONLY new internal synthetic DEV, no TRAIN/EVAL/J7/GOLD")
    return [
        {"id":f"S119Q2I8Z12_{kind}_{i:02d}", "kind":kind,"split":"DEV",
         "origin":ORIGIN,"prompt":q,"completion":a}
        for kind in KINDS for i,(q,a) in enumerate(CASES[kind])
    ]

def digest(items):
    return hashlib.sha256("".join(json.dumps(x,sort_keys=True,
        ensure_ascii=False,separators=(",",":"))+"\n" for x in items
    ).encode("utf8")).hexdigest()

def verify():
    now=rows("DEV")
    old=(e_rows()+g_rows()+k_rows()+r_rows()+
         t_rows("TRAIN")+t_rows("DEV")+
         w_rows("TRAIN")+w_rows("DEV")+
         x_rows("TRAIN")+x_rows("DEV")+
         z2_rows("TRAIN")+z2_rows("DEV")+
         z9_rows("TRAIN")+z9_rows("DEV"))
    seen={x["prompt"].strip().casefold() for x in now}
    historical={x["prompt"].strip().casefold() for x in old}
    if not (len(now)==14 and len(seen)==14 and seen.isdisjoint(historical)
            and len({x["id"] for x in now})==14
            and Counter(x["kind"] for x in now)=={k:2 for k in KINDS}
            and set(CASES)==set(KINDS) and BRANCH_WIDTH==8
            and THIRD_WIDTH==2 and HORIZON==2
            and all(x["completion"].endswith(".") and
                    len(x["completion"].split())>=4 and
                    len(x["prompt"])>=26 and x["origin"]==ORIGIN for x in now)):
        raise ValueError("Z12 new DEV preregistered, strict 7 domains x2 and disjoint")
    return {
        "stage":STAGE,
        "DEV14_SHA256":digest(now),
        "fresh_DEV_rows":14,
        "seven_domains_exactly_2_each":True,
        "disjoint_prior_E_G_K_R_T_W_X_Z2_Z9":True,
        "BRANCH_WIDTH":BRANCH_WIDTH,
        "THIRD_WIDTH":THIRD_WIDTH,
        "HORIZON":HORIZON,
        "source_internal_synthetic_not_independent_blind":True,
        "new_DEV_becomes_spent_after_first_forward":True,
        "protected_616J7_EVAL128_GOLD_never_touched":True,
    }
