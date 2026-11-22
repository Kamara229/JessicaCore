"""S119q2i8r: new pre-registered author 14 RU DEV solely for future evaluation.

21+21 previously used E70 TRAIN must remain disjoint. Exactly 28
remaining E70 TRAIN reserved for matched head-only gradient experiments.
DEV never used for optimization, hyperparameter tuning, gradient or weight
updates; 14 small synthetic internal cases are not independently blinded.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import json
from research.calibri_nra3.s119q2i8e_new_ru_multiword_train_source import (
    KINDS,rows as registered_train,
)
from research.calibri_nra3.s119q2i8g_new_ru_multiword_dev_source import (
    rows as previous_g_dev,
)
from research.calibri_nra3.s119q2i8k_new_ru_multiword_dev_source import (
    rows as previous_k_dev,
)
STAGE="CALIBRI_S119Q2I8R_PREDECLARED_14_UNSEEN_INTERNAL_DEV_EOS_CALIBRATION"
ORIGIN="original_author_S119q2i8r_multiword_ru_DEV_only_before_head_training"
CASES={
  "sentence_formulation": [
    [
      "Составьте предложение о смотрителе маяка на берегу моря.",
      "Смотритель проверяет яркий маячный огонь."
    ],
    [
      "Напишите короткое предложение о художнице в мастерской.",
      "Художница смешивает краски на палитре."
    ]
  ],
  "case_agreement": [
    [
      "Исправьте ошибку: «Пешеход подошёл к высокий мост».",
      "Пешеход подошёл к высокому мосту."
    ],
    [
      "Исправьте ошибку: «Садовник любовался красивый роза».",
      "Садовник любовался красивой розой."
    ]
  ],
  "agent_action_object": [
    [
      "Фотографу передал открытку Сергей. Кто передал открытку? Ответьте предложением.",
      "Сергей передал открытку фотографу."
    ],
    [
      "Керамическую вазу Елене подарил Никита. Кто подарил вазу? Ответьте предложением.",
      "Никита подарил Елене вазу."
    ]
  ],
  "time_causality": [
    [
      "Сначала механик завёл двигатель, затем открыл ворота. Что произошло раньше?",
      "Сначала механик завёл двигатель."
    ],
    [
      "После того как закончился дождь, дети вышли во двор. Что произошло позже?",
      "Позже дети вышли во двор."
    ]
  ],
  "negation_scope": [
    [
      "Мария не варила суп, она пекла пирог. Что делала Мария? Полной фразой.",
      "Мария пекла домашний пирог."
    ],
    [
      "Фёдор не читал журнал, он читал роман. Что он читал? Полной фразой.",
      "Фёдор читал интересный роман."
    ]
  ],
  "arithmetic_units": [
    [
      "На прилавке было девять яблок, четыре продали. Сколько осталось? Ответьте предложением.",
      "Осталось пять спелых яблок."
    ],
    [
      "В коробке было шесть мелков, положили ещё пять. Сколько стало? Ответьте предложением.",
      "Стало одиннадцать цветных мелков."
    ]
  ],
  "semantic_paraphrase": [
    [
      "Перефразируйте проще: «Участник соревнования завершил дистанцию».",
      "Спортсмен закончил свой забег."
    ],
    [
      "Скажите проще: «Техник произвёл замену повреждённой детали».",
      "Техник заменил сломанную деталь."
    ]
  ]
}

def rows(split="DEV"):
    if split!="DEV":
        raise ValueError("r author source is ONLY DEV: never TRAIN/EVAL/J7")
    return [
        {"id":f"S119Q2I8R_{kind}_{i:02d}","kind":kind,
         "split":"DEV","origin":ORIGIN,"prompt":prompt,"completion":answer}
        for kind in KINDS
        for i,(prompt,answer) in enumerate(CASES[kind])
    ]

def verify():
    data=rows()
    previous=registered_train()+previous_g_dev()+previous_k_dev()
    prior={x["prompt"].strip().casefold() for x in previous}
    prompts=[x["prompt"].strip().casefold() for x in data]
    if not (len(data)==14 and len(set(prompts))==14 and
            set(prompts).isdisjoint(prior) and
            Counter(x["kind"] for x in data)=={k:2 for k in KINDS} and
            all(len(x["completion"].split())>=3
                and x["completion"].endswith(".")
                and x["origin"]==ORIGIN for x in data)):
        raise ValueError("new 14 original DEV must be balanced and source-disjoint")
    serialized="".join(
        json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(",",":"))+"\n"
        for x in data)
    return {
        "stage":STAGE,"exact_14_new_source_DEV":14,
        "source_sha256":hashlib.sha256(serialized.encode("utf8")).hexdigest(),
        "DEV_never_in_optimizer_or_teacher":True,
        "separate_from_70E_TRAIN_and_14G_14K_visible_DEV":True,
        "synthetic_internal_visible_not_external_blind":True,
        "protected_616J7_EVAL128_not_present":True,
    }
