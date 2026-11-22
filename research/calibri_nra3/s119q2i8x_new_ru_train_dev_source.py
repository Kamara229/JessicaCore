"""S119q2i8x author preregistered new21 TRAIN / 14 internal DEV.

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

STAGE="CALIBRI_S119Q2I8X_NEW_AUTHORED_RU_21TRAIN_AND_14_INTERNAL_DEV"
ORIGIN="research_original_author_S119Q2I8X_new_before_43M_joint_20261122"
CASES={
  "sentence_formulation": [
    [
      "Напишите предложение о геологе, который сортирует найденные камни.",
      "Геолог сортирует редкие образцы минералов."
    ],
    [
      "Составьте предложение о дирижёре перед началом выступления.",
      "Дирижёр готовится к вечернему выступлению."
    ],
    [
      "Напишите одно предложение о продавце, раскладывающем овощи.",
      "Продавец раскладывает свежие сезонные овощи."
    ],
    [
      "Составьте предложение о реставраторе, очищающем старинную картину.",
      "Реставратор бережно очищает старинную картину."
    ],
    [
      "Составьте предложение о фотографе, выбирающем объектив.",
      "Фотограф выбирает подходящий новый объектив."
    ]
  ],
  "case_agreement": [
    [
      "Исправьте предложение: «Мальчик забыл шарф на деревянная скамейке».",
      "Мальчик забыл шарф на деревянной скамейке."
    ],
    [
      "Исправьте падежное окончание: «Учёный работал с сложный прибором».",
      "Учёный работал со сложным прибором."
    ],
    [
      "Исправьте согласование: «Водитель подъехал к просторный парковке».",
      "Водитель подъехал к просторной парковке."
    ],
    [
      "Исправьте форму слова: «Зрители восхищались красивый спектаклем».",
      "Зрители восхищались красивым спектаклем."
    ],
    [
      "Исправьте согласование: «Дети сидели под старый деревом».",
      "Дети сидели под старым деревом."
    ]
  ],
  "agent_action_object": [
    [
      "Пакет Екатерине передал знакомый продавец. Кто передал пакет?",
      "Продавец передал Екатерине пакет."
    ],
    [
      "Стол отремонтировал столяр Виктор. Кто отремонтировал стол?",
      "Столяр Виктор отремонтировал стол."
    ],
    [
      "Письмо Антону вручила сотрудница почты. Кто вручил письмо?",
      "Сотрудница почты вручила Антону письмо."
    ],
    [
      "Сумку незнакомцу вернул честный таксист. Кто вернул сумку?",
      "Таксист вернул сумку незнакомцу."
    ],
    [
      "Билет соседу передал пассажир Николай. Кто передал билет?",
      "Николай передал билет соседу."
    ]
  ],
  "time_causality": [
    [
      "Сначала капитан проверил карту, затем дал команду. Что произошло первым?",
      "Сначала капитан проверил морскую карту."
    ],
    [
      "После утреннего дождя агроном осмотрел поле. Что произошло позже?",
      "Позже агроном осмотрел мокрое поле."
    ],
    [
      "Прежде чем закрыть мастерскую, токарь выключил станок. Что он сделал раньше?",
      "Раньше токарь выключил рабочий станок."
    ],
    [
      "Сначала учитель проверил тетради, затем встретил родителей. Что было позже?",
      "Позже учитель встретил родителей учеников."
    ],
    [
      "Когда прибыл поезд, кондуктор открыл двери. Что произошло после прибытия?",
      "После прибытия кондуктор открыл двери."
    ]
  ],
  "negation_scope": [
    [
      "Борис не покрасил забор, а починил калитку. Что сделал Борис?",
      "Борис починил сломанную калитку."
    ],
    [
      "Экскурсовод показал не фонтан, а памятник. Что показал экскурсовод?",
      "Экскурсовод показал старинный памятник."
    ],
    [
      "Медсестра не взяла градусник, а принесла бинт. Что она принесла?",
      "Медсестра принесла стерильный бинт."
    ],
    [
      "Георгий не позвонил сестре, а написал сообщение. Что сделал Георгий?",
      "Георгий написал короткое сообщение."
    ],
    [
      "Мастер не выбросил шестерню, а очистил её. Что сделал мастер?",
      "Мастер очистил старую шестерню."
    ]
  ],
  "arithmetic_units": [
    [
      "В наборе было пятнадцать пуговиц, семь использовали. Сколько осталось? Ответьте предложением.",
      "Осталось восемь запасных пуговиц."
    ],
    [
      "На пристани стояли четыре лодки, причалили ещё пять. Сколько стало? Ответьте предложением.",
      "Стало девять небольших лодок."
    ],
    [
      "В холодильнике лежали десять яблок, три съели. Сколько осталось? Ответьте предложением.",
      "Осталось семь свежих яблок."
    ],
    [
      "В питомнике росли шесть ёлок, посадили ещё четыре. Сколько стало? Ответьте предложением.",
      "Стало десять молодых ёлок."
    ],
    [
      "На складе было тринадцать коробок, пять увезли. Сколько осталось? Ответьте предложением.",
      "Осталось восемь больших коробок."
    ]
  ],
  "semantic_paraphrase": [
    [
      "Скажите проще: «Оператор осуществил загрузку программного обеспечения».",
      "Оператор установил нужную программу."
    ],
    [
      "Перефразируйте: «Тренер произвёл оценку результатов команды».",
      "Тренер оценил результаты команды."
    ],
    [
      "Скажите проще: «Директор принял решение о закрытии магазина».",
      "Директор решил закрыть магазин."
    ],
    [
      "Перефразируйте: «Строители выполнили восстановление крыши здания».",
      "Строители восстановили повреждённую крышу."
    ],
    [
      "Скажите проще: «Библиотекарь произвёл выдачу читателю книги».",
      "Библиотекарь выдал читателю книгу."
    ]
  ]
}

def rows(split):
    if split not in ("TRAIN","DEV"):
        raise ValueError("Only original X TRAIN and internal synthetic DEV, not EVAL/J7")
    return [
        {"id":f"S119Q2I8X_{kind}_{j:02d}",
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
         w_rows("TRAIN")+w_rows("DEV"))
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
        "fresh_X_TRAIN_rows":21,
        "fresh_X_DEV_rows":14,
        "X21_TRAIN_sha256":digest(train),
        "X14_DEV_sha256":digest(dev),
        "all_X_prompts_disjoint_prior_E_G_K_R_T_W":True,
        "DEV_is_small_synthetic_AND_not_independent_blind":True,
        "protected_616J7_EVAL128_GOLD_inaccessible":True,
    }
