"""S119q2i8z18: preregister brand NEW 28TRAIN+14DEV for 12-layer h320 representation audit.

All seven types x 4 TRAIN+2 DEV. No old E/G/K/R/T/W/X/Z2/Z9/Z12/Z14/Z16
source prompts, no J7 616/EVAL128/GOLD. DEV human gold NEVER passed
to ranker or used for training. Before any physical model forward freeze:
baseline FINAL actual native head h320 vs fixed equal-weight mix of layer
outputs at 3, 7, 11 (zero-based 2,6,10) and actual final LM h320.
Each gets the SAME lowrank2 1280-param readout, same top64, optimizer,
and the SAME 28 TRAIN human second BPE. No new CORE/NRA3/NPA1 areas,
new pretrained weights or architecture changes. Internal author DEV
becomes SPENT after first physical scoring; no blind quality claims.
"""
from __future__ import annotations
import hashlib
import json
from collections import Counter
from research.calibri_nra3.s119q2i8e_new_ru_multiword_train_source import KINDS,rows as e_rows
from research.calibri_nra3.s119q2i8g_new_ru_multiword_dev_source import rows as g_rows
from research.calibri_nra3.s119q2i8k_new_ru_multiword_dev_source import rows as k_rows
from research.calibri_nra3.s119q2i8r_new_ru_DEV_eos_source import rows as r_rows
from research.calibri_nra3.s119q2i8t_new_ru_train_dev_source import rows as t_rows
from research.calibri_nra3.s119q2i8w_new_ru_train_dev_source import rows as w_rows
from research.calibri_nra3.s119q2i8x_new_ru_train_dev_source import rows as x_rows
from research.calibri_nra3.s119q2i8z2_new_ru_train_dev_source import rows as z2_rows
from research.calibri_nra3.s119q2i8z9_new_ru_train_dev_source import rows as z9_rows
from research.calibri_nra3.s119q2i8z12_parallel_native_new14_source import rows as z12_rows
from research.calibri_nra3.s119q2i8z14_fresh14_bpe2_coverage_source import rows as z14_rows
from research.calibri_nra3.s119q2i8z16_fresh28train14dev_rerank_source import rows as z16_rows

STAGE="CALIBRI_S119Q2I8Z18_PREREG_NEW28TRAIN14DEV_MULTILAYER_H320_FIXED_CONSENSUS"
ORIGIN="original_author_S119Q2I8Z18_20261123_new_consensus"
KINDS_PER_TRAIN=4
KINDS_PER_DEV=2
CANDIDATE_TOP_K=64
MULTI_LAYER_ZERO_BASED=(2,6,10)
MULTI_LAYER_MIX_WEIGHTS=(.25,.25,.25,.25)
BASELINE="FINAL_LAST_H320"
ALTERNATE="FIXED_L3_L7_L11_FINAL_MEAN_H320"
CASES={
  "sentence_formulation": [
    [
      "Напишите предложение о реставраторе, который очищает древнюю мозаику.",
      "Реставратор осторожно очищает древнюю цветную мозаику."
    ],
    [
      "Составьте предложение о ветеринаре, осматривающем домашнюю черепаху.",
      "Ветеринар внимательно осматривает домашнюю черепаху."
    ],
    [
      "Опишите предложение о геодезисте, измеряющем поле перед строительством.",
      "Геодезист измеряет ровное поле перед строительством."
    ],
    [
      "Напишите предложение о водолазе, изучающем затонувший якорь.",
      "Водолаз изучает тяжёлый затонувший якорь."
    ],
    [
      "Составьте фразу о переплётчике, восстанавливающем старую книгу.",
      "Переплётчик восстанавливает повреждённую старую книгу."
    ],
    [
      "Напишите предложение о садовнике, ухаживающем за лимонным деревом.",
      "Садовник ухаживает за молодым лимонным деревом."
    ]
  ],
  "case_agreement": [
    [
      "Исправьте окончание: «Туристы приблизились к каменный башне».",
      "Туристы приблизились к каменной башне."
    ],
    [
      "Исправьте согласование: «Ключ лежал рядом с красная сумкой».",
      "Ключ лежал рядом с красной сумкой."
    ],
    [
      "Исправьте ошибку: «Лектор говорил о интересная книге».",
      "Лектор говорил об интересной книге."
    ],
    [
      "Исправьте окончание: «Дети играли возле глубокий ямы».",
      "Дети играли возле глубокой ямы."
    ],
    [
      "Исправьте грамматику: «Пассажиры ждали последний электричку».",
      "Пассажиры ждали последнюю электричку."
    ],
    [
      "Исправьте форму: «Художник восхищался яркое закатом».",
      "Художник восхищался ярким закатом."
    ]
  ],
  "agent_action_object": [
    [
      "Справочник архитектору передал редактор Максим. Кто передал справочник?",
      "Редактор Максим передал справочник архитектору."
    ],
    [
      "Награду капитану вручила судья Алёна. Кто вручил награду?",
      "Судья Алёна вручила награду капитану."
    ],
    [
      "Карту экскурсоводу показал студент Роман. Кто показал карту?",
      "Студент Роман показал карту экскурсоводу."
    ],
    [
      "План педагогу отправила секретарь Лидия. Кто отправил план?",
      "Секретарь Лидия отправила план педагогу."
    ],
    [
      "Запчасть механику передал диспетчер Константин. Кто передал запчасть?",
      "Диспетчер Константин передал механику запчасть."
    ],
    [
      "Фотографию музею подарила коллекционер Дарья. Кто подарил фотографию?",
      "Коллекционер Дарья подарила фотографию музею."
    ]
  ],
  "time_causality": [
    [
      "Перед началом лекции преподаватель проверил микрофон. Что случилось раньше?",
      "Раньше преподаватель проверил рабочий микрофон."
    ],
    [
      "После просушки досок плотник нанёс лак. Что произошло позже?",
      "Позже плотник нанёс прозрачный лак."
    ],
    [
      "Сначала медсестра подготовила бинты, потом открыла аптечку. Что было первым?",
      "Сначала медсестра подготовила стерильные бинты."
    ],
    [
      "Прежде чем заправить лодку, рыбак проверил мотор. Что он сделал первым?",
      "Сначала рыбак проверил лодочный мотор."
    ],
    [
      "Повар посолил суп только после добавления овощей. Что произошло раньше?",
      "Сначала повар добавил нарезанные овощи."
    ],
    [
      "После сборки шкафа мастер прикрепил дверцы. Что случилось позже?",
      "Позже мастер прикрепил новые дверцы."
    ]
  ],
  "negation_scope": [
    [
      "Инженер не открыл окно, а включил кондиционер. Что он включил?",
      "Инженер включил настенный бытовой кондиционер."
    ],
    [
      "Водитель не потерял карту, а оставил перчатки. Что он оставил?",
      "Водитель оставил кожаные рабочие перчатки."
    ],
    [
      "Студентка не прочитала журнал, а изучила учебник. Что она изучила?",
      "Студентка изучила новый университетский учебник."
    ],
    [
      "Экскурсовод не показал башню, а описал мост. Что он описал?",
      "Экскурсовод описал старый каменный мост."
    ],
    [
      "Слесарь не заменил кран, а починил трубу. Что он починил?",
      "Слесарь починил повреждённую водопроводную трубу."
    ],
    [
      "Пилот не проверил сиденье, а настроил компас. Что он настроил?",
      "Пилот настроил электронный навигационный компас."
    ]
  ],
  "arithmetic_units": [
    [
      "На полке было двадцать книг, семь взяли. Сколько осталось? Ответьте предложением.",
      "На полке осталось тринадцать интересных книг."
    ],
    [
      "Почтальон доставил четырнадцать писем, затем ещё шесть. Сколько всего? Ответьте предложением.",
      "Почтальон доставил двадцать почтовых писем."
    ],
    [
      "Повар нарезал девятнадцать яблок, девять использовал. Сколько осталось? Ответьте предложением.",
      "Осталось десять нарезанных спелых яблок."
    ],
    [
      "В саду выросло восемь деревьев, посадили ещё семь. Сколько стало? Ответьте предложением.",
      "В саду стало пятнадцать молодых деревьев."
    ],
    [
      "На верстаке лежало одиннадцать деталей, четыре убрали. Сколько осталось? Ответьте предложением.",
      "На верстаке осталось семь металлических деталей."
    ],
    [
      "Пекарня приготовила девять батонов утром и двенадцать вечером. Сколько всего? Ответьте предложением.",
      "Пекарня приготовила двадцать один свежий батон."
    ]
  ],
  "semantic_paraphrase": [
    [
      "Скажите проще: «Астроном осуществил наблюдение за движением планеты».",
      "Астроном наблюдал движение далёкой планеты."
    ],
    [
      "Перефразируйте: «Лаборант произвёл очистку лабораторного сосуда».",
      "Лаборант очистил грязный лабораторный сосуд."
    ],
    [
      "Скажите проще: «Инспектор осуществил проверку технического паспорта».",
      "Инспектор проверил технический паспорт оборудования."
    ],
    [
      "Перефразируйте: «Портной произвёл подгонку праздничного костюма».",
      "Портной подогнал новый праздничный костюм."
    ],
    [
      "Скажите проще: «Редактор выполнил корректировку заголовка сообщения».",
      "Редактор исправил заголовок короткого сообщения."
    ],
    [
      "Перефразируйте: «Инженер осуществил измерение температуры двигателя».",
      "Инженер измерил температуру работающего двигателя."
    ]
  ]
}

def rows(split):
    if split not in ("TRAIN","DEV"):
        raise ValueError("Z18 only new original author TRAIN/DEV, no J7/EVAL128/GOLD")
    return [
        {"id":f"S119Q2I8Z18_{kind}_{i:02d}","kind":kind,"split":split,
         "origin":ORIGIN,"prompt":p,"completion":a}
        for kind in KINDS for i,(p,a) in enumerate(CASES[kind])
        if (i<KINDS_PER_TRAIN)==(split=="TRAIN")
    ]

def digest(seq):
    return hashlib.sha256("".join(
        json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(",",":"))+"\n"
        for x in seq).encode("utf8")).hexdigest()

def verify():
    train,dev=rows("TRAIN"),rows("DEV")
    historical=(e_rows()+g_rows()+k_rows()+r_rows()
        +t_rows("TRAIN")+t_rows("DEV")
        +w_rows("TRAIN")+w_rows("DEV")
        +x_rows("TRAIN")+x_rows("DEV")
        +z2_rows("TRAIN")+z2_rows("DEV")
        +z9_rows("TRAIN")+z9_rows("DEV")
        +z12_rows("DEV")+z14_rows("DEV")
        +z16_rows("TRAIN")+z16_rows("DEV"))
    seen={x["prompt"].strip().casefold() for x in train+dev}
    prior={x["prompt"].strip().casefold() for x in historical}
    if not (len(train)==28 and len(dev)==14 and len(seen)==42
            and seen.isdisjoint(prior)
            and len({x["id"] for x in train+dev})==42
            and set(CASES)==set(KINDS)
            and Counter(x["kind"] for x in train)=={k:4 for k in KINDS}
            and Counter(x["kind"] for x in dev)=={k:2 for k in KINDS}
            and CANDIDATE_TOP_K==64
            and MULTI_LAYER_ZERO_BASED==(2,6,10)
            and MULTI_LAYER_MIX_WEIGHTS==(.25,.25,.25,.25)
            and KINDS_PER_TRAIN==4 and KINDS_PER_DEV==2
            and all(x["origin"]==ORIGIN and x["completion"].endswith(".")
                and len(x["prompt"])>20 and len(x["completion"].split())>=4
                for x in train+dev)):
        raise ValueError("Z18 source split/disjoint/weight or fixed budget failed")
    return {
        "stage":STAGE,"TRAIN_count":28,"DEV_count":14,
        "TRAIN_SHA256":digest(train),"DEV_SHA256":digest(dev),
        "seven_domains_train4_dev2":True,
        "disjoint_old_E_G_K_R_T_W_X_Z2_Z9_Z12_Z14_Z16":True,
        "candidate_topK":CANDIDATE_TOP_K,
        "multilayer_selected_layer_indices":MULTI_LAYER_ZERO_BASED,
        "fixed_mix_weights":MULTI_LAYER_MIX_WEIGHTS,
        "matched_control":BASELINE,"experimental_representation":ALTERNATE,
        "DEV_is_internal_synthetic_not_independent_blind":True,
        "NO_GOLD_EVAL128_616J7_training":True,
        "no_DEV_gold_reaches_training_or_candidate_selection":True,
    }
