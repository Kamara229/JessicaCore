"""S119q2i8t: 21 pre-registered original Russian TRAIN and 14 novel DEV.

Separate new author synthetic 7 domains × (3 TRAIN + 2 DEV). All source
SHA pinned, no 616 J7 / EVAL128 / GOLD, prior R DEV never used for Adam.
This is a small internal synthetic test only, not independent language
generalization. Do not tune after evaluation or claim blind proof.
"""
from __future__ import annotations
import json
import hashlib
from collections import Counter
from research.calibri_nra3.s119q2i8e_new_ru_multiword_train_source import (
    KINDS,rows as e70,
)
from research.calibri_nra3.s119q2i8g_new_ru_multiword_dev_source import (
    rows as g_dev,
)
from research.calibri_nra3.s119q2i8k_new_ru_multiword_dev_source import (
    rows as k_dev,
)
from research.calibri_nra3.s119q2i8r_new_ru_DEV_eos_source import (
    rows as r_dev,
)
STAGE="CALIBRI_S119Q2I8T_NEW_PRECOMMITTED_ORIGINAL_RU_21TRAIN_14DEV"
ORIGIN="original_author_20261122_new_RU_21TRAIN_14DEV_before_43M_joint_EOS"
CASES={
  "sentence_formulation": [
    [
      "Составьте одно предложение о рыбаке, который готовит лодку к выходу.",
      "Рыбак аккуратно проверяет прочные вёсла."
    ],
    [
      "Составьте предложение о лаборанте, изучающем образец почвы.",
      "Лаборант внимательно исследует влажную почву."
    ],
    [
      "Составьте предложение о пекаре, достающем хлеб из печи.",
      "Пекарь осторожно достаёт свежий хлеб."
    ],
    [
      "Напишите одно предложение о горничной, открывающей окно.",
      "Горничная открывает широкое окно."
    ],
    [
      "Напишите одно предложение о мастере, смазывающем старый замок.",
      "Мастер смазывает ржавый дверной замок."
    ]
  ],
  "case_agreement": [
    [
      "Исправьте согласование: «Мы подошли к деревянный воротам».",
      "Мы подошли к деревянным воротам."
    ],
    [
      "Исправьте согласование: «Письмо лежало в почтовый ящике».",
      "Письмо лежало в почтовом ящике."
    ],
    [
      "Исправьте согласование: «Она гордилась новый проектом».",
      "Она гордилась новым проектом."
    ],
    [
      "Исправьте согласование: «Мальчик играл с маленький котёнком».",
      "Мальчик играл с маленьким котёнком."
    ],
    [
      "Исправьте согласование: «Птицы летели над широкий рекой».",
      "Птицы летели над широкой рекой."
    ]
  ],
  "agent_action_object": [
    [
      "Наташе передала тетрадь Ольга. Кто передал тетрадь? Ответьте полным предложением.",
      "Ольга передала тетрадь Наташе."
    ],
    [
      "Грузовик осмотрел механик Андрей. Кто осмотрел грузовик?",
      "Механик Андрей осмотрел грузовик."
    ],
    [
      "Бабушке прочитала письмо внучка. Кто прочитал письмо?",
      "Внучка прочитала письмо бабушке."
    ],
    [
      "Мальчику протянула шарф его сестра. Кто протянул шарф?",
      "Сестра протянула мальчику шарф."
    ],
    [
      "Картину музейному смотрителю показал художник. Кто показал картину?",
      "Художник показал картину смотрителю."
    ]
  ],
  "time_causality": [
    [
      "Сначала дворник убрал листья, потом открыл ворота. Какое действие было первым?",
      "Сначала дворник убрал сухие листья."
    ],
    [
      "После отключения света сторож зажёг фонарь. Что сделал сторож?",
      "Сторож зажёг переносной фонарь."
    ],
    [
      "После утренней разминки спортсмен пошёл на тренировку. Что было раньше?",
      "Раньше спортсмен сделал утреннюю разминку."
    ],
    [
      "Когда закончилось совещание, инженер подписал отчёт. Что произошло позже?",
      "Позже инженер подписал готовый отчёт."
    ],
    [
      "Перед закатом туристы вернулись на стоянку. Когда они вернулись?",
      "Туристы вернулись перед закатом."
    ]
  ],
  "negation_scope": [
    [
      "Врач не выписал рецепт, а назначил обследование. Что назначил врач?",
      "Врач назначил дополнительное обследование."
    ],
    [
      "Курьер не отправил посылку, а забрал её со склада. Что сделал курьер?",
      "Курьер забрал посылку со склада."
    ],
    [
      "Ольга купила не чай, а кофе. Что купила Ольга?",
      "Ольга купила ароматный кофе."
    ],
    [
      "Мальчик не потерял мяч, а отдал его другу. Что он сделал?",
      "Мальчик отдал мяч другу."
    ],
    [
      "Студент не писал доклад, а читал учебник. Что делал студент?",
      "Студент читал новый учебник."
    ]
  ],
  "arithmetic_units": [
    [
      "В пакете лежало восемь орехов, три съели. Сколько осталось? Ответьте предложением.",
      "Осталось пять целых орехов."
    ],
    [
      "У Марии было семь карандашей, она купила ещё два. Сколько стало? Ответьте предложением.",
      "Стало девять цветных карандашей."
    ],
    [
      "В автобусе ехали десять пассажиров, четверо вышли. Сколько осталось? Ответьте предложением.",
      "Осталось шесть пассажиров автобуса."
    ],
    [
      "На полке стояли пять книг, принесли ещё четыре. Сколько стало? Ответьте предложением.",
      "Стало девять интересных книг."
    ],
    [
      "В корзине было двенадцать груш, продали пять. Сколько осталось? Ответьте предложением.",
      "Осталось семь спелых груш."
    ]
  ],
  "semantic_paraphrase": [
    [
      "Перефразируйте проще: «Садовод осуществил полив растений».",
      "Садовод полил зелёные растения."
    ],
    [
      "Скажите проще: «Рабочий произвёл очистку поверхности».",
      "Рабочий очистил грязную поверхность."
    ],
    [
      "Перефразируйте проще: «Пассажир осуществил посадку в вагон».",
      "Пассажир вошёл в поезд."
    ],
    [
      "Скажите проще: «Редактор выполнил проверку рукописи».",
      "Редактор проверил новую рукопись."
    ],
    [
      "Перефразируйте проще: «Организаторы осуществили перенос мероприятия».",
      "Организаторы перенесли запланированное мероприятие."
    ]
  ]
}
TRAIN_PER_KIND=3
DEV_PER_KIND=2
def rows(split):
    if split not in ("TRAIN","DEV"):
        raise ValueError("T only TRAIN or INTERNAL DEV, NO EVAL GOLD J7")
    target=[]
    for kind in KINDS:
        for n,(prompt,answer) in enumerate(CASES[kind]):
            if (n<TRAIN_PER_KIND)==(split=="TRAIN"):
                target.append({
                    "id":f"S119Q2I8T_{kind}_{n:02d}",
                    "kind":kind,
                    "split":split,
                    "origin":ORIGIN,
                    "prompt":prompt,
                    "completion":answer,
                })
    return target

def _digest(items):
    return hashlib.sha256("".join(
        json.dumps(x,ensure_ascii=False,sort_keys=True,
                   separators=(",",":"))+"\n"
        for x in items).encode("utf8")).hexdigest()

def verify():
    train=rows("TRAIN");dev=rows("DEV")
    allrows=train+dev
    historical=e70()+g_dev()+k_dev()+r_dev()
    old={x["prompt"].strip().casefold() for x in historical}
    seen={x["prompt"].strip().casefold() for x in allrows}
    valid=(
        len(train)==21 and len(dev)==14
        and len(seen)==35
        and set(seen).isdisjoint(old)
        and Counter(x["kind"] for x in train)=={k:3 for k in KINDS}
        and Counter(x["kind"] for x in dev)=={k:2 for k in KINDS}
        and len({x["id"] for x in allrows})==35
        and all(x["origin"]==ORIGIN and x["split"] in ("TRAIN","DEV")
                and len(x["completion"].split())>=3
                and x["completion"].endswith(".")
                and 15<=len(x["prompt"])<230 for x in allrows)
    )
    if not valid:
        raise ValueError("S119q2i8t exact original TRAIN/DEV provenance fail")
    return {
        "stage":STAGE,
        "original_21_TRAIN_SHA256":_digest(train),
        "original_14_DEV_SHA256":_digest(dev),
        "new_train_rows":21,"new_dev_rows":14,
        "source_disjoint_prior_E70_G_K_R":True,
        "no_EVAL128_GOLD_J7_answers":True,
        "new_dev_NOT_independent_blind":True,
    }
