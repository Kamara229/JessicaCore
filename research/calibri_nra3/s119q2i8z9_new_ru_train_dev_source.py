"""S119q2i8z9 author preregistered new21 TRAIN / 14 internal DEV.

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
from research.calibri_nra3.s119q2i8z2_new_ru_train_dev_source import rows as spent_z2_rows

STAGE="CALIBRI_S119Q2I8Z9_NEW_AUTHORED_21TRAIN_AND_14_DEV_FOR_CORE_GATE_TRAINONLY"
ORIGIN="research_original_author_S119Q2I8Z9_new_before_43M_joint_20261123"
CASES={
  "sentence_formulation": [
    [
      "Напишите одно предложение о геологе, изучающем образец гранита.",
      "Геолог изучает крупный образец гранита."
    ],
    [
      "Составьте предложение о реставраторе, очищающем старую икону.",
      "Реставратор осторожно очищает старинную икону."
    ],
    [
      "Опишите одним предложением садовника, который поливает лилии.",
      "Садовник поливает белые садовые лилии."
    ],
    [
      "Напишите предложение о картографе, который обновляет карту острова.",
      "Картограф обновляет подробную карту острова."
    ],
    [
      "Составьте фразу о пекаре, который вынимает хлеб из печи.",
      "Пекарь вынимает свежий хлеб из печи."
    ]
  ],
  "case_agreement": [
    [
      "Исправьте падеж: «Инженер подошёл к старый мосту».",
      "Инженер подошёл к старому мосту."
    ],
    [
      "Исправьте согласование: «Птица сидела на высокий ветке».",
      "Птица сидела на высокой ветке."
    ],
    [
      "Исправьте ошибку: «Медик помогла маленький девочке».",
      "Медик помогла маленькой девочке."
    ],
    [
      "Исправьте падеж: «Школьники любовались красивый озером».",
      "Школьники любовались красивым озером."
    ],
    [
      "Исправьте окончание: «Письмо лежало в красная папке».",
      "Письмо лежало в красной папке."
    ]
  ],
  "agent_action_object": [
    [
      "Письмо директору передал курьер Никита. Кто передал письмо?",
      "Курьер Никита передал письмо директору."
    ],
    [
      "Лампу в мастерской починила электрик Анна. Кто починил лампу?",
      "Электрик Анна починила лампу в мастерской."
    ],
    [
      "Ключи соседке вернул мальчик Артём. Кто вернул ключи?",
      "Мальчик Артём вернул ключи соседке."
    ],
    [
      "Документ редактору отправила переводчица Лена. Кто отправил документ?",
      "Переводчица Лена отправила документ редактору."
    ],
    [
      "Подарок бабушке принёс внук Иван. Кто принёс подарок?",
      "Внук Иван принёс подарок бабушке."
    ]
  ],
  "time_causality": [
    [
      "До рассвета рыбак подготовил лодку, а позже вышел в море. Что произошло раньше?",
      "Раньше рыбак подготовил лодку к выходу."
    ],
    [
      "После звонка учительница закрыла журнал. Что произошло после звонка?",
      "После звонка учительница закрыла журнал."
    ],
    [
      "Прежде чем включить компьютер, оператор проверил кабель. Что он сделал сначала?",
      "Сначала оператор проверил соединительный кабель."
    ],
    [
      "Сначала поезд остановился, затем проводник открыл дверь. Что произошло позже?",
      "Позже проводник открыл дверь вагона."
    ],
    [
      "Перед поливом цветов Мария наполнила лейку. Что произошло первым?",
      "Сначала Мария наполнила садовую лейку."
    ]
  ],
  "negation_scope": [
    [
      "Олег не купил чай, а выбрал кофе. Что купил Олег?",
      "Олег купил ароматный кофе."
    ],
    [
      "Врач не осмотрел руку, а проверил ногу. Что осмотрел врач?",
      "Врач осмотрел больную ногу."
    ],
    [
      "Девочка не закрыла окно, а открыла дверь. Что открыла девочка?",
      "Девочка открыла деревянную дверь."
    ],
    [
      "Художник не нарисовал дом, а изобразил мост. Что изобразил художник?",
      "Художник изобразил каменный мост."
    ],
    [
      "Мастер не заменил экран, а починил кнопку. Что починил мастер?",
      "Мастер починил неисправную кнопку."
    ]
  ],
  "arithmetic_units": [
    [
      "У фермера было четырнадцать корзин, четыре продали. Сколько осталось? Ответьте предложением.",
      "Осталось десять плетёных корзин."
    ],
    [
      "На полке стояло девять банок, поставили ещё пять. Сколько стало? Ответьте предложением.",
      "Стало четырнадцать стеклянных банок."
    ],
    [
      "На складе лежало двадцать ремней, семь отправили. Сколько осталось? Ответьте предложением.",
      "Осталось тринадцать кожаных ремней."
    ],
    [
      "В коробке было шестнадцать ламп, шесть забрали. Сколько осталось? Ответьте предложением.",
      "Осталось десять запасных ламп."
    ],
    [
      "На пристани стояло восемь лодок, приплыли ещё три. Сколько стало? Ответьте предложением.",
      "Стало одиннадцать небольших лодок."
    ]
  ],
  "semantic_paraphrase": [
    [
      "Перефразируйте проще: «Секретарь осуществил регистрацию входящей корреспонденции».",
      "Секретарь зарегистрировал входящие письма."
    ],
    [
      "Скажите проще: «Экспедитор произвёл доставку заказанных коробок».",
      "Экспедитор доставил заказанные коробки."
    ],
    [
      "Перефразируйте: «Техник выполнил измерение температуры раствора».",
      "Техник измерил температуру раствора."
    ],
    [
      "Скажите проще: «Диспетчер осуществил проверку графика рейсов».",
      "Диспетчер проверил график рейсов."
    ],
    [
      "Перефразируйте: «Бригадир произвёл осмотр строительных лесов».",
      "Бригадир осмотрел строительные леса."
    ]
  ]
}

def rows(split):
    if split not in ("TRAIN","DEV"):
        raise ValueError("Only original Z9 TRAIN and internal synthetic DEV, not EVAL/J7")
    return [
        {"id":f"S119Q2I8Z9_{kind}_{j:02d}",
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
         x_rows("TRAIN")+x_rows("DEV")+
         spent_z2_rows("TRAIN")+spent_z2_rows("DEV"))
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
        raise ValueError("Z9 exact NEW author 21 TRAIN & 14 synthetic DEV disjointness")
    return {
        "stage":STAGE,
        "fresh_Z9_TRAIN_rows":21,
        "fresh_Z9_DEV_rows":14,
        "Z9_21_TRAIN_sha256":digest(train),
        "Z9_14_DEV_sha256":digest(dev),
        "all_Z9_prompts_disjoint_prior_E_G_K_R_T_W_X_Z2":True,
        "DEV_is_small_synthetic_AND_not_independent_blind":True,
        "protected_616J7_EVAL128_GOLD_inaccessible":True,
    }
