"""S119q2i8z21: BRAND NEW 28 TRAIN + 14 internal synthetic DEV, pre-scoring.

Predictive State *proxy*, NOT demonstrated general semantics:
From actual OWN-first hidden h320, train 320->16->320 shadow forecast
of frozen original native-token embedding mean of future human BPE
positions 2/3/4 (index1:4). Candidate target never inserted into
original native49k top64; gold never fed to DEV model/selection.
Compare identical 10240-parameter predictors with real aligned
TRAIN future embeddings vs reproducibly PERMUTED TRAIN future
embedding labels (negative control), SAME original43M hidden/logits
and candidate top64. Train 30 fixed epochs AdamW lr=.01 wd=.1.
Inference re-rank by full native logits + fixed 2.0 * cosine(
predicted_future_state, original_native_candidate_embedding). This
measures ability to learn short-term future representation and BPE2
selection, NOT event structures, reliable semantics, or speed gains.
DEV becomes SPENT on first test; no training on used E/G/K/R/T/W/X/
Z2/Z9/Z12/Z14/Z16/Z18, no 616 J7/EVAL128/GOLD. All gate0.
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
from research.calibri_nra3.s119q2i8z18_fresh28train14dev_multilayer_source import rows as z18_rows

STAGE="CALIBRI_S119Q2I8Z21_FRESH28TRAIN14DEV_ALIGNED_VS_PERMUTED_BPE2_4_PREDICTIVE_STATE_PREREG"
ORIGIN="original_author_S119Q2I8Z21_20261123_predictive_future_state"
TRAIN_PER_KIND=4
DEV_PER_KIND=2
TOP_K=64
LATENT_DIM=16
PREDICTED_FUTURE_POSITIONS=(1,2,3)
FIXED_LOGIT_COSINE_SCALE=2.0
TRAIN_EPOCHS=30
TRAIN_LR=.01
TRAIN_WD=.10
PERMUTED_TRAIN_SEED=119121
CASES={
  "sentence_formulation": [
    [
      "Составьте одно предложение об орнитологе, который наблюдает утренний перелёт ласточек.",
      "Орнитолог наблюдает утренний перелёт быстрых ласточек."
    ],
    [
      "Напишите предложение о палеонтологе, исследующем морскую окаменелость.",
      "Палеонтолог исследует необычную древнюю морскую окаменелость."
    ],
    [
      "Опишите одним предложением инженера, проверяющего подвесной мост.",
      "Инженер тщательно проверяет надёжность подвесного моста."
    ],
    [
      "Напишите предложение о картографе, наносящем новую реку на карту.",
      "Картограф наносит новую извилистую реку на карту."
    ],
    [
      "Составьте фразу о мастере, который изготавливает керамические тарелки.",
      "Мастер изготавливает красивые керамические обеденные тарелки."
    ],
    [
      "Напишите предложение о фотографе, снимающем зимний лес.",
      "Фотограф снимает заснеженный лес ранним утром."
    ]
  ],
  "case_agreement": [
    [
      "Исправьте форму слова: «Путешественник подошёл к старинный арке».",
      "Путешественник подошёл к старинной каменной арке."
    ],
    [
      "Исправьте согласование: «Бабушка укрыла ребёнка тёплая одеялом».",
      "Бабушка укрыла ребёнка тёплым шерстяным одеялом."
    ],
    [
      "Исправьте грамматику: «Гости остановились возле высокий башни».",
      "Гости остановились возле высокой городской башни."
    ],
    [
      "Исправьте падеж: «Девочка любовалась яркое бабочкой».",
      "Девочка любовалась яркой разноцветной бабочкой."
    ],
    [
      "Исправьте ошибку: «Плотник работал с новый рубанком».",
      "Плотник работал с новым острым рубанком."
    ],
    [
      "Исправьте предложение: «Пешеход свернул к узкий тропинке».",
      "Пешеход свернул к узкой лесной тропинке."
    ]
  ],
  "agent_action_object": [
    [
      "Фонарь охраннику передала смотрительница Лариса. Кто передал фонарь?",
      "Смотрительница Лариса передала охраннику запасной фонарь."
    ],
    [
      "Альбом художнице подарил писатель Артур. Кто подарил альбом?",
      "Писатель Артур подарил художнице новый альбом."
    ],
    [
      "Компьютер ученику показала преподавательница София. Кто показал компьютер?",
      "Преподавательница София показала ученику учебный компьютер."
    ],
    [
      "Лекарство фельдшеру принёс курьер Борис. Кто принёс лекарство?",
      "Курьер Борис принёс фельдшеру заказанное лекарство."
    ],
    [
      "Инструмент мастеру отдала помощница Нина. Кто отдал инструмент?",
      "Помощница Нина отдала мастеру рабочий инструмент."
    ],
    [
      "Рисунок преподавателю отправила школьница Юлия. Кто отправил рисунок?",
      "Школьница Юлия отправила преподавателю цветной рисунок."
    ]
  ],
  "time_causality": [
    [
      "Прежде чем закрыть мастерскую, Сергей проверил замок. Что он сделал сначала?",
      "Сначала Сергей проверил прочность дверного замка."
    ],
    [
      "После подготовки почвы огородник посадил редис. Что произошло позже?",
      "Позже огородник посадил ранний красный редис."
    ],
    [
      "Перед установкой антенны техник осмотрел крышу. Что было первым?",
      "Сначала техник осмотрел поверхность высокой крыши."
    ],
    [
      "Сначала рабочий разметил доску, потом сделал распил. Что произошло раньше?",
      "Раньше рабочий разметил длинную деревянную доску."
    ],
    [
      "После сверки адреса курьер передал посылку. Что произошло позже?",
      "Позже курьер передал небольшую почтовую посылку."
    ],
    [
      "Перед стрижкой газона садовник убрал ветки. Что он сделал первым?",
      "Сначала садовник убрал сухие опавшие ветки."
    ]
  ],
  "negation_scope": [
    [
      "Моряк не поднял парус, а проверил штурвал. Что он проверил?",
      "Моряк проверил надёжный корабельный штурвал."
    ],
    [
      "Туристка не купила путеводитель, а выбрала карту. Что выбрала туристка?",
      "Туристка выбрала подробную туристическую карту."
    ],
    [
      "Мастер не покрасил раму, а заменил стекло. Что он заменил?",
      "Мастер заменил разбитое оконное стекло."
    ],
    [
      "Продавец не показал сапоги, а предложил ботинки. Что он предложил?",
      "Продавец предложил удобные кожаные ботинки."
    ],
    [
      "Садовод не полил клумбу, а подрезал куст. Что он подрезал?",
      "Садовод подрезал высокий декоративный куст."
    ],
    [
      "Эксперт не изучил письмо, а открыл отчёт. Что он открыл?",
      "Эксперт открыл подробный технический отчёт."
    ]
  ],
  "arithmetic_units": [
    [
      "Библиотека получила двадцать четыре книги, шесть отправила в филиал. Сколько осталось? Ответьте предложением.",
      "В библиотеке осталось восемнадцать новых книг."
    ],
    [
      "Мастер сделал семь полок и ещё восемь. Сколько полок он сделал? Ответьте предложением.",
      "Мастер сделал пятнадцать деревянных настенных полок."
    ],
    [
      "В автобусе ехало девятнадцать пассажиров, восемь вышли. Сколько осталось? Ответьте предложением.",
      "В автобусе осталось одиннадцать взрослых пассажиров."
    ],
    [
      "Кондитер испёк тринадцать кексов утром и девять вечером. Сколько всего? Ответьте предложением.",
      "Кондитер испёк двадцать два шоколадных кекса."
    ],
    [
      "На складе хранилось шестнадцать мешков, пять вывезли. Сколько осталось? Ответьте предложением.",
      "На складе осталось одиннадцать тяжёлых мешков."
    ],
    [
      "Цветовод вырастил девять тюльпанов, потом ещё семь. Сколько всего? Ответьте предложением.",
      "Цветовод вырастил шестнадцать красивых красных тюльпанов."
    ]
  ],
  "semantic_paraphrase": [
    [
      "Перефразируйте проще: «Археолог осуществил описание найденного предмета».",
      "Археолог описал найденный древний предмет."
    ],
    [
      "Скажите проще: «Техник произвёл проверку состояния кабеля».",
      "Техник проверил состояние электрического кабеля."
    ],
    [
      "Перефразируйте: «Директор осуществил согласование рабочего графика».",
      "Директор согласовал новый рабочий график."
    ],
    [
      "Скажите проще: «Ученик выполнил измерение длины доски».",
      "Ученик измерил длину деревянной доски."
    ],
    [
      "Перефразируйте: «Проводник осуществил осмотр пустого вагона».",
      "Проводник осмотрел свободный пассажирский вагон."
    ],
    [
      "Скажите проще: «Фермер произвёл оценку качества урожая».",
      "Фермер оценил качество собранного урожая."
    ]
  ]
}

def rows(split):
    if split not in ("TRAIN","DEV"):
        raise ValueError("Z21 only BRAND NEW TRAIN/DEV, never J7 EVAL128 GOLD")
    return [{
        "id":f"S119Q2I8Z21_{kind}_{i:02d}", "kind":kind,
        "split":split,"origin":ORIGIN,"prompt":p,"completion":a}
        for kind in KINDS for i,(p,a) in enumerate(CASES[kind])
        if (i<TRAIN_PER_KIND)==(split=="TRAIN")
    ]

def digest(v):
    return hashlib.sha256("".join(
        json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(",",":"))+"\n"
        for x in v).encode("utf8")).hexdigest()

def verify():
    train,dev=rows("TRAIN"),rows("DEV")
    old=(e_rows()+g_rows()+k_rows()+r_rows()
        +t_rows("TRAIN")+t_rows("DEV")
        +w_rows("TRAIN")+w_rows("DEV")
        +x_rows("TRAIN")+x_rows("DEV")
        +z2_rows("TRAIN")+z2_rows("DEV")
        +z9_rows("TRAIN")+z9_rows("DEV")
        +z12_rows("DEV")+z14_rows("DEV")
        +z16_rows("TRAIN")+z16_rows("DEV")
        +z18_rows("TRAIN")+z18_rows("DEV"))
    now_prompts={r["prompt"].strip().casefold() for r in train+dev}
    prior={r["prompt"].strip().casefold() for r in old}
    if not (len(train)==28 and len(dev)==14 and len(now_prompts)==42
            and now_prompts.isdisjoint(prior)
            and set(CASES)==set(KINDS)
            and len({r["id"] for r in train+dev})==42
            and Counter(r["kind"] for r in train)=={k:4 for k in KINDS}
            and Counter(r["kind"] for r in dev)=={k:2 for k in KINDS}
            and (TRAIN_PER_KIND,DEV_PER_KIND)==(4,2)
            and (TOP_K,LATENT_DIM,PREDICTED_FUTURE_POSITIONS)==(64,16,(1,2,3))
            and (FIXED_LOGIT_COSINE_SCALE,TRAIN_EPOCHS,TRAIN_LR,TRAIN_WD,PERMUTED_TRAIN_SEED)==
                 (2.0,30,.01,.10,119121)
            and all(r["origin"]==ORIGIN and r["completion"].endswith(".")
                    and len(r["completion"].split())>=5 and len(r["prompt"])>=35
                    for r in train+dev)):
        raise ValueError("Z21 brand new 28/14 disjoint or fixed predictive state contract violated")
    return {
        "stage":STAGE,"TRAIN_count":28,"DEV_count":14,
        "TRAIN_SHA256":digest(train),"DEV_SHA256":digest(dev),
        "disjoint_old_E_G_K_R_T_W_X_Z2_Z9_Z12_Z14_Z16_Z18":True,
        "seven_domains_4_train_2_dev":True,
        "fixed_topK":TOP_K,"latent_dim":LATENT_DIM,
        "future_BPE_positions":PREDICTED_FUTURE_POSITIONS,
        "fixed_cosine_logit_scale":FIXED_LOGIT_COSINE_SCALE,
        "fixed_epochs":TRAIN_EPOCHS,"fixed_lr":TRAIN_LR,
        "fixed_weight_decay":TRAIN_WD,"permuted_training_seed":PERMUTED_TRAIN_SEED,
        "author_synthetic_DEV_not_independent_blind":True,
        "DEV_becomes_spent_after_first_physical_forward":True,
        "NO_J7_616_EVAL128_GOLD_training":True,
    }
