"""S119q2i8z16: original Russian 28 TRAIN / 14 internal synthetic DEV.

PRE-REGISTERED source ONLY for CHEAP native49k candidate reranker training:
seven kinds x4 new TRAIN and x2 new DEV. Original human targets used
only in TRAIN loss or AFTER DEV candidate selection, never input to
model scoring. No reusing previously SPENT DEV to train or tune.
All source prompts strictly disjoint E/G/K/R/T/W/X/Z2/Z9/Z12/Z14.
Z16 DEV is not independently blind and becomes spent after first test.
No 616 J7/EVAL128/GOLD, no production, always gate0.
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

STAGE="CALIBRI_S119Q2I8Z16_NEW_RERANKER_28TRAIN_14DEV_PREREG"
ORIGIN="original_author_s119q2i8z16_20261123_new_reranker"
CANDIDATE_K=64
TRAIN_PER_KIND=4
DEV_PER_KIND=2
CASES={
  "sentence_formulation": [
    [
      "Составьте предложение о микологе, изучающем лесные грибы.",
      "Миколог исследует необычные лесные грибы."
    ],
    [
      "Напишите одно предложение о моряке, проверяющем навигационные приборы.",
      "Моряк проверяет точность навигационных приборов."
    ],
    [
      "Опишите одним предложением скульптора, который обрабатывает мрамор.",
      "Скульптор обрабатывает светлый мраморный блок."
    ],
    [
      "Напишите фразу о цветоводе, пересаживающем молодые кусты.",
      "Цветовод пересаживает молодые розовые кусты."
    ],
    [
      "Составьте предложение о часовщике, ремонтирующем настольные часы.",
      "Часовщик ремонтирует старые настольные часы."
    ],
    [
      "Опишите одним предложением архитектора, изучающего чертежи стадиона.",
      "Архитектор изучает подробные чертежи стадиона."
    ]
  ],
  "case_agreement": [
    [
      "Исправьте грамматику: «Мальчик подбежал к зелёный беседке».",
      "Мальчик подбежал к зелёной беседке."
    ],
    [
      "Исправьте окончание: «Книга лежала под тяжёлый коробкой».",
      "Книга лежала под тяжёлой коробкой."
    ],
    [
      "Исправьте падеж: «Рабочий подошёл к высокая лестнице».",
      "Рабочий подошёл к высокой лестнице."
    ],
    [
      "Исправьте ошибку: «Друзья любовались яркая кометой».",
      "Друзья любовались яркой кометой."
    ],
    [
      "Исправьте ошибку: «Письмо находилось возле большой окно».",
      "Письмо находилось возле большого окна."
    ],
    [
      "Исправьте грамматику: «Подарок достался старший сестре».",
      "Подарок достался старшей сестре."
    ]
  ],
  "agent_action_object": [
    [
      "Письмо учителю передал ученик Глеб. Кто передал письмо учителю?",
      "Ученик Глеб передал письмо учителю."
    ],
    [
      "Аквариум соседке подарила продавец Светлана. Кто подарил аквариум?",
      "Продавец Светлана подарила соседке аквариум."
    ],
    [
      "Книгу редактору отправил писатель Мирон. Кто отправил книгу?",
      "Писатель Мирон отправил редактору книгу."
    ],
    [
      "Таблицу инженеру передала аналитик Ольга. Кто передал таблицу?",
      "Аналитик Ольга передала инженеру таблицу."
    ],
    [
      "Посылку спортсмену доставил водитель Семён. Кто доставил посылку?",
      "Водитель Семён доставил спортсмену посылку."
    ],
    [
      "Чертёж ученице показала наставница Вера. Кто показал чертёж?",
      "Наставница Вера показала чертёж ученице."
    ]
  ],
  "time_causality": [
    [
      "Сначала спасатель осмотрел мост, затем вызвал команду. Что сделал раньше?",
      "Сначала спасатель осмотрел старый мост."
    ],
    [
      "До измерения давления лаборант очистил прибор. Что произошло раньше?",
      "Раньше лаборант очистил измерительный прибор."
    ],
    [
      "После проверки сумки турист вышел из дома. Что произошло позже?",
      "Позже турист вышел из дома."
    ],
    [
      "Прежде чем зажечь фонарь, сторож нашёл батарейки. Что было первым?",
      "Сначала сторож нашёл новые батарейки."
    ],
    [
      "Когда вода закипела, повар добавил рис. Что произошло позже?",
      "Позже повар добавил промытый рис."
    ],
    [
      "Перед фотографированием Марк протёр объектив. Что произошло первым?",
      "Сначала Марк протёр объектив камеры."
    ]
  ],
  "negation_scope": [
    [
      "Гонщик не выбрал велосипед, а приобрёл самокат. Что приобрёл гонщик?",
      "Гонщик приобрёл новый городской самокат."
    ],
    [
      "Наталья не красила стены, а вымыла пол. Что вымыла Наталья?",
      "Наталья вымыла деревянный пол комнаты."
    ],
    [
      "Врач не взял фонендоскоп, а использовал термометр. Что использовал врач?",
      "Врач использовал цифровой медицинский термометр."
    ],
    [
      "Редактор не удалил фотографию, а исправил подпись. Что исправил редактор?",
      "Редактор исправил подпись под фотографией."
    ],
    [
      "Инспектор не проверял билет, а осмотрел удостоверение. Что осмотрел инспектор?",
      "Инспектор осмотрел предъявленное ему удостоверение."
    ],
    [
      "Пчеловод не купил улей, а отремонтировал рамки. Что отремонтировал пчеловод?",
      "Пчеловод отремонтировал деревянные ульевые рамки."
    ]
  ],
  "arithmetic_units": [
    [
      "Пекарь испёк тринадцать пирогов, продал пять. Сколько осталось? Ответьте предложением.",
      "Осталось восемь свежих домашних пирогов."
    ],
    [
      "Школьник собрал восемнадцать марок, подарил семь. Сколько осталось? Ответьте предложением.",
      "Осталось одиннадцать коллекционных почтовых марок."
    ],
    [
      "На складе было двадцать две пачки, забрали девять. Сколько осталось? Ответьте предложением.",
      "Осталось тринадцать бумажных упаковок товара."
    ],
    [
      "Курьер развёз шесть коробок утром и восемь вечером. Сколько всего? Ответьте предложением.",
      "Курьер развёз четырнадцать картонных коробок."
    ],
    [
      "На ферме было двенадцать гусей, появились ещё четыре. Сколько стало? Ответьте предложением.",
      "На ферме стало шестнадцать белых гусей."
    ],
    [
      "В музее висело девять картин, добавили ещё шесть. Сколько стало? Ответьте предложением.",
      "В музее стало пятнадцать больших картин."
    ]
  ],
  "semantic_paraphrase": [
    [
      "Перефразируйте проще: «Смотритель произвёл осмотр выставочного зала».",
      "Смотритель осмотрел просторный выставочный зал."
    ],
    [
      "Скажите проще: «Диспетчер осуществил уточнение времени отправления».",
      "Диспетчер уточнил время отправления автобуса."
    ],
    [
      "Перефразируйте: «Мастер провёл диагностику электрического насоса».",
      "Мастер проверил работу электрического насоса."
    ],
    [
      "Скажите проще: «Организатор произвёл регистрацию гостей мероприятия».",
      "Организатор зарегистрировал гостей праздничного мероприятия."
    ],
    [
      "Перефразируйте: «Библиотекарь осуществил выдачу книг посетителям».",
      "Библиотекарь выдал книги новым посетителям."
    ],
    [
      "Скажите проще: «Техник произвёл настройку звукового оборудования».",
      "Техник настроил новое звуковое оборудование."
    ]
  ]
}

def rows(split):
    if split not in ("TRAIN","DEV"):
        raise ValueError("Only source Z16 TRAIN/DEV, no GOLD/EVAL/J7")
    return [
      {"id":f"S119Q2I8Z16_{kind}_{idx:02d}","kind":kind,
       "split":split,"origin":ORIGIN,"prompt":prompt,"completion":answer}
      for kind in KINDS for idx,(prompt,answer) in enumerate(CASES[kind])
      if (idx<TRAIN_PER_KIND)==(split=="TRAIN")
    ]

def digest(v):
    return hashlib.sha256("".join(
        json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":"))+"\n"
        for x in v).encode("utf8")).hexdigest()

def verify():
    train,dev=rows("TRAIN"),rows("DEV")
    spent=(e_rows()+g_rows()+k_rows()+r_rows()
           +t_rows("TRAIN")+t_rows("DEV")
           +w_rows("TRAIN")+w_rows("DEV")
           +x_rows("TRAIN")+x_rows("DEV")
           +z2_rows("TRAIN")+z2_rows("DEV")
           +z9_rows("TRAIN")+z9_rows("DEV")
           +z12_rows("DEV")+z14_rows("DEV"))
    unique={v["prompt"].strip().casefold() for v in train+dev}
    prior={v["prompt"].strip().casefold() for v in spent}
    if not (
        len(train)==28 and len(dev)==14 and len(unique)==42
        and unique.isdisjoint(prior)
        and len({v["id"] for v in train+dev})==42
        and set(CASES)==set(KINDS)
        and Counter(v["kind"] for v in train)=={k:4 for k in KINDS}
        and Counter(v["kind"] for v in dev)=={k:2 for k in KINDS}
        and CANDIDATE_K==64 and TRAIN_PER_KIND==4 and DEV_PER_KIND==2
        and all(v["origin"]==ORIGIN and v["completion"].endswith(".")
            and len(v["completion"].split())>=4 and len(v["prompt"])>=24
            for v in train+dev)):
        raise ValueError("Z16 fresh 28 TRAIN 14 DEV source SHA and split violation")
    return {
      "stage":STAGE,"TRAIN_n":28,"DEV_n":14,
      "TRAIN_sha256":digest(train),"DEV_sha256":digest(dev),
      "seven_domains_train4_dev2":True,
      "disjoint_all_prior_E_G_K_R_T_W_X_Z2_Z9_Z12_Z14":True,
      "fixed_candidate_topK":CANDIDATE_K,
      "only_NEW_human_TRAIN_targets_in_reranker_gradient":True,
      "DEV_not_independent_blind":True,
      "no_J7_616_EVAL128_GOLD_in_training":True,
    }
