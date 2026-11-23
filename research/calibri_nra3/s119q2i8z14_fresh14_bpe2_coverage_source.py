"""S119q2i8z14 NEW untouched author 14 RU synthetic DEV for BPE2 candidate coverage.
Precommitted before any ORIGINAL43M logits evaluated. Gold rank is
POSTHOC only after own FIRST native BPE, no training/hyperparameter tuning.
Strict disjoint all historical E/G/K/R/T/W/X/Z2/Z9/Z12. Top-k budgets
8/16/32/64/128/512 locked BEFORE physical test and not promoted.
No protected 616 J7, EVAL128 or GOLD, no LIVE/Render/Supabase writes.
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

STAGE="CALIBRI_S119Q2I8Z14_PREREG_14_NEW_RU_DEV_NATIVE_BPE2_COVERAGE"
ORIGIN="research_original_author_S119Q2I8Z14_20261123_sourceonly"
BUDGETS=(8,16,32,64,128,512)
CASES={
  "sentence_formulation": [
    [
      "Напишите предложение о зоологе, наблюдающем степных журавлей весной.",
      "Зоолог наблюдает за редкими степными журавлями."
    ],
    [
      "Составьте предложение о стеклодуве, который изготавливает маленькую вазу.",
      "Стеклодув выдувает небольшую прозрачную вазу."
    ]
  ],
  "case_agreement": [
    [
      "Исправьте ошибку в предложении: «Проводник подошёл к закрытая калитке».",
      "Проводник подошёл к закрытой калитке."
    ],
    [
      "Исправьте согласование: «Дети любовались яркий радугой».",
      "Дети любовались яркой радугой."
    ]
  ],
  "agent_action_object": [
    [
      "Спортивную форму другу принесла тренер Марина. Кто принёс спортивную форму?",
      "Тренер Марина принесла другу спортивную форму."
    ],
    [
      "Письмо заведующей передал лаборант Фёдор. Кто передал письмо?",
      "Лаборант Фёдор передал письмо заведующей."
    ]
  ],
  "time_causality": [
    [
      "Прежде чем открыть термос, путешественник снял крышку чехла. Что случилось первым?",
      "Сначала путешественник снял крышку чехла."
    ],
    [
      "Вначале художник загрунтовал холст, после чего начал рисовать. Что было раньше?",
      "Раньше художник загрунтовал чистый холст."
    ]
  ],
  "negation_scope": [
    [
      "Ребёнок не купил карандаш, а выбрал альбом. Что выбрал ребёнок?",
      "Ребёнок выбрал большой бумажный альбом."
    ],
    [
      "Оператор не включил лампу, а запустил вентилятор. Что запустил оператор?",
      "Оператор запустил новый электрический вентилятор."
    ]
  ],
  "arithmetic_units": [
    [
      "На столе лежало пятнадцать ложек, шесть убрали. Сколько осталось? Ответьте предложением.",
      "Осталось девять металлических чайных ложек."
    ],
    [
      "Мастер подготовил семь рам, потом ещё девять. Сколько всего рам? Ответьте предложением.",
      "Мастер подготовил шестнадцать деревянных рам."
    ]
  ],
  "semantic_paraphrase": [
    [
      "Перефразируйте проще: «Архивариус осуществил систематизацию старых документов».",
      "Архивариус упорядочил старые архивные документы."
    ],
    [
      "Скажите проще: «Механик произвёл осмотр тормозной системы».",
      "Механик осмотрел автомобильную тормозную систему."
    ]
  ]
}

def rows(split="DEV"):
    if split!="DEV":
        raise ValueError("Only untouched Z14 synthetic DEV, never TRAIN/J7/EVAL128/GOLD")
    return [
        {"id":f"S119Q2I8Z14_{kind}_{i:02d}","kind":kind,"split":"DEV",
         "origin":ORIGIN,"prompt":prompt,"completion":answer}
        for kind in KINDS for i,(prompt,answer) in enumerate(CASES[kind])
    ]

def digest(entries):
    return hashlib.sha256("".join(
        json.dumps(z,ensure_ascii=False,sort_keys=True,separators=(",",":"))+"\n"
        for z in entries).encode("utf8")).hexdigest()

def verify():
    now=rows()
    old=(e_rows()+g_rows()+k_rows()+r_rows()
         +t_rows("TRAIN")+t_rows("DEV")
         +w_rows("TRAIN")+w_rows("DEV")
         +x_rows("TRAIN")+x_rows("DEV")
         +z2_rows("TRAIN")+z2_rows("DEV")
         +z9_rows("TRAIN")+z9_rows("DEV")+z12_rows("DEV"))
    names={v["prompt"].strip().casefold() for v in now}
    history={v["prompt"].strip().casefold() for v in old}
    if not (
        len(now)==14 and len(names)==14 and names.isdisjoint(history)
        and len({v["id"] for v in now})==14
        and Counter(v["kind"] for v in now)=={kind:2 for kind in KINDS}
        and set(CASES)==set(KINDS)
        and BUDGETS==(8,16,32,64,128,512)
        and all(v["origin"]==ORIGIN and v["completion"].endswith(".")
                and len(v["completion"].split())>=4 and len(v["prompt"])>=22
                for v in now)):
        raise ValueError("Z14 fresh14 source, seven domain2, strict disjoint and budgets")
    return {
        "stage":STAGE,
        "source_sha256":digest(now),
        "new_DEV14":14,
        "budget_k":BUDGETS,
        "all_source_prompts_disjoint_E_G_K_R_T_W_X_Z2_Z9_Z12":True,
        "source_is_internal_synthetic_NOT_blind":True,
        "DEV_becomes_spent_at_first_actual43M_forward":True,
        "no_target_in_candidate_generation":True,
        "no_protected_616J7_EVAL128_GOLD":True,
    }
