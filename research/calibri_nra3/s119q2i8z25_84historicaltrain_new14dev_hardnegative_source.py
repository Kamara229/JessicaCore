"""S119q2i8z25 prereg native49152 hard-negative candidate ranking.

Reuse ONLY already-labelled 84 legitimate historical TRAIN examples
Z16, Z18, Z21 (all 28 TRAIN each), NEVER their SPENT DEV.
Add brand NEW 14 internal author Russian DEV = seven domains x2.
Strict DEV disjoint all past E/G/K/R/T/W/X/Z2/Z9/Z12/Z14/Z16/Z18/
Z21/Z23, train vs dev, protected J7-616/EVAL128/GOLD.

Fixed hypothesis: previous future-embedding proxy failed to rank true
gold second BPE above greedy, even when gold inside native top64.
Train a compact candidate-specific lowrank16 neural 320x16x320
ranker with original native49152 model top64 and gold second BPE
only if gold is in model's OWN-first candidate top64 on TRAIN.
Compare fixed shuffled candidate-target negative control.
Protect original logits with KL anchor and residual bound ±3; never
insert gold, never mask EOS, no original pretrained/GT/Core/NRA3/NPA1
updates. Score fresh DEV first4 native BPE auto-regressive by ACTUAL
original43M forwards without teacher forcing, compare original greedy.
DEV becomes SPENT on first evaluation. No public/blind quality claim.
"""
from __future__ import annotations
import hashlib,json
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
from research.calibri_nra3.s119q2i8z21_fresh28train14dev_future_state_source import rows as z21_rows
from research.calibri_nra3.s119q2i8z23_new28dev_frozen_future_calibration_source import rows as z23_rows

STAGE="CALIBRI_S119Q2I8Z25_PREREG_84_PREVIOUS_TRAIN_14_NEW_DEV_HARDNEG_NATIVE64"
ORIGIN="original_author_Z25_20261123_native_hardnegative_new14DEV"
TRAIN_SOURCES=("S119Q2I8Z16_","S119Q2I8Z18_","S119Q2I8Z21_")
TOP_K=64
RANK=16
RESIDUAL_BOUND=3.0
TRAIN_EPOCHS=30
TRAIN_ADAMW_LR=.008
TRAIN_ADAMW_WD=.05
TRAIN_KL_ANCHOR=.2
TRAIN_RESIDUAL_L2=.04
NEGATIVE_TARGET_OFFSET=13
ANSWER_EVAL_FIRST_NATIVE_BPE_COUNT=4
DEV_CASES={
  "sentence_formulation": [
    [
      "Напишите предложение о ботанике, который фотографирует редкую степную траву.",
      "Ботаник фотографирует редкую степную траву на склоне."
    ],
    [
      "Составьте фразу о судостроителе, который проверяет корпус парусной лодки.",
      "Судостроитель проверяет прочный корпус новой парусной лодки."
    ]
  ],
  "case_agreement": [
    [
      "Исправьте согласование: «Покупатель подошёл к деревянный прилавку».",
      "Покупатель подошёл к деревянному торговому прилавку."
    ],
    [
      "Исправьте ошибку: «Оператор любовался яркая зарёй».",
      "Оператор любовался яркой утренней зарёй."
    ]
  ],
  "agent_action_object": [
    [
      "Тетрадь преподавателю отдала ученица Варвара. Кто отдал тетрадь?",
      "Ученица Варвара отдала преподавателю школьную тетрадь."
    ],
    [
      "Корзину повару принёс садовник Тимофей. Кто принёс корзину?",
      "Садовник Тимофей принёс повару плетёную корзину."
    ]
  ],
  "time_causality": [
    [
      "До начала ремонта механик выключил питание. Что произошло сначала?",
      "Сначала механик выключил электрическое питание станка."
    ],
    [
      "После уборки комнаты хозяин открыл окно. Что произошло позже?",
      "Позже хозяин открыл большое комнатное окно."
    ]
  ],
  "negation_scope": [
    [
      "Писатель не взял карандаш, а открыл блокнот. Что открыл писатель?",
      "Писатель открыл небольшой карманный блокнот."
    ],
    [
      "Фермер не продал овцу, а обменял козу. Что обменял фермер?",
      "Фермер обменял молодую домашнюю козу."
    ]
  ],
  "arithmetic_units": [
    [
      "На складе лежало тридцать две коробки, девять отправили. Сколько осталось? Ответьте предложением.",
      "На складе осталось двадцать три картонные коробки."
    ],
    [
      "В поезд сели восемнадцать пассажиров, затем ещё шесть. Сколько всего? Ответьте предложением.",
      "В поезд сели двадцать четыре новых пассажира."
    ]
  ],
  "semantic_paraphrase": [
    [
      "Скажите проще: «Эксперт осуществил исследование качества металла».",
      "Эксперт исследовал качество нового металлического сплава."
    ],
    [
      "Перефразируйте: «Тренер произвёл корректировку тренировочного расписания».",
      "Тренер исправил расписание предстоящих спортивных тренировок."
    ]
  ]
}

def rows(split):
    if split=="TRAIN":
        old=z16_rows("TRAIN")+z18_rows("TRAIN")+z21_rows("TRAIN")
        return [{**r,"split":"TRAIN",
                 "id":"S119Q2I8Z25_TRONLY_"+r["id"],
                 "origin":"historical_previously_legitimate_TRAIN_only_"+r["origin"]}
                 for r in old]
    if split=="DEV":
        return [{
            "id":f"S119Q2I8Z25_{kind}_{i:02d}",
            "kind":kind,"split":"DEV","origin":ORIGIN,
            "prompt":p,"completion":a,
        } for kind in KINDS for i,(p,a) in enumerate(DEV_CASES[kind])]
    raise ValueError("Z25 strict 84 historical TRAIN /14 NEW DEV only; no J7/GOLD/EVAL")

def digest(v):
    return hashlib.sha256("".join(
       json.dumps(r,ensure_ascii=False,sort_keys=True,separators=(",",":"))+"\n"
       for r in v).encode("utf8")).hexdigest()

def verify():
    train,dev=rows("TRAIN"),rows("DEV")
    previous_DEV=(g_rows()+k_rows()+r_rows()
        +t_rows("DEV")+w_rows("DEV")+x_rows("DEV")
        +z2_rows("DEV")+z9_rows("DEV")+z12_rows("DEV")
        +z14_rows("DEV")+z16_rows("DEV")+z18_rows("DEV")
        +z21_rows("DEV")+z23_rows("DEV"))
    previous_all=(previous_DEV+e_rows()+t_rows("TRAIN")+w_rows("TRAIN")+x_rows("TRAIN")
        +z2_rows("TRAIN")+z9_rows("TRAIN")+z16_rows("TRAIN")+z18_rows("TRAIN")+z21_rows("TRAIN"))
    old_prompts={x["prompt"].casefold().strip() for x in previous_all}
    dev_prompts={x["prompt"].casefold().strip() for x in dev}
    train_prompts={x["prompt"].casefold().strip() for x in train}
    if not (len(train)==84 and len(dev)==14
        and len({x["id"] for x in train+dev})==98
        and len(train_prompts)==84 and len(dev_prompts)==14
        and dev_prompts.isdisjoint(old_prompts)
        and dev_prompts.isdisjoint(train_prompts)
        and train_prompts.isdisjoint({x["prompt"].casefold().strip() for x in previous_DEV})
        and set(DEV_CASES)==set(KINDS)
        and Counter(x["kind"] for x in train)=={k:12 for k in KINDS}
        and Counter(x["kind"] for x in dev)=={k:2 for k in KINDS}
        and all(any(r["id"].startswith("S119Q2I8Z25_TRONLY_"+k)
                    for k in TRAIN_SOURCES) for r in train)
        and all(x["origin"]==ORIGIN and x["completion"].endswith(".")
                and len(x["completion"].split())>=5
                and len(x["prompt"])>34 for x in dev)
        and (TOP_K,RANK,RESIDUAL_BOUND,TRAIN_EPOCHS,
             TRAIN_ADAMW_LR,TRAIN_ADAMW_WD,TRAIN_KL_ANCHOR,
             TRAIN_RESIDUAL_L2,NEGATIVE_TARGET_OFFSET,
             ANSWER_EVAL_FIRST_NATIVE_BPE_COUNT)==
            (64,16,3.0,30,.008,.05,.2,.04,13,4)):
        raise ValueError("Z25 frozen source: only historical legitimate 84 TRAIN +14 genuinely unseen synthetic DEV")
    return {
        "stage":STAGE,
        "historical_previously_consumed_TRAIN_count":84,
        "brand_new_author_synthetic_DEV_count":14,
        "TRAIN_SHA256":digest(train),"DEV_SHA256":digest(dev),
        "seven_domains_12_TRAIN_2_DEV":True,
        "no_SPENT_DEV_reused_as_training":True,
        "new_DEV_disjoint_all_E_G_K_R_T_W_X_Z2_Z9_Z12_Z14_Z16_Z18_Z21_Z23":True,
        "params":{"candidate_topK":TOP_K,"rank":RANK,
                  "residual_bound":RESIDUAL_BOUND,
                  "train_epochs":TRAIN_EPOCHS,"train_lr":TRAIN_ADAMW_LR,
                  "train_weight_decay":TRAIN_ADAMW_WD,
                  "KL_anchor":TRAIN_KL_ANCHOR,
                  "residual_L2":TRAIN_RESIDUAL_L2,
                  "train_control_wrong_candidate_offset":NEGATIVE_TARGET_OFFSET,
                  "first_auto_native_BPE_eval_count":ANSWER_EVAL_FIRST_NATIVE_BPE_COUNT},
        "new_DEV_is_author_synthetic_not_independent_blind":True,
        "no_616J7_EVAL128_GOLD_in_training":True,
        "output_gate":0,
    }
