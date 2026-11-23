"""S119q2i8z34 prereg after Z33 primary native BPE2 first failure.

Older genuine Z29 84 legal TRAIN x 5 positions decreased CE, yet
rank4 contextual down/up caused 2-token mode collapse, not native
output_bias. New experiment freezes bias and trains only 197888
lowrank contextual neural parameters within bounded ±2 pre-gain
residual to frozen Z10 rank4 head, KL(original native logits ||
new logits) 1.0; original native logits reconstructed from SAME paid
S142 native49152 forward, no duplicate 43M pass required.
Original pretrained 43M, GT12, Core/NRA3/NPA1 and original fitted
Z10 all frozen. Only previous legal84 human TRAIN Z16/Z18/Z21.
NEW author14 DEV seven domains x2, disjoint ALL prior old
E/G/K/R/T/W/X/Z2/Z9/Z12/Z14/Z16/Z18/Z21/Z23/Z25/Z28.
Precommitted 2 epochs, lr .004, position CE BPE1-4+EOS and
margins, same as previous. No DEV target in training/inference,
no protected 616J7/EVAL128/GOLD, bpp or CPU CB59 acceptance.
Draft research only, output_gate0, no LIVE.
"""
from __future__ import annotations
import json,hashlib
from collections import Counter
from research.calibri_nra3.s119q2i8e_new_ru_multiword_train_source import KINDS
from research.calibri_nra3.s119q2i8e_new_ru_multiword_train_source import rows as e_rows
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
from research.calibri_nra3.s119q2i8z25_84historicaltrain_new14dev_hardnegative_source import rows as z25_rows
from research.calibri_nra3.s119q2i8z28_precommit_84train14fresh_native_head_multistep import rows as z28_rows
from research.calibri_nra3.s119q2i8z31_84legaltrain_new14dev_bounded_context_source import rows as z31_rows
STAGE="CALIBRI_S119Q2I8Z34_PREREG_PRECISE_OWNFIRST_SECOND_NATIVE_BPE_GATED_RANKER_84TRAIN14NEWDEV"
ORIGIN="author_internal_Z34_new14_20261123_native_BPE2_position_gated"
EPOCHS=2
NATIVE_HEAD_GAIN=.25
TRAIN_LR=.004
TRAIN_WD=.02
NATIVE_REFERENCE_KL=1.0
CONTEXT_PREGAIN_RESIDUAL_BOUND=2.0
ORIGINAL_WEIGHT_L2=.02
TRAINABLE_DOWN_UP=197888
FIXED_VOCAB_OUTPUT_BIAS=49152
EVAL_FIRST_NATIVE_BPE=6
TRAIN_POSITIONS=(0,1,2,3,-1)
TRAIN_POSITION_WEIGHTS=(2.,1.,1.,1.,2.)
TRAIN_REPETITION_MARGIN=.25
TRAIN_REPETITION_WEIGHT=.25
TRAIN_PREMATURE_EOS_WEIGHT=.10
DEV_CASES={
  "sentence_formulation": [
    [
      "Составьте фразу о биохимике, который исследует образец воды.",
      "Биохимик исследует чистый образец речной воды."
    ],
    [
      "Напишите фразу о часовщике, который проверяет механизм часов.",
      "Часовщик проверяет старый механизм настенных часов."
    ]
  ],
  "case_agreement": [
    [
      "Исправьте падеж: «Художник приблизился к светлая витрине».",
      "Художник приблизился к светлой магазинной витрине."
    ],
    [
      "Исправьте согласование: «Турист шёл вдоль каменный стены».",
      "Турист шёл вдоль каменной городской стены."
    ]
  ],
  "agent_action_object": [
    [
      "Словарь ученику передала учительница Елена. Кто передал словарь?",
      "Учительница Елена передала ученику русский словарь."
    ],
    [
      "Велосипед соседу починил мастер Виктор. Кто починил велосипед?",
      "Мастер Виктор починил соседу старый велосипед."
    ]
  ],
  "time_causality": [
    [
      "До начала дождя сторож закрыл ворота. Что произошло раньше?",
      "Сначала сторож закрыл тяжёлые железные ворота."
    ],
    [
      "После прогулки турист почистил ботинки. Что произошло позже?",
      "Позже турист почистил свои дорожные ботинки."
    ]
  ],
  "negation_scope": [
    [
      "Повар не разогрел суп, а испёк хлеб. Что испёк повар?",
      "Повар испёк свежий домашний белый хлеб."
    ],
    [
      "Художница не купила кисть, а выбрала краску. Что выбрала художница?",
      "Художница выбрала яркую синюю акварельную краску."
    ]
  ],
  "arithmetic_units": [
    [
      "В клубе было шестнадцать книг, добавили восемь. Сколько стало? Ответьте предложением.",
      "В клубе стало двадцать четыре новые книги."
    ],
    [
      "На столе стояло тридцать кружек, семь убрали. Сколько осталось? Ответьте предложением.",
      "На столе осталось двадцать три кружки."
    ]
  ],
  "semantic_paraphrase": [
    [
      "Скажите проще: «Садовник осуществил полив молодых деревьев».",
      "Садовник полил молодые деревья чистой водой."
    ],
    [
      "Перефразируйте: «Переводчик осуществил проверку готового текста».",
      "Переводчик проверил готовый текст перед отправкой."
    ]
  ]
}
def rows(split):
    if split=="TRAIN":
        old=z16_rows("TRAIN")+z18_rows("TRAIN")+z21_rows("TRAIN")
        return [{**v,"id":"S119Q2I8Z34_LEGAL_"+v["id"],
                 "split":"TRAIN","origin":"previously_legitimate_TRAIN_only_"+v["origin"]}
                for v in old]
    if split=="DEV":
        return [{"id":f"S119Q2I8Z34_{kind}_{i:02d}","kind":kind,
                 "split":"DEV","origin":ORIGIN,"prompt":prompt,"completion":completion}
                for kind in KINDS for i,(prompt,completion) in enumerate(DEV_CASES[kind])]
    raise ValueError("S119q2i8z31 only prior TRAIN84 and brandNEW synthetic DEV14")
def digest(seq):
    return hashlib.sha256("".join(json.dumps(v,ensure_ascii=False,sort_keys=True,
          separators=(",",":"))+"\n" for v in seq).encode("utf8")).hexdigest()
def verify():
    train,dev=rows("TRAIN"),rows("DEV")
    past=(g_rows()+
        k_rows()+
        r_rows()+
        e_rows()+
        t_rows("TRAIN")+
        t_rows("DEV")+
        w_rows("TRAIN")+
        w_rows("DEV")+
        x_rows("TRAIN")+
        x_rows("DEV")+
        z2_rows("TRAIN")+
        z2_rows("DEV")+
        z9_rows("TRAIN")+
        z9_rows("DEV")+
        z16_rows("TRAIN")+
        z16_rows("DEV")+
        z18_rows("TRAIN")+
        z18_rows("DEV")+
        z21_rows("TRAIN")+
        z21_rows("DEV")+
        z12_rows("DEV")+
        z14_rows("DEV")+
        z23_rows("DEV")+
        z25_rows("DEV")+
        z28_rows("DEV"))
    past_prompts={v["prompt"].casefold().strip() for v in past}
    train_prompts={v["prompt"].casefold().strip() for v in train}
    dev_prompts={v["prompt"].casefold().strip() for v in dev}
    old_dev=(g_rows()+k_rows()+r_rows()+t_rows("DEV")+w_rows("DEV")+
          x_rows("DEV")+z2_rows("DEV")+z9_rows("DEV")+z12_rows("DEV")+
          z14_rows("DEV")+z16_rows("DEV")+z18_rows("DEV")+z21_rows("DEV")+
          z23_rows("DEV")+z25_rows("DEV")+z28_rows("DEV")+z31_rows("DEV"))
    old_dev_prompts={v["prompt"].casefold().strip() for v in old_dev}
    if not (len(train)==84 and len(dev)==14
         and len(train_prompts)==84 and len(dev_prompts)==14
         and len({v["id"] for v in train+dev})==98
         and dev_prompts.isdisjoint(past_prompts)
         and dev_prompts.isdisjoint(train_prompts)
         and train_prompts.isdisjoint(old_dev_prompts)
         and Counter(v["kind"] for v in train)=={k:12 for k in KINDS}
         and Counter(v["kind"] for v in dev)=={k:2 for k in KINDS}
         and all(v["origin"]==ORIGIN and v["completion"].endswith(".")
                 and len(v["completion"].split())>=5
                 and len(v["prompt"])>35 for v in dev)
         and EPOCHS==2 and NATIVE_HEAD_GAIN==.25
         and TRAIN_LR==.004 and TRAIN_WD==.02
         and NATIVE_REFERENCE_KL==1.0
         and CONTEXT_PREGAIN_RESIDUAL_BOUND==2.0
         and ORIGINAL_WEIGHT_L2==.02
         and TRAINABLE_DOWN_UP==197888
         and FIXED_VOCAB_OUTPUT_BIAS==49152
         and EVAL_FIRST_NATIVE_BPE==6
         and TRAIN_POSITIONS==(0,1,2,3,-1)
         and TRAIN_POSITION_WEIGHTS==(2.,1.,1.,1.,2.)
         and TRAIN_REPETITION_MARGIN==.25
         and TRAIN_REPETITION_WEIGHT==.25
         and TRAIN_PREMATURE_EOS_WEIGHT==.10):
        raise ValueError("Z31 frozen human original 84 previous legal TRAIN and new14 DEV / trust region constants failed")
    return {"stage":STAGE,"prior_legal_TRAIN84":84,"brandNEW_author_synthetic_DEV14":14,
            "TRAIN_SHA256":digest(train),"NEW_DEV_SHA256":digest(dev),
            "new_DEV_disjoint_all_prior_spent_including_Z31":True,
            "no_previous_SPENT_DEV_in_TRAIN":True,
            "frozen_before_any_physical_new_DEV_forward":True,
            "fixed_contextual_rank4_trust":{"epochs":EPOCHS,"lr":TRAIN_LR,
              "wd":TRAIN_WD,"gain":NATIVE_HEAD_GAIN,
              "reference_fullnative49152_KL":NATIVE_REFERENCE_KL,
              "bound_pregain":CONTEXT_PREGAIN_RESIDUAL_BOUND,
              "original_down_up_weight_L2":ORIGINAL_WEIGHT_L2,
              "down_up_trainable_params":TRAINABLE_DOWN_UP,
              "frozen_global_vocab_bias_params":FIXED_VOCAB_OUTPUT_BIAS,
              "positions":TRAIN_POSITIONS,"weights":TRAIN_POSITION_WEIGHTS,
              "first_autonomous_DEV_native_BPE":EVAL_FIRST_NATIVE_BPE},
            "new_internal_DEV_not_external_blind_and_spent_on_first_test":True,
            "no_J7_EVAL128_GOLD_or_bpp_CPU_acceptance":True,
            "output_gate":0}
