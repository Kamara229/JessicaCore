"""Z34 original model native BPE2-only position-gated neural comparator.

Physical Z33 showed ALL 14/14 previous SPENT Z31 DEV own FIRST native
BPE was correct, but native BPE2 correct ONLY 2/14; the first
error thus occurs BEFORE any teacher-versus-own history divergence.
Its actual next native BPE target generally remains in pretrained
native top64, including after wrong OWN histories. Do NOT train
a made-up human-gold suffix on wrong OWN prefix.

Precommit 84 previously legally labelled original-author TRAIN only
Z16/Z18/Z21, plus 14 NEW synthetic Russian author DEV disjoint all
previously SPENT up through Z31. Exact original native49152 top64
hard-negative training after OWN model first BPE, condition on
same-pass frozen last h320 + preceding own original native
embedding320; train small 640x32x320=30720param neural readout,
with native delta ±1.5, original shortlist KL0.08 and hard
greedy-vs-gold margin. Wrong-offset17 label control trained same
budget; no gold injection or EOS masking. At inference activate
only native BPE position2: ORIGINAL native first and all later
steps unchanged, except propagated own prefix from BPE2.
Strict no old DEV optimisation and no 616J7/EVAL128/GOLD,
packed bpp/CPU CB59 claims. Gate0 only; no LIVE.
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
TOP_K=64
RANK=32
HIDDEN_DIM=320
INPUT_DIM=640
TRAIN_EPOCHS=25
LEARNING_RATE=.006
WEIGHT_DECAY=.04
LOGIT_RESIDUAL_BOUND=1.5
ORIGINAL_TOP64_KL_WEIGHT=.08
HARD_NEGATIVE_MARGIN=.25
CORRECT_ORIGINAL_GREEDY_WEIGHT=2.
WRONG_OFFSET17_CONTROL=17
ACTIVATE_SECOND_BPE_ONLY=True
TRAIN_ELIGIBLE_MIN=20
EVAL_FIRST_NATIVE_BPE=6
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
        z28_rows("DEV")+
        z31_rows("DEV"))
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
         and TOP_K==64 and RANK==32
         and HIDDEN_DIM==320 and INPUT_DIM==640
         and TRAIN_EPOCHS==25 and LEARNING_RATE==.006
         and WEIGHT_DECAY==.04
         and LOGIT_RESIDUAL_BOUND==1.5
         and ORIGINAL_TOP64_KL_WEIGHT==.08
         and HARD_NEGATIVE_MARGIN==.25
         and CORRECT_ORIGINAL_GREEDY_WEIGHT==2.
         and WRONG_OFFSET17_CONTROL==17
         and ACTIVATE_SECOND_BPE_ONLY
         and TRAIN_ELIGIBLE_MIN==20
         and EVAL_FIRST_NATIVE_BPE==6):
        raise ValueError("Z34 prereg legal human TRAIN84 and brand new synthetic DEV14, BPE2-only native top64 supervised gated control")
    return {"stage":STAGE,"prior_legal_TRAIN84":84,"brandNEW_author_synthetic_DEV14":14,
            "TRAIN_SHA256":digest(train),"NEW_DEV_SHA256":digest(dev),
            "new_DEV_disjoint_all_prior_spent_including_Z31":True,
            "no_previous_SPENT_DEV_in_TRAIN":True,
            "frozen_before_any_physical_new_DEV_forward":True,
            "fixed_second_native_BPE_only":{
                "native_full49152_candidate_topK":TOP_K,
                "rank":RANK,"hidden":HIDDEN_DIM,
                "input_h320_plus_preceding_gold_or_own_BPE_emb320":INPUT_DIM,
                "train_epochs":TRAIN_EPOCHS,"train_lr":LEARNING_RATE,
                "train_weight_decay":WEIGHT_DECAY,
                "logit_delta_max":LOGIT_RESIDUAL_BOUND,
                "KL(original_top64||candidate)":ORIGINAL_TOP64_KL_WEIGHT,
                "human_gold_vs_original_greedy_margin":HARD_NEGATIVE_MARGIN,
                "protect_original_correct_greedy_weight":CORRECT_ORIGINAL_GREEDY_WEIGHT,
                "wrong_candidate_negative_control_offset":WRONG_OFFSET17_CONTROL,
                "ONLY_BPE2_position_activation":ACTIVATE_SECOND_BPE_ONLY,
                "min_legal_TRAIN_ownfirstcorrect_and_goldinside_top64":TRAIN_ELIGIBLE_MIN,
                "autonomous_DEV_first_native_count":EVAL_FIRST_NATIVE_BPE,
            },
            "new_internal_DEV_not_external_blind_and_spent_on_first_test":True,
            "no_J7_EVAL128_GOLD_or_bpp_CPU_acceptance":True,
            "output_gate":0}
