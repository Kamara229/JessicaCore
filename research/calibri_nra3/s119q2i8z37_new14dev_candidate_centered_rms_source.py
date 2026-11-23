"""Z37: prereg genuinely NEW author14 DEV and a BPE2-only
candidate-contrast mechanism after completed physical Z36.

Z36 original S142 43M spent Z34 proved the native BPE2 human gold
candidate is inside target-blind original TOP64 for 8/13 cases
whose OWN first is correct. Greedy minus gold gap ~0.87 logits;
ALL trained Z35 rank32 (true human and WRONG offset17) produced
gold-minus-greedy relative learned residual 0.0, even with
absolute tanh residual at hard bound 1.5. This suggests saturated
non-discriminative candidate logits, not unfixable 3.0 margin.

Precommit the ONLY change to old 30720param rank32 head:
center per-slate raw candidate inner-product logits BEFORE tanh,
divide by max(per-slate centered RMS, .10), then tanh(.5*z)
and native residual bound +/-1.5. No per-run tuning, no gold
in model inference and no changed original 43M/GT12/learned Z10.
Train only previous legal84 human authored TRAIN Z16/Z18/Z21
and WRONG-offset17 matched control, same 25 epochs LR=.006
and KL top64. NEW authored DEV14 seven categories x2 disjoint
all earlier SPENT through Z34. Exact first6 autonomous model
on fresh DEV must compare baseline vs correct-trained vs
wrong-label control; no promotion without strong evidence.
NO 616J7/EVAL128/GOLD, bpp or CB59 CPU acceptance, gate0.
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
from research.calibri_nra3.s119q2i8z34_fresh14dev_BPE2_only_source import rows as z34_rows
STAGE="CALIBRI_S119Q2I8Z37_PREREG_CENTERED_RMS_NATIVE_BPE2_ONLY_84LEGALTRAIN14NEWDEV"
ORIGIN="author_internal_Z37_new14_20261123_candidate_RMS_nonuniform_native_BPE2"
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
CENTERED_CANDIDATE_RMS_FLOOR=.10
CENTERED_PRE_TANH_DIVISOR=2.0
SAME_30720PARAMS_AS_Z35=True
ACTIVATE_SECOND_BPE_ONLY=True
TRAIN_ELIGIBLE_MIN=20
EVAL_FIRST_NATIVE_BPE=6
DEV_CASES={
  "sentence_formulation": [
    [
      "Составьте предложение о метеорологе, который проверяет осадки.",
      "Метеоролог проверяет количество осадков ранним утром."
    ],
    [
      "Напишите предложение о керамисте, который обжигает чашки.",
      "Керамист обжигает новые глиняные чашки в печи."
    ]
  ],
  "case_agreement": [
    [
      "Исправьте падеж: «Письмо лежало возле высокий шкафа».",
      "Письмо лежало возле высокого деревянного шкафа."
    ],
    [
      "Исправьте согласование: «Лодочник подошёл к тихая пристани».",
      "Лодочник подошёл к тихой речной пристани."
    ]
  ],
  "agent_action_object": [
    [
      "Ножницы портному передал помощник Максим. Кто передал ножницы?",
      "Помощник Максим передал портному острые ножницы."
    ],
    [
      "Чертёж инженеру принесла студентка Мария. Кто принёс чертёж?",
      "Студентка Мария принесла инженеру готовый чертёж."
    ]
  ],
  "time_causality": [
    [
      "Перед запуском лифта техник проверил кнопку. Что было раньше?",
      "Сначала техник проверил исправность кнопки лифта."
    ],
    [
      "После уборки садовник закрыл сарай. Что было позже?",
      "Позже садовник закрыл деревянный садовый сарай."
    ]
  ],
  "negation_scope": [
    [
      "Фотограф не взял штатив, а настроил вспышку. Что настроил фотограф?",
      "Фотограф настроил переносную мощную вспышку."
    ],
    [
      "Строитель не поднял доску, а перенёс ящик. Что перенёс строитель?",
      "Строитель перенёс тяжёлый ящик с инструментами."
    ]
  ],
  "arithmetic_units": [
    [
      "У музыканта было двадцать билетов, девять он отдал. Сколько осталось? Ответьте предложением.",
      "У музыканта осталось одиннадцать свободных билетов."
    ],
    [
      "На складе было двенадцать коробок, прибыло ещё пять. Сколько стало? Ответьте предложением.",
      "На складе стало семнадцать новых коробок."
    ]
  ],
  "semantic_paraphrase": [
    [
      "Скажите проще: «Экспедитор осуществил проверку комплектации груза».",
      "Экспедитор проверил комплектность доставленного груза."
    ],
    [
      "Перефразируйте: «Сотрудник произвёл осмотр входной двери».",
      "Сотрудник осмотрел большую входную дверь."
    ]
  ]
}
def rows(split):
    if split=="TRAIN":
        old=z16_rows("TRAIN")+z18_rows("TRAIN")+z21_rows("TRAIN")
        return [{**v,"id":"S119Q2I8Z37_LEGAL_"+v["id"],
                 "split":"TRAIN","origin":"previously_legitimate_TRAIN_only_"+v["origin"]}
                for v in old]
    if split=="DEV":
        return [{"id":f"S119Q2I8Z37_{kind}_{i:02d}","kind":kind,
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
        z31_rows("DEV")+z34_rows("DEV"))
    past_prompts={v["prompt"].casefold().strip() for v in past}
    train_prompts={v["prompt"].casefold().strip() for v in train}
    dev_prompts={v["prompt"].casefold().strip() for v in dev}
    old_dev=(g_rows()+k_rows()+r_rows()+t_rows("DEV")+w_rows("DEV")+
          x_rows("DEV")+z2_rows("DEV")+z9_rows("DEV")+z12_rows("DEV")+
          z14_rows("DEV")+z16_rows("DEV")+z18_rows("DEV")+z21_rows("DEV")+
          z23_rows("DEV")+z25_rows("DEV")+z28_rows("DEV")+z31_rows("DEV")+z34_rows("DEV"))
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
         and CENTERED_CANDIDATE_RMS_FLOOR==.10
         and CENTERED_PRE_TANH_DIVISOR==2.0
         and SAME_30720PARAMS_AS_Z35
         and ACTIVATE_SECOND_BPE_ONLY
         and TRAIN_ELIGIBLE_MIN==20
         and EVAL_FIRST_NATIVE_BPE==6):
        raise ValueError("Z37 frozen human TRAIN84+new14 author DEV, centered candidate contrast RMS normalization and BPE2 only")
    return {"stage":STAGE,"prior_legal_TRAIN84":84,"brandNEW_author_synthetic_DEV14":14,
            "TRAIN_SHA256":digest(train),"NEW_DEV_SHA256":digest(dev),
            "new_DEV_disjoint_all_prior_spent_including_Z34":True,
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
                "candidate_raw_logits_mean_center_before_tanh":True,
                "candidate_raw_logits_RMS_scale_floor":CENTERED_CANDIDATE_RMS_FLOOR,
                "pre_tanh_divisor":CENTERED_PRE_TANH_DIVISOR,
                "contrast_residual_zero_center_by_construction":True,
                "ONLY_BPE2_position_activation":ACTIVATE_SECOND_BPE_ONLY,
                "min_legal_TRAIN_ownfirstcorrect_and_goldinside_top64":TRAIN_ELIGIBLE_MIN,
                "autonomous_DEV_first_native_count":EVAL_FIRST_NATIVE_BPE,
            },
            "new_internal_DEV_not_external_blind_and_spent_on_first_test":True,
            "no_J7_EVAL128_GOLD_or_bpp_CPU_acceptance":True,
            "output_gate":0}
