"""S119q2i8z28 precommit long-needed actual original native BPE1-4+EOS training.

Previous actual Z27 source-only root cause: ORIGINAL bare pretrained
S142 43M prematurely EOS14/14. Full fitted Core/NRA3/NPA1+GT12
with head gain0 stutters BPE2=BPE3=BPE4 11/14. Gain .25 protects
first BPE13/14 but still stutters9/14. Gain1 premature EOS14/14.
Neither native64 hard-negative reranker nor future h320 state fixes.

Change TRAIN objective (not DEV token tricks): from short one-token
shadow to TRAIN-ONLY full native49152 teacher-forced true human
BPE positions 1,2,3,4 and real EOS from same 84 previously legal
historical TRAIN-only Z16/Z18/Z21. Optimize ONLY a COPIED prior
original Z10 rank4 native language head (247040 parameters), not
S142 pretrained, GT12, Core/NRA3/NPA1 or old trained A/B.
Fixed two epochs, per-example teacher-forced same 5 native steps,
first BPE/EOS weighted×2, native full49152 CE plus BPE-repeat and
early EOS margins as causal differentiable TRAIN-only losses.
No GOLD beyond previous legitimate TRAIN. Inference no masking
and no hand-coded output tokens. Real new 14 original author DEV
7 domains x2, disjoint EVERY prior SPENT DEV incl Z25 and Z23,
first6 autonomous native BPE, direct baseline Z10 vs new head.

DEV becomes spent after first score, not independent blind.
No original protected 616 J7/EVAL128/GOLD or bpp/CPU claim.
No main, LIVE, Render, Supabase, APK, Jessica; gate0.
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
from research.calibri_nra3.s119q2i8z25_84historicaltrain_new14dev_hardnegative_source import rows as z25_rows

STAGE="CALIBRI_S119Q2I8Z28_PREREG_84LEGALTRAIN_14NEWDEV_FULL_NATIVE_BPE1_4_EOS_HEADONLY"
ORIGIN="original_author_Z28_20261123_new14dev_native_multiposition"
TRAIN_EPOCHS=2
HEAD_LR=.006
HEAD_WEIGHT_DECAY=.02
HEAD_GAIN=.25
TRAIN_SUPERVISED_POSITIONS=(0,1,2,3,-1)
TRAIN_WEIGHTS=(2.,1.,1.,1.,2.)
TRAIN_REPETITION_MARGIN=.25
TRAIN_REPETITION_LOSS=.25
TRAIN_PREMATURE_EOS_LOSS=.10
EVAL_NATIVE_FIRST_K=6
BASE_S142_FROZEN=True
BASE_FITTED_Z10_FP16_FROZEN=True
DEV_CASES={
  "sentence_formulation": [
    [
      "Составьте предложение о вулканологе, который изучает застывшую лаву.",
      "Вулканолог внимательно изучает застывшую лаву древнего вулкана."
    ],
    [
      "Напишите предложение о лаборанте, который проверяет образцы речного песка.",
      "Лаборант проверяет свежие образцы мелкого речного песка."
    ]
  ],
  "case_agreement": [
    [
      "Исправьте падеж: «Путник двигался по каменистый дороге».",
      "Путник двигался по каменистой горной дороге."
    ],
    [
      "Исправьте грамматику: «Ребёнок спрятался за большая скамейкой».",
      "Ребёнок спрятался за большой деревянной скамейкой."
    ]
  ],
  "agent_action_object": [
    [
      "Морской атлас капитану отдал штурман Николай. Кто отдал атлас?",
      "Штурман Николай отдал капитану морской атлас."
    ],
    [
      "Новый шлем гонщику принесла механик Арина. Кто принёс шлем?",
      "Механик Арина принесла гонщику защитный шлем."
    ]
  ],
  "time_causality": [
    [
      "Прежде чем включить проектор, лектор закрыл шторы. Что было раньше?",
      "Сначала лектор закрыл плотные оконные шторы."
    ],
    [
      "После полива грядок фермер собрал инструменты. Что произошло позже?",
      "Позже фермер собрал оставленные садовые инструменты."
    ]
  ],
  "negation_scope": [
    [
      "Биолог не нашёл гнездо, а обнаружил перо. Что обнаружил биолог?",
      "Биолог обнаружил необычное птичье перо."
    ],
    [
      "Водитель не починил фару, а заменил зеркало. Что заменил водитель?",
      "Водитель заменил повреждённое боковое зеркало."
    ]
  ],
  "arithmetic_units": [
    [
      "В коробке было двадцать пять карандашей, восемь достали. Сколько осталось? Ответьте предложением.",
      "В коробке осталось семнадцать цветных карандашей."
    ],
    [
      "Художник купил четырнадцать кисточек и ещё шесть. Сколько всего? Ответьте предложением.",
      "Художник купил двадцать новых кисточек."
    ]
  ],
  "semantic_paraphrase": [
    [
      "Скажите проще: «Зоолог осуществил обследование места обитания зверя».",
      "Зоолог обследовал природное место обитания зверя."
    ],
    [
      "Перефразируйте: «Секретарь произвёл регистрацию входящего письма».",
      "Секретарь зарегистрировал новое входящее письмо."
    ]
  ]
}
def rows(split):
    if split=="TRAIN":
        previous=z16_rows("TRAIN")+z18_rows("TRAIN")+z21_rows("TRAIN")
        return [{**r,"split":"TRAIN","id":"S119Q2I8Z28_LEGAL_"+r["id"],
          "origin":"prior_legitimate_TRAIN_only_"+r["origin"]}
          for r in previous]
    if split=="DEV":
        return [{"id":f"S119Q2I8Z28_{kind}_{i:02d}","kind":kind,
            "split":"DEV","origin":ORIGIN,"prompt":p,"completion":a}
            for kind in KINDS for i,(p,a) in enumerate(DEV_CASES[kind])]
    raise ValueError("S119q2i8z28 only TRAIN84/DEV14; never J7/EVAL128/GOLD")

def digest(src):
    return hashlib.sha256("".join(
        json.dumps(r,ensure_ascii=False,sort_keys=True,separators=(",",":"))+"\n"
        for r in src).encode("utf8")).hexdigest()

def verify():
    tr,dev=rows("TRAIN"),rows("DEV")
    prev_all=(e_rows()+g_rows()+k_rows()+r_rows()
       +t_rows("TRAIN")+t_rows("DEV")
       +w_rows("TRAIN")+w_rows("DEV")
       +x_rows("TRAIN")+x_rows("DEV")
       +z2_rows("TRAIN")+z2_rows("DEV")
       +z9_rows("TRAIN")+z9_rows("DEV")
       +z12_rows("DEV")+z14_rows("DEV")
       +z16_rows("TRAIN")+z16_rows("DEV")
       +z18_rows("TRAIN")+z18_rows("DEV")
       +z21_rows("TRAIN")+z21_rows("DEV")
       +z23_rows("DEV")+z25_rows("DEV"))
    old={x["prompt"].strip().casefold() for x in prev_all}
    future={x["prompt"].strip().casefold() for x in dev}
    train={x["prompt"].strip().casefold() for x in tr}
    earlier_dev={x["prompt"].strip().casefold() for x in (
        g_rows()+k_rows()+r_rows()
        +t_rows("DEV")+w_rows("DEV")+x_rows("DEV")
        +z2_rows("DEV")+z9_rows("DEV")
        +z12_rows("DEV")+z14_rows("DEV")
        +z16_rows("DEV")+z18_rows("DEV")
        +z21_rows("DEV")+z23_rows("DEV")+z25_rows("DEV"))}
    if not (
        len(tr)==84 and len(dev)==14 and len(train)==84
        and len(future)==14 and future.isdisjoint(old)
        and train.isdisjoint(earlier_dev)
        and len({x["id"] for x in tr+dev})==98
        and set(DEV_CASES)==set(KINDS)
        and Counter(x["kind"] for x in tr)=={k:12 for k in KINDS}
        and Counter(x["kind"] for x in dev)=={k:2 for k in KINDS}
        and all(x["origin"]==ORIGIN and x["completion"].endswith(".")
                and len(x["prompt"])>35
                and len(x["completion"].split())>=5 for x in dev)
        and (TRAIN_EPOCHS,HEAD_LR,HEAD_WEIGHT_DECAY,HEAD_GAIN,
             TRAIN_SUPERVISED_POSITIONS,TRAIN_WEIGHTS,
             TRAIN_REPETITION_MARGIN,TRAIN_REPETITION_LOSS,
             TRAIN_PREMATURE_EOS_LOSS,EVAL_NATIVE_FIRST_K)==
            (2,.006,.02,.25,(0,1,2,3,-1),(2.,1.,1.,1.,2.),.25,.25,.10,6)
        and BASE_S142_FROZEN and BASE_FITTED_Z10_FP16_FROZEN):
        raise ValueError("Z28 fresh original author14 DEV and previous legal84 TRAIN, fixed multiposition native training")
    return {
        "stage":STAGE,"prior_legal_TRAIN_count":84,
        "NEW_author_synthetic_DEV_count":14,
        "TRAIN_SHA256":digest(tr),"DEV_SHA256":digest(dev),
        "seven_domains_train12_dev2":True,
        "strict_no_previous_SPENT_DEV_in_training":True,
        "NEW_DEV_disjoint_all_old_including_Z25":True,
        "fixed_training":{
            "train_epochs":TRAIN_EPOCHS,"head_only_lr":HEAD_LR,
            "head_only_weight_decay":HEAD_WEIGHT_DECAY,
            "native_head_gain":HEAD_GAIN,
            "supervised_positions_including_EOS":TRAIN_SUPERVISED_POSITIONS,
            "position_weights":TRAIN_WEIGHTS,
            "native_selfrepeat_margin":TRAIN_REPETITION_MARGIN,
            "native_selfrepeat_loss_weight":TRAIN_REPETITION_LOSS,
            "premature_EOS_loss_weight":TRAIN_PREMATURE_EOS_LOSS,
            "free_autoregressive_native_dev_first_K":EVAL_NATIVE_FIRST_K},
        "original_S142_and_GT_Core_NRA3_NPA1_frozen":True,
        "previous_Z10_A_B_fitted_head_immutable":True,
        "internal_synthetic_DEV_becomes_SPENT_on_first_test":True,
        "no_616J7_EVAL128_GOLD_bpp_or_CPU_acceptance":True,
    }
