"""Z40 SOURCE-ONLY preregistration: native BPE2–4 causal rank32, entropy guard.

Frozen original S142 43M / all12 GT / Core+NRA3+NPA1 Z10 parent A;
research-only separate position-aware shadow, NO original changes.
Physical follow-up uses previous legitimate 84 TRAIN examples only.
Its teacher-gold prefixes are allowed for TRAIN, but inference MUST use
actual own generated prefixes, never teacher DEV forcing. Each native
candidate slate comes from ORIGINAL top64 without gold insertion.
Compare baseline vs true-label vs shifted WRONG17 label control on the
14 entirely new authored DEV prompts (seven domains x two); exactly
one scoring pass, then DEV is spent. No promotion, 616J7/GOLD/EVAL128
or bpp/CPU proof. All hyperparameters fixed before any DEV evaluation.
"""
from __future__ import annotations
import hashlib
import json
from collections import Counter
from research.calibri_nra3.s119q2i8e_new_ru_multiword_train_source import KINDS
from research.calibri_nra3 import s119q2i8z37_new14dev_candidate_centered_rms_source as z37

STAGE="CALIBRI_S119Q2I8Z40_PRECOMMITTED_CAUSAL_BPE2_3_4_ENTROPY_GUARD_84_TRAIN_14_FRESH_DEV"
ORIGIN="author_internal_Z40_brandNEW_20261124_sequence_entropy_multitoken"
TOP_K=64
RANK=32
HIDDEN_DIM=320
INPUT_DIM=640
SHADOW_PARAMS=INPUT_DIM*RANK+RANK*HIDDEN_DIM+3*HIDDEN_DIM  # 31,680
POSITIONS=(2,3,4)
TRAIN_EPOCHS=16
LEARNING_RATE=.004
WEIGHT_DECAY=.04
LOGIT_RESIDUAL_BOUND=1.5
KL_TO_ORIGINAL_TOP64=.30
ENTROPY_FLOOR_TOLERANCE_BITS=.05
ENTROPY_COLLAPSE_PENALTY=0.75
HARD_NEGATIVE_MARGIN=.25
HARD_NEGATIVE_WEIGHT=.25
PER_POSITION_LOSS_WEIGHT={2:1.,3:1.5,4:1.5}
WRONG_OFFSET17_CONTROL=17
CENTERED_RMS_FLOOR=.10
CENTERED_DIVISOR=2.
MIN_ELIGIBLE_TRAIN_PER_POSITION=6
EVAL_OWN_NATIVE_TOKENS=6
ALLOW_ORIGINAL_FIRST_BPE_MODIFICATION=False
ALLOW_ORIGINAL_GT_AND_CORE_OPTIMIZER=False
# These are R&D diagnostic gates, not validation/production certification.
MIN_DISTINCT_BPE2_FRACTION_VS_ORIGINAL=.75
MIN_FIRST3_PAIRED_NEW_CORRECT=2
REQUIRE_FIRST3_GAIN_OVER_WRONG_CONTROL=True

DEV_CASES={
 "sentence_formulation":[
  ("Сформулируйте полное предложение о реставраторе, который чистит старую раму.",
   "Реставратор аккуратно очистил старую деревянную раму."),
  ("Напишите развёрнутое предложение о геодезисте, измеряющем участок для дороги.",
   "Геодезист точно измерил участок перед строительством дороги.")],
 "case_agreement":[
  ("Исправьте ошибки согласования: «Повар посыпал суп свежий зеленью».",
   "Повар посыпал суп свежей рубленой зеленью."),
  ("Исправьте падежную ошибку: «Мы остановились перед высокий башней».",
   "Мы остановились перед высокой каменной башней.")],
 "agent_action_object":[
  ("Архивариусу журнал передал курьер Денис. Кто передал журнал и кому?",
   "Курьер Денис передал архивариусу старый журнал."),
  ("Радиоприёмник мастеру принесла соседка Анна. Кто совершил действие?",
   "Соседка Анна принесла мастеру сломанный радиоприёмник.")],
 "time_causality":[
  ("Прежде чем открыть шлюз, дежурный осмотрел клапан. Что произошло раньше?",
   "Сначала дежурный осмотрел исправный клапан шлюза."),
  ("После измерения давления врач записал показания. Какое действие произошло позже?",
   "Позже врач записал полученные показания давления.")],
 "negation_scope":[
  ("Художник не смешивал краски, а промывал кисточки. Что делал художник?",
   "Художник промывал использованные кисточки чистой водой."),
  ("Механик не подтягивал гайку, а проверял тормоз. Что проверял механик?",
   "Механик проверял исправность заднего тормоза автомобиля.")],
 "arithmetic_units":[
  ("В коробке было двадцать четыре карандаша, семь использовали. Сколько осталось? Дайте ответ предложением.",
   "В коробке осталось семнадцать цветных карандашей."),
  ("На стоянке стояло восемнадцать велосипедов, приехало ещё четыре. Сколько велосипедов теперь? Ответьте предложением.",
   "На стоянке теперь двадцать два велосипеда.")],
 "semantic_paraphrase":[
  ("Перефразируйте проще: «Смотритель осуществил проверку замкового механизма».",
   "Смотритель проверил исправность дверного замка."),
  ("Выразите проще: «Оператор выполнил контроль работоспособности насоса».",
   "Оператор проверил работу водяного насоса.")],
}

def rows(split):
    if split=="TRAIN":
        # No newly authored TRAIN masquerades as a historical human example.
        return [{**row,"id":"S119Q2I8Z40_LEGAL_"+row["id"],
                 "split":"TRAIN", "origin":"z40_retag_previous_legal_"+row["origin"]}
                for row in z37.rows("TRAIN")]
    if split=="DEV":
        return [{"id":f"S119Q2I8Z40_{kind}_{i:02d}", "kind":kind,
                 "split":"DEV","origin":ORIGIN,"prompt":prompt,"completion":completion}
                for kind in KINDS for i,(prompt,completion) in enumerate(DEV_CASES[kind])]
    raise ValueError("Z40 only TRAIN/DEV; no GOLD or EVAL128")

def digest(values):
    return hashlib.sha256("".join(json.dumps(
        row,sort_keys=True,ensure_ascii=False,separators=(",",":"))+"\n"
        for row in values).encode("utf-8")).hexdigest()

def previous_spent_and_train_prompts():
    # All previously consumed authors, plus immediately previous Z37 SPENT.
    return (
        z37.g_rows()+z37.k_rows()+z37.r_rows()+z37.e_rows()+
        z37.t_rows("TRAIN")+z37.t_rows("DEV")+
        z37.w_rows("TRAIN")+z37.w_rows("DEV")+
        z37.x_rows("TRAIN")+z37.x_rows("DEV")+
        z37.z2_rows("TRAIN")+z37.z2_rows("DEV")+
        z37.z9_rows("TRAIN")+z37.z9_rows("DEV")+
        z37.z12_rows("DEV")+z37.z14_rows("DEV")+
        z37.z16_rows("TRAIN")+z37.z16_rows("DEV")+
        z37.z18_rows("TRAIN")+z37.z18_rows("DEV")+
        z37.z21_rows("TRAIN")+z37.z21_rows("DEV")+
        z37.z23_rows("DEV")+z37.z25_rows("DEV")+
        z37.z28_rows("DEV")+z37.z31_rows("DEV")+
        z37.z34_rows("DEV")+z37.rows("DEV")
    )

def verify():
    prev = z37.verify()
    train,dev=rows("TRAIN"),rows("DEV")
    old=previous_spent_and_train_prompts()
    prompt=lambda row: row["prompt"].strip().casefold()
    training={prompt(x) for x in train}
    holdout={prompt(x) for x in dev}
    prior={prompt(x) for x in old}
    assert prev["brandNEW_author_synthetic_DEV14"]==14
    assert len(train)==84 and len(dev)==14
    assert len(training)==84 and len(holdout)==14
    assert len({x["id"] for x in train+dev})==98
    assert not (holdout & training) and not (holdout & prior)
    assert Counter(x["kind"] for x in train)=={k:12 for k in KINDS}
    assert Counter(x["kind"] for x in dev)=={k:2 for k in KINDS}
    assert all(x["origin"]==ORIGIN and x["completion"].endswith(".")
               and len(x["completion"].split())>=5
               and len(x["prompt"])>35 for x in dev)
    assert SHADOW_PARAMS==31680 and POSITIONS==(2,3,4)
    assert TOP_K==64 and RANK==32 and TRAIN_EPOCHS==16
    assert 0<KL_TO_ORIGINAL_TOP64<1
    assert 0<ENTROPY_FLOOR_TOLERANCE_BITS<.5
    assert 0<ENTROPY_COLLAPSE_PENALTY<2
    assert all(p in PER_POSITION_LOSS_WEIGHT for p in POSITIONS)
    assert not ALLOW_ORIGINAL_FIRST_BPE_MODIFICATION
    assert not ALLOW_ORIGINAL_GT_AND_CORE_OPTIMIZER
    assert EVAL_OWN_NATIVE_TOKENS==6 and WRONG_OFFSET17_CONTROL==17
    return {
      "stage":STAGE,
      "previous_legal_TRAIN84":84,
      "NEW_original_author_internal_DEV14":14,
      "TRAIN_SHA256":digest(train),
      "DEV_SHA256":digest(dev),
      "zero_past_spent_or_train_prompt_overlap":True,
      "frozen_physical_config":{
        "topk_native49152":TOP_K,"rank":RANK,"shadow_params":SHADOW_PARAMS,
        "allowed_BPE_positions":POSITIONS,
        "train_epochs":TRAIN_EPOCHS,"train_lr":LEARNING_RATE,
        "weight_decay":WEIGHT_DECAY,"bounded_residual":LOGIT_RESIDUAL_BOUND,
        "original_KL_weight":KL_TO_ORIGINAL_TOP64,
        "entropy_floor_tolerance_bits":ENTROPY_FLOOR_TOLERANCE_BITS,
        "entropy_penalty":ENTROPY_COLLAPSE_PENALTY,
        "hard_negative_weight":HARD_NEGATIVE_WEIGHT,
        "position_loss_weight":PER_POSITION_LOSS_WEIGHT,
        "wrong_label_offset":WRONG_OFFSET17_CONTROL,
        "min_eligible_per_position":MIN_ELIGIBLE_TRAIN_PER_POSITION,
        "own_native_generation_steps":EVAL_OWN_NATIVE_TOKENS,
        "gold_prefix_only_during_train":True,
        "own_prefix_only_during_DEV":True,
        "unchanged_original_first_native_BPE":True,
        "no_candidate_gold_injection_during_DEV":True,
        "diversity_gate_original_fraction":MIN_DISTINCT_BPE2_FRACTION_VS_ORIGINAL,
        "paired_new_correct_first3_gate":MIN_FIRST3_PAIRED_NEW_CORRECT,
      },
      "new_internal_DEV_becomes_spent_after_first_physical_scoring":True,
      "not_external_blind_heldout":True,
      "no_J7_EVAL128_GOLD_bpp_CB59_CPU_certification":True,
      "candidate_accepted":False,"output_gate":0,
    }
