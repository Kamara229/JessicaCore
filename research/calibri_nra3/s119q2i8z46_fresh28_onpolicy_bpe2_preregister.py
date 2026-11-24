"""Z46 SOURCE-ONLY PRECOMMIT of 28 brand-new internal human authored DEV.

Preregister one true original S142 pretrained43M + original 12GT + Z10 CE
autonomous native49152 BPE1/2/3 comparison vs Z45 exact centered-RMS
30,720-param BPE2 shadow trained on PREVIOUS legal84 own-correct-first,
native-gold-in-original-target-blind-top64 only.

Three decoding arms: original full-native greedy; same HUMAN-TRAIN shadow;
wrong-offset17-label-control shadow with identical initialization/epoch
budget. BPE1 and BPE3 unchanged native S142+GT12; only BPE2 uses guarded
top64 candidate scores. All prompt gold is POSTHOC during DEV inference.
One score per fresh DEV, no dev optimizer, sweeps or post-score retuning.
All 28 become SPENT on first physical scoring; this is internal
author-created DEV, NOT external blind holdout, 616J7/GOLD/EVAL128,
nor any acceptance/production evidence.
"""
from __future__ import annotations
import hashlib
import json
import math
from collections import Counter
from research.calibri_nra3.s119q2i8e_new_ru_multiword_train_source import KINDS
from research.calibri_nra3 import s119q2i8z40_new14dev_multiposition_entropy_prereg as z40
from research.calibri_nra3 import s119q2i8z45_trainonly_onpolicy_bpe2_domain_holdout as z45

STAGE="CALIBRI_S119Q2I8Z46_PRECOMMITTED_28_UNSEEN_INTERNAL_DEV_ONPOLICY_BPE2_CROSSPROMPT_DIVERSITY"
ORIGIN="author_internal_new_Z46_20261124_nativeBPE2_entropy_crossprompt_holdout"
DEV_COUNT=28
TRAIN_COUNT=84
SAMPLES_PER_KIND=4
TOP_K=64
SEED=119145
RANK=32
PARAMS=30720
TRAIN_EPOCHS=8
TRAIN_LR=.004
TRAIN_WEIGHT_DECAY=.04
WRONG_OFFSET=17
EVAL_FIRST_NATIVE_TOKENS=3
ORIGINAL_FIRST_NATIVE_UNCHANGED=True
ORIGINAL_THIRD_NATIVE_UNCHANGED=True
SHADOW_ONLY_SECOND_NATIVE=True
PER_PROMPT_ENTROPY_KL_GUARD=True
NO_GOLD_CANDIDATE_INSERTION=True
NO_DEV_GRADIENTS=True
NO_DEV_TEACHER_FORCING=True
NO_NEW_TRAIN_EXAMPLES=True
# Predeclared informative exploratory thresholds, never deployment promotion.
MIN_HUMAN_DEV_FIRST2=4
MIN_HUMAN_FIRST2_OVER_EACH_CONTROL=3
MIN_HUMAN_DEV_FIRST3=1
MIN_DISTINCT_SECOND_FRACTION_VS_BASELINE=.75
MIN_CROSSPROMPT_ENTROPY_DELTA_BITS=-.25
MAX_PREMATURE_EOS_FIRST3=0
REQUIRE_ALL_POSITIONS_CAUSAL_OWN_PREFIX=True
CANDIDATE_ACCEPTED=False
OUTPUT_GATE=0

DEV_CASES={
 "sentence_formulation":[
  ("Составьте предложение о смотрителе маяка, который проверяет сигнальный фонарь.",
   "Смотритель маяка осторожно проверил яркий сигнальный фонарь."),
  ("Напишите полное предложение о пекаре, который украшает праздничный пирог.",
   "Пекарь аккуратно украсил свежий праздничный пирог."),
  ("Составьте предложение о леснике, который отмечает молодые сосны на карте.",
   "Лесник отметил молодые сосны на подробной карте."),
  ("Напишите предложение об инженере, который проверяет крепления мостика.",
   "Инженер внимательно осмотрел металлические крепления небольшого мостика.")],
 "case_agreement":[
  ("Исправьте падежные ошибки: «Ребёнок подошёл к высокая ограде». Ответьте полным предложением.",
   "Ребёнок подошёл к высокой садовой ограде."),
  ("Исправьте согласование: «Путник остановился возле старый башни». Ответьте предложением.",
   "Путник остановился возле старой каменной башни."),
  ("Исправьте ошибку: «Садовник любовался яркая клумбой». Сформулируйте правильное предложение.",
   "Садовник любовался яркой весенней клумбой."),
  ("Исправьте падеж: «Письма лежали под тяжёлый коробкой». Сформулируйте правильно.",
   "Письма лежали под тяжёлой картонной коробкой.")],
 "agent_action_object":[
  ("Секретарю папку передала курьер Ирина. Кто передал папку и кому? Ответьте предложением.",
   "Курьер Ирина передала секретарю синюю рабочую папку."),
  ("Ремонтнику инструменты принёс сосед Павел. Кто принёс инструменты и кому? Ответьте полным предложением.",
   "Сосед Павел принёс ремонтнику необходимые рабочие инструменты."),
  ("Редактору рукопись показала студентка Ольга. Кто совершил действие? Дайте полный ответ.",
   "Студентка Ольга показала редактору новую рукопись."),
  ("Фотографу объектив отдала ассистентка Лена. Кто отдал объектив и кому? Ответьте предложением.",
   "Ассистентка Лена отдала фотографу запасной объектив.")],
 "time_causality":[
  ("Перед тем как включить насос, оператор проверил вентиль. Что произошло раньше? Ответьте предложением.",
   "Сначала оператор тщательно проверил закрытый вентиль."),
  ("После завершения уборки заведующий запер склад. Что случилось позже? Ответьте полным предложением.",
   "Позже заведующий надёжно запер пустой склад."),
  ("Сначала диспетчер записал адрес, затем вызвал курьера. Какое действие произошло вторым? Ответьте предложением.",
   "Затем диспетчер вызвал свободного курьера."),
  ("Прежде чем развернуть карту, штурман убрал компас. Что произошло первым? Ответьте полным предложением.",
   "Сначала штурман убрал точный морской компас.")],
 "negation_scope":[
  ("Художница не рисовала портрет, а грунтовала холст. Что делала художница? Ответьте предложением.",
   "Художница тщательно грунтовала чистый большой холст."),
  ("Электрик не заменял лампу, а измерял напряжение. Чем занимался электрик? Ответьте предложением.",
   "Электрик аккуратно измерял напряжение в розетке."),
  ("Садовод не срезал ветви, а собирал опавшие листья. Что делал садовод? Ответьте полным предложением.",
   "Садовод собирал сухие опавшие осенние листья."),
  ("Повар не жарил рыбу, а нарезал овощи. Что делал повар? Ответьте предложением.",
   "Повар мелко нарезал свежие сезонные овощи.")],
 "arithmetic_units":[
  ("В корзине было тридцать одно яблоко, девять взяли. Сколько осталось? Ответьте полным предложением.",
   "В корзине осталось двадцать два спелых яблока."),
  ("На стоянке стояло двадцать семь самокатов, добавили шесть. Сколько их теперь? Ответьте предложением.",
   "На стоянке теперь тридцать три самоката."),
  ("В альбоме было сорок две фотографии, восемь удалили. Сколько осталось? Ответьте полным предложением.",
   "В альбоме осталось тридцать четыре фотографии."),
  ("На полке стояло девятнадцать словарей, привезли пять новых. Сколько стало? Ответьте предложением.",
   "На полке теперь двадцать четыре словаря.")],
 "semantic_paraphrase":[
  ("Переформулируйте просто: «Механик произвёл диагностику системы охлаждения». Ответьте предложением.",
   "Механик проверил работу системы охлаждения двигателя."),
  ("Скажите проще: «Куратор осуществил проверку списка участников». Сформулируйте полное предложение.",
   "Куратор проверил список зарегистрированных участников встречи."),
  ("Перефразируйте: «Техник выполнил контроль состояния аккумулятора». Ответьте предложением.",
   "Техник проверил состояние старого аккумулятора."),
  ("Скажите проще: «Экскурсовод осуществил демонстрацию исторического макета». Ответьте предложением.",
   "Экскурсовод показал посетителям старинный музейный макет.")],
}
def rows(split):
    if split=="TRAIN":
        return z40.rows("TRAIN")
    if split=="DEV":
        return [
            {"id":f"S119Q2I8Z46_{kind}_{idx+1:02d}",
             "kind":kind,"split":"DEV","origin":ORIGIN,
             "prompt":prompt,"completion":completion}
            for kind in KINDS
            for idx,(prompt,completion) in enumerate(DEV_CASES[kind])
        ]
    raise ValueError("Only previous LEGAL TRAIN or preregistered new DEV; no EVAL/J7/GOLD")

def digest(items):
    return hashlib.sha256("".join(json.dumps(
        row,sort_keys=True,ensure_ascii=False,separators=(",",":"))+"\n"
        for row in items).encode("utf-8")).hexdigest()

def verify():
    previous=z40.verify()
    train=rows("TRAIN");dev=rows("DEV")
    spent=z40.previous_spent_and_train_prompts()+z40.rows("DEV")
    norm=lambda r:" ".join(r["prompt"].strip().casefold().split())
    old={norm(r) for r in spent}
    training={norm(r) for r in train}
    new={norm(r) for r in dev}
    assert len(train)==TRAIN_COUNT==84
    assert len(dev)==DEV_COUNT==28
    assert len(training)==84 and len(new)==28
    assert not new.intersection(old) and not new.intersection(training)
    assert len({r["id"] for r in train+dev})==112
    assert Counter(r["kind"] for r in dev)==Counter({k:SAMPLES_PER_KIND for k in KINDS})
    assert all(r["split"]=="DEV" and r["origin"]==ORIGIN and
               r["completion"].endswith(".") and
               len(r["completion"].split())>=5 and
               len(r["prompt"])>=45 for r in dev)
    assert previous["zero_past_spent_or_train_prompt_overlap"] is True
    assert previous["TRAIN_SHA256"]==z45.SOURCE_SHA
    assert TOP_K==64 and SEED==119145 and RANK==32 and PARAMS==30720
    assert (TRAIN_EPOCHS,TRAIN_LR,TRAIN_WEIGHT_DECAY)==(z45.EPOCHS,z45.LR,z45.WEIGHT_DECAY)
    assert EVAL_FIRST_NATIVE_TOKENS==3
    assert WRONG_OFFSET==17
    assert 0<MIN_CROSSPROMPT_ENTROPY_DELTA_BITS+1 <1
    assert not CANDIDATE_ACCEPTED and OUTPUT_GATE==0
    return {
        "stage":STAGE,
        "previous_84_legal_TRAIN_SHA256":digest(train),
        "brand_NEW_28_internal_DEV_SHA256":digest(dev),
        "previous_84_legal_TRAIN":84,
        "NEW_28_INTERNAL_DEV":28,
        "four_cases_per_kind":4,
        "source_disjoint_all_known_prior_spent_DEV_and_train":True,
        "frozen_physical_config":{
          "topk_native":TOP_K,"native_vocab":49152,
          "rank":RANK,"shadow_params":PARAMS,
          "seed":SEED,"train_epochs":TRAIN_EPOCHS,
          "train_lr":TRAIN_LR,"train_weight_decay":TRAIN_WEIGHT_DECAY,
          "wrong_label_offset":WRONG_OFFSET,
          "eval_own_native_positions":EVAL_FIRST_NATIVE_TOKENS,
          "only_second_native_BPE_rerank":SHADOW_ONLY_SECOND_NATIVE,
          "original_first_and_third_untouched":
              ORIGINAL_FIRST_NATIVE_UNCHANGED and ORIGINAL_THIRD_NATIVE_UNCHANGED,
          "per_prompt_original_KL_and_entropy_guard":PER_PROMPT_ENTROPY_KL_GUARD,
          "DEV_target_blind_candidate_selection":NO_GOLD_CANDIDATE_INSERTION,
          "DEV_own_prefix_no_teacher_forcing":NO_DEV_TEACHER_FORCING,
          "DEV_optimizer_steps_zero":NO_DEV_GRADIENTS,
          "thresholds_nonpromoting":{
            "human_first2_min":MIN_HUMAN_DEV_FIRST2,
            "human_minus_each_control_first2_min":MIN_HUMAN_FIRST2_OVER_EACH_CONTROL,
            "human_first3_min":MIN_HUMAN_DEV_FIRST3,
            "distinct_second_original_fraction_min":MIN_DISTINCT_SECOND_FRACTION_VS_BASELINE,
            "crossprompt_entropy_change_vs_original_min_bits":MIN_CROSSPROMPT_ENTROPY_DELTA_BITS,
            "premature_EOS_first3_max":MAX_PREMATURE_EOS_FIRST3
          },
        },
        "independent_external_blind_holdout":False,
        "becomes_SPENT_after_first_physical_DEV_evaluation":True,
        "no_previous_SPENT_DEV_optimized_or_evaluated":True,
        "no_616J7_EVAL128_GOLD_CPU_bpp_certification":True,
        "candidate_accepted":False,"output_gate":0,
    }
