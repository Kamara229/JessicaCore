"""S119q2i8z23 FROZEN BEFORE next physical original43M readout.

Brand new 28 author RU synthetic DEV (7 domain x4); strict prompt
disjoint all spent E/G/K/R/T/W/X/Z2/Z9/Z12/Z14/Z16/Z18/Z21.
NO TRAIN in this stage; BOTH saved Z22 A/B ALIGNED+PERMUTED
shadow checkpoints must be strictly SHA pinned and FROZEN.

Calibration scales frozen for diagnostic ablation = (0,2,8,16),
NO selection of 'best' scale on this small DEV. TopK=64, no
EOS masking or new original43M passes for each scale. Compute
original wrong-gold full49152 logit gap and candidate cosine
differential only POSTHOC after target-blind decisions have been
constructed for all scales. Capture fractions of candidate gold
reachable and whether fixed scale actually corrects second native BPE.

This is an internal author synthetic DEV, NOT independent benchmark;
it becomes SPENT on first physical original43M forward. No
616 J7, EVAL128, GOLD, CPU>=90, packed bpp<=0.7 acceptance.
No main, LIVE, Render, Supabase, APK, Jessica, gate0 only.
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

STAGE="CALIBRI_S119Q2I8Z23_PREREG_FRESH28DEV_FROZEN_NATIVE_FUTURE_CALIBRATION_SCALES"
ORIGIN="original_author_S119Q2I8Z23_20261123_frozen_synthetic_DEV"
SCALES=(0.,2.,8.,16.)
TOP_K=64
BASE_PHYSICAL_RUN_ID=45221922458
BASE_ARTIFACT_ID=14841506942
BASE_ARTIFACT_NAME="calibri-s119q2i8z22-original43m-new28train14dev-future16-vs-shuffle-four-shadows-gate0"
BASE_FUTURE_FP16_SHA={
 "CE_ALIGNED":"0a653066470288d0f893ac3791244287728cf9cef587972552167f6ee25b6e8d",
 "CE_PERMUTED_TRAIN":"4aa397f903cbca2cceb9fa1810af1a1589a72a880d39f7d95664eec1e75d910d",
 "CONTEXT_TOP8_ALIGNED":"5977dbf5420eed4e31453a6540afaf038045c9756ebfa05fdcf20cbf92dcd3cd",
 "CONTEXT_TOP8_PERMUTED_TRAIN":"b5a4508edc338e20efdade1dba333a607fcbdd1cee644e2610412abd07b56b3f",
}
BASE_Z22_SOURCE_TRAIN_SHA="0fae97ab0b8ebde70ddc69383ac63b7e61905b035c81c3ec2d0df7a71e3b562f"
BASE_Z22_SOURCE_DEV_SHA="80d3d7e19a7efa7ff3784d85989cd542b69aa1c56db7d4da22ef50b10c239de9"

CASES={
  "sentence_formulation": [
    [
      "Составьте предложение об океанографе, который измеряет солёность воды у острова.",
      "Океанограф измеряет солёность морской воды возле острова."
    ],
    [
      "Напишите предложение о кинологе, который тренирует молодую служебную собаку.",
      "Кинолог тренирует молодую служебную собаку на площадке."
    ],
    [
      "Составьте предложение о стекольщике, который заменяет окно в теплице.",
      "Стекольщик заменяет повреждённое окно в большой теплице."
    ],
    [
      "Напишите предложение о гидрологе, который исследует течение горной реки.",
      "Гидролог исследует быстрое течение холодной горной реки."
    ]
  ],
  "case_agreement": [
    [
      "Исправьте согласование в предложении: «Экскурсанты шли вдоль широкая набережной».",
      "Экскурсанты шли вдоль широкой каменной набережной."
    ],
    [
      "Исправьте грамматическую ошибку: «Мастер подошёл к сломанное станку».",
      "Мастер подошёл к сломанному рабочему станку."
    ],
    [
      "Исправьте окончание слова: «Птица пролетела над высокий сосной».",
      "Птица пролетела над высокой зелёной сосной."
    ],
    [
      "Исправьте ошибку в предложении: «Дети любовались далёкий звездой».",
      "Дети любовались далёкой яркой звездой."
    ]
  ],
  "agent_action_object": [
    [
      "Модель самолёта конструктору показала стажёр Полина. Кто показал модель?",
      "Стажёр Полина показала конструктору модель самолёта."
    ],
    [
      "Набор кистей преподавателю передал художник Денис. Кто передал кисти?",
      "Художник Денис передал преподавателю набор кистей."
    ],
    [
      "Рецепт кондитеру отправила помощница Зоя. Кто отправил рецепт?",
      "Помощница Зоя отправила кондитеру новый рецепт."
    ],
    [
      "Ключи охраннику принесла уборщица Галина. Кто принёс ключи?",
      "Уборщица Галина принесла охраннику запасные ключи."
    ]
  ],
  "time_causality": [
    [
      "Прежде чем отправить письмо, секретарь проверила адрес. Что произошло сначала?",
      "Сначала секретарь проверила правильность почтового адреса."
    ],
    [
      "После монтажа двери строитель установил ручку. Что случилось позже?",
      "Позже строитель установил металлическую дверную ручку."
    ],
    [
      "Перед поездкой в горы проводник осмотрел рюкзак. Что он сделал сначала?",
      "Сначала проводник осмотрел туристический походный рюкзак."
    ],
    [
      "Механик смазал цепь только после чистки велосипеда. Что было раньше?",
      "Сначала механик очистил грязный горный велосипед."
    ]
  ],
  "negation_scope": [
    [
      "Рыбак не вытащил удочку, а достал сачок. Что он достал?",
      "Рыбак достал небольшой рыболовный сачок."
    ],
    [
      "Пилот не включил рацию, а проверил высотомер. Что он проверил?",
      "Пилот проверил исправный бортовой высотомер."
    ],
    [
      "Ландшафтный дизайнер не высадил сосну, а пересадил можжевельник. Что он пересадил?",
      "Дизайнер пересадил молодой декоративный можжевельник."
    ],
    [
      "Музейный работник не упаковал статую, а закрепил картину. Что он закрепил?",
      "Работник закрепил старинную картину на стене."
    ]
  ],
  "arithmetic_units": [
    [
      "На выставке стояло двадцать семь стульев, девять убрали. Сколько осталось? Ответьте предложением.",
      "На выставке осталось восемнадцать деревянных стульев."
    ],
    [
      "У часовщика было одиннадцать шестерёнок, добавили ещё восемь. Сколько стало? Ответьте предложением.",
      "У часовщика стало девятнадцать маленьких шестерёнок."
    ],
    [
      "В оранжерее росло двадцать три орхидеи, семь продали. Сколько осталось? Ответьте предложением.",
      "В оранжерее осталось шестнадцать цветущих орхидей."
    ],
    [
      "На пристани стояло девять лодок, приплыли ещё пять. Сколько стало? Ответьте предложением.",
      "На пристани стало четырнадцать небольших лодок."
    ]
  ],
  "semantic_paraphrase": [
    [
      "Перефразируйте проще: «Метеоролог осуществил регистрацию скорости ветра».",
      "Метеоролог записал измеренную скорость сильного ветра."
    ],
    [
      "Скажите проще: «Капитан произвёл проверку состояния палубы».",
      "Капитан проверил состояние корабельной палубы."
    ],
    [
      "Перефразируйте: «Переводчик осуществил подготовку письменного перевода».",
      "Переводчик подготовил точный письменный перевод."
    ],
    [
      "Скажите проще: «Сотрудник склада произвёл инвентаризацию запасных частей».",
      "Складской работник пересчитал оставшиеся запасные части."
    ]
  ]
}
def rows(split="DEV"):
    if split!="DEV":
        raise ValueError("Z23 only NEW synthetic DEV, NEVER TRAIN, J7, EVAL128 or GOLD")
    return [{"id":f"S119Q2I8Z23_{kind}_{i:02d}","kind":kind,
        "split":"DEV","origin":ORIGIN,"prompt":p,"completion":a}
        for kind in KINDS for i,(p,a) in enumerate(CASES[kind])]

def digest(seq):
    return hashlib.sha256("".join(
        json.dumps(row,ensure_ascii=False,sort_keys=True,separators=(",",":"))+"\n"
        for row in seq).encode("utf8")).hexdigest()

def verify():
    current=rows("DEV")
    past=(e_rows()+g_rows()+k_rows()+r_rows()
        +t_rows("TRAIN")+t_rows("DEV")
        +w_rows("TRAIN")+w_rows("DEV")
        +x_rows("TRAIN")+x_rows("DEV")
        +z2_rows("TRAIN")+z2_rows("DEV")
        +z9_rows("TRAIN")+z9_rows("DEV")
        +z12_rows("DEV")+z14_rows("DEV")
        +z16_rows("TRAIN")+z16_rows("DEV")
        +z18_rows("TRAIN")+z18_rows("DEV")
        +z21_rows("TRAIN")+z21_rows("DEV"))
    now={v["prompt"].strip().casefold() for v in current}
    prev={v["prompt"].strip().casefold() for v in past}
    if not (set(CASES)==set(KINDS)
         and len(current)==28 and len(now)==28 and now.isdisjoint(prev)
         and len({v["id"] for v in current})==28
         and Counter(x["kind"] for x in current)=={k:4 for k in KINDS}
         and all(x["origin"]==ORIGIN and x["completion"].endswith(".")
                 and len(x["completion"].split())>=5
                 and len(x["prompt"])>=40 for x in current)
         and SCALES==(0.,2.,8.,16.) and TOP_K==64
         and BASE_PHYSICAL_RUN_ID==45221922458
         and BASE_ARTIFACT_ID==14841506942
         and BASE_Z22_SOURCE_TRAIN_SHA=="0fae97ab0b8ebde70ddc69383ac63b7e61905b035c81c3ec2d0df7a71e3b562f"
         and BASE_Z22_SOURCE_DEV_SHA=="80d3d7e19a7efa7ff3784d85989cd542b69aa1c56db7d4da22ef50b10c239de9"
         and len(BASE_FUTURE_FP16_SHA)==4
         and all(len(v)==64 and set(v)<=set("0123456789abcdef")
                 for v in BASE_FUTURE_FP16_SHA.values())):
        raise ValueError("Z23 locked brand new28 + fixed scales + frozen Z22 artifact SHA")
    return {
       "stage":STAGE,"new_unseen_synthetic_DEV":28,
       "new_DEV_SHA256":digest(current),
       "seven_domains_four_per_domain":True,
       "disjoint_all_prior_spent_E_G_K_R_T_W_X_Z2_Z9_Z12_Z14_Z16_Z18_Z21":True,
       "precommitted_scale_coefficients":SCALES,
       "precommitted_original_native_topK":TOP_K,
       "frozen_original_Z22_43M_run":BASE_PHYSICAL_RUN_ID,
       "frozen_original_Z22_artifact":BASE_ARTIFACT_ID,
       "frozen_shadow_SHA256":BASE_FUTURE_FP16_SHA,
       "frozen_Z22_TRAIN_SHA256":BASE_Z22_SOURCE_TRAIN_SHA,
       "frozen_Z22_DEV_SHA256":BASE_Z22_SOURCE_DEV_SHA,
       "DEV_becomes_SPENT_on_first_physical_forward":True,
       "no_train_or_tune_on_Z23_DEV":True,
       "source_internal_synthetic_not_blind":True,
    }
