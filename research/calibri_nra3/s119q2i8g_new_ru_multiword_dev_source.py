"""S119q2i8g: NEW source-disjoint synthetic Russian multiword DEV14.
Predeclared before any S119q2i8g physical model training, not seen by Adam
or teacher 135M. Never claims external independent/blind; no protected J7.
"""
from __future__ import annotations
import hashlib
import json
from collections import Counter
from research.calibri_nra3.s119q2i8e_new_ru_multiword_train_source import (
    KINDS,rows as train_rows,
)
from research.calibri_nra3.s119q2i8d_new_russian_heldout_source import (
    rows as earlier_dev,
)
STAGE="CALIBRI_S119Q2I8G_NEW_14_INTERNAL_MULTIWORD_DEV_READONLY"
ORIGIN="original_author_new_russian_multiword_14DEV_not_external_blind"
ITEMS={
  "sentence_formulation": [
    [
      "Напишите краткое законченное предложение о звенящем будильнике.",
      "Будильник громко звенит."
    ],
    [
      "Составьте простую фразу о том, как солнце согревает камни.",
      "Солнце согревает камни."
    ]
  ],
  "case_agreement": [
    [
      "Исправьте падежные формы: «Мальчик подошёл к высокая башня».",
      "Мальчик подошёл к высокой башне."
    ],
    [
      "Исправьте фразу: «Она любовалась зелёный сад».",
      "Она любовалась зелёным садом."
    ]
  ],
  "agent_action_object": [
    [
      "Свечу Валере зажгла Инга. Кто зажёг свечу? Краткой фразой.",
      "Инга зажгла свечу."
    ],
    [
      "Записку Тамаре принёс Фома. Кто принёс записку? Полной фразой.",
      "Фома принёс записку."
    ]
  ],
  "time_causality": [
    [
      "Перед прогулкой выключили утюг. Что сделали раньше? Короткой фразой.",
      "Сначала выключили утюг."
    ],
    [
      "Дождь закончился, затем открыли окна. Что сделали позже? Фразой.",
      "Потом открыли окна."
    ]
  ],
  "negation_scope": [
    [
      "Рая не убирала книги, она убрала чашки. Что убрала Рая? Кратко.",
      "Рая убрала чашки."
    ],
    [
      "Гоша не ел суп, он ел кашу. Что он ел? Короткой фразой.",
      "Гоша ел кашу."
    ]
  ],
  "arithmetic_units": [
    [
      "Три груши лежали на столе, принесли ещё две. Сколько теперь? Краткой фразой.",
      "Стало пять груш."
    ],
    [
      "В корзине были девять мячей, четыре вынули. Сколько осталось? Фразой.",
      "Осталось пять мячей."
    ]
  ],
  "semantic_paraphrase": [
    [
      "Перефразируйте просто: «Турист поспешил к выходу».",
      "Путешественник быстро вышел."
    ],
    [
      "Скажите проще: «Режиссёр завершил обсуждение».",
      "Режиссёр закончил разговор."
    ]
  ]
}
def rows(split="DEV"):
    if split!="DEV":
        raise ValueError("Q2I8g heldout contains only DEV; never TRAIN/EVAL")
    return [
        {"id":f"S119Q2I8G_{k}_{i:02d}","kind":k,
         "split":"DEV","prompt":q,"completion":a,"origin":ORIGIN}
        for k in KINDS for i,(q,a) in enumerate(ITEMS[k])
    ]
def _digest(xs):
    return hashlib.sha256(b"".join((
        json.dumps(x,ensure_ascii=False,sort_keys=True,
                   separators=(",",":"),allow_nan=False)+"\n").encode("utf8")
        for x in xs)).hexdigest()
def verify():
    xs=rows()
    previous={x["prompt"].casefold() for x in train_rows()+earlier_dev()}
    if (len(xs)!=14 or len({x["prompt"].casefold() for x in xs})!=14
        or {x["prompt"].casefold() for x in xs}&previous
        or Counter(x["kind"] for x in xs)!={k:2 for k in KINDS}
        or any(len(x["completion"].split())<2
               or len(x["completion"])>80
               or len(x["prompt"])>180 for x in xs)):
        raise ValueError("DEV-only 14 unseen questions, 2 per domain")
    return {
        "stage":STAGE,
        "original_new_author_heldout_DEV_count":14,
        "source_new_DEV_sha256":_digest(xs),
        "source_disjoint_new_70_TRAIN_and_old_14_DEV":True,
        "internally_visible_NOT_independent_external_blind":True,
        "no_EVAL_J7_protected_gold_exposed":True,
    }
