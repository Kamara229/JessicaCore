"""S119q2i8k NEW predeclared 14 author multiword Russian DEV, never TRAIN.

After completed 8j physical source135M gold continuation rank forensic.
Synthetic, visible internal, cannot be called independent external blind.
Never tune optimizer, sample, CE/KL weight, or architecture on DEV labels.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import json
from research.calibri_nra3.s119q2i8e_new_ru_multiword_train_source import (
    KINDS,rows as earlier_train,
)
from research.calibri_nra3.s119q2i8g_new_ru_multiword_dev_source import (
    rows as earlier_g_dev,
)
from research.calibri_nra3.s119q2i8d_new_russian_heldout_source import (
    rows as earlier_d_dev,
)
STAGE="CALIBRI_S119Q2I8K_14_NEW_AUTHOR_MULTIWORD_DEV_NEVER_TRAIN"
ORIGIN="original_new_author_14_multiword_DEV_20261122_s119q2i8k"
CASES={
  "sentence_formulation": [
    [
      "Напишите короткое предложение о кошке, которая прячется в тени.",
      "Кошка спокойно отдыхает в тени."
    ],
    [
      "Составьте полное предложение о рыбаке возле озера.",
      "Рыбак сидит возле озера."
    ]
  ],
  "case_agreement": [
    [
      "Исправьте падежи: «Школьник подошёл к новая пристань».",
      "Школьник подошёл к новой пристани."
    ],
    [
      "Исправьте фразу: «Писатель гордился интересная книга».",
      "Писатель гордился интересной книгой."
    ]
  ],
  "agent_action_object": [
    [
      "Фонарик Андрею отдала Нина. Кто отдал фонарик? Коротко.",
      "Нина отдала фонарик."
    ],
    [
      "Картину Аркадию показала Лиза. Кто показал картину? Коротко.",
      "Лиза показала картину."
    ]
  ],
  "time_causality": [
    [
      "Сначала выключили радио, потом закрыли дверь. Что было раньше? Фразой.",
      "Сначала выключили радио."
    ],
    [
      "Когда туман рассеялся, самолёт взлетел. Что произошло позже? Фразой.",
      "Позже взлетел самолёт."
    ]
  ],
  "negation_scope": [
    [
      "Оля не рисовала дерево, она рисовала солнце. Что рисовала Оля? Коротко.",
      "Оля рисовала солнце."
    ],
    [
      "Илья не мыл машину, он мыл велосипед. Что мыл Илья? Фразой.",
      "Илья мыл велосипед."
    ]
  ],
  "arithmetic_units": [
    [
      "В вазе было семь роз, три розы подарили. Сколько осталось? Фразой.",
      "Осталось четыре розы."
    ],
    [
      "На полке стояли четыре книги, добавили ещё три. Сколько книг теперь? Кратко.",
      "Стало семь книг."
    ]
  ],
  "semantic_paraphrase": [
    [
      "Перефразируйте просто: «Строитель приступил к ремонту».",
      "Рабочий начал ремонт."
    ],
    [
      "Скажите проще: «Курьер доставил посылку адресату».",
      "Курьер привёз посылку."
    ]
  ]
}

def rows(split="DEV"):
    if split!="DEV":
        raise ValueError("S119q2i8k DEV-only source, sealed-EVAL and TRAIN forbidden")
    return [{
        "id":f"S119Q2I8K_{kind}_{i:02d}","kind":kind,"split":"DEV",
        "prompt":p,"completion":a,"origin":ORIGIN}
        for kind in KINDS for i,(p,a) in enumerate(CASES[kind])
    ]

def verify():
    d=rows()
    blocked={str(x["prompt"]).strip().casefold() for x in (
        earlier_train()+earlier_g_dev()+earlier_d_dev())}
    live=[str(x["prompt"]).strip().casefold() for x in d]
    if (len(d)!=14 or len(set(live))!=14
        or not set(live).isdisjoint(blocked)
        or Counter(x["kind"] for x in d)!={k:2 for k in KINDS}
        or any(len(x["completion"].split())<3
               or not x["completion"].endswith(".")
               for x in d)):
        raise ValueError("new author DEV only, disjoint all existing TRAIN / spent DEV")
    sha=hashlib.sha256("".join(
        json.dumps(x,ensure_ascii=False,sort_keys=True,
                   separators=(",",":"))+"\n" for x in d).encode("utf8")).hexdigest()
    return {"stage":STAGE,"new_DEV":14,"sha256":sha,
            "DEV_only_no_training":True,
            "source_disjoint_prior70_TRAIN_14G_14D_DEV":True,
            "synthetic_internal_VISIBLE_not_external_blind":True,
            "no_protected_616J7_or_EVAL128":True}
