"""S119q2g new original-authored Russian own-token generation prompts.

Prepared BEFORE loading any model weights. Not part of q2a/q2c1 TRAIN,
not q2a CALIBRATION, not the q2c1 familiar SOURCE_PROBE nor the previously
spent S119n six diagnostic prompts. New lexemes, grammar and instruction
forms. No learning, teacher forcing, gold answers in any model inputs.
These 12 author-constructed prompts are a FRESH-UNTRAINED diagnostic,
NOT a hidden, externally independent or statistically general QA sample.

All examples and normalization are frozen in repository source. Do not
use scores as optimizer data or tune future model architecture on them.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter

STAGE="CALIBRI_S119Q2G_FRESH_AUTHOR_12_OWN_TOKEN_GENERATION_SOURCE"
KINDS=("roles_word_order","negation","transitive_comparison",
       "event_time","arithmetic","paraphrase")
EXAMPLES=(
 ("roles_word_order",
  "Письмо Тане отправил Илья. Кто отправил письмо? Только имя.",
  "Илья."),
 ("roles_word_order",
  "Книгу Игорю принесла Даша. Кто принёс книгу? Только имя.",
  "Даша."),
 ("negation",
  "Алина не купила билет на автобус. Купила ли Алина билет? Да или нет.",
  "Нет."),
 ("negation",
  "Сергей не опоздал на поезд. Опоздал ли Сергей? Да или нет.",
  "Нет."),
 ("transitive_comparison",
  "Лента Лизы длиннее ленты Нины. Чья лента длиннее? Только имя.",
  "Лизы."),
 ("transitive_comparison",
  "Олег бежит медленнее Никиты. Кто быстрее? Только имя.",
  "Никита."),
 ("event_time",
  "Сразу после закрытия окна раздался звонок. Что случилось раньше? Кратко.",
  "Закрытие окна."),
 ("event_time",
  "Сначала выключили компьютер. Лишь потом заперли дверь. Что было позже? Кратко.",
  "Заперли дверь."),
 ("arithmetic",
  "На столе лежали 17 монет. Девять забрали. Сколько осталось? Только число.",
  "8."),
 ("arithmetic",
  "В корзине находилось шесть груш. Добавили ещё семь. Сколько стало? Только число.",
  "13."),
 ("paraphrase",
  "Назови одним словом синоним глагола «торопился».",
  "Спешил."),
 ("paraphrase",
  "Замени глагол «взглянул» одним словом-синонимом.",
  "Посмотрел."),
)
EXPECTED_SOURCE_SHA256=None  # frozen after first actual successful source verification


def check_source(ok,reason):
    if not ok:
        raise ValueError("S119q2g SOURCE FAIL-CLOSED: "+reason)


def rows():
    check_source(len(EXAMPLES)==12,"exact twelve distinct author-written probes")
    result=[]
    for index,(kind,prompt,gold) in enumerate(EXAMPLES):
        result.append({
            "source_id":"S119Q2G_AUTHOR_FRESH_%02d"%index,
            "source_kind":kind,
            "question":prompt,
            "canonical_short_answer":gold,
            "provenance":"original_author_constructed_untrained_not_blind",
            "split":"FRESH_SOURCE_PROBE_NEVER_TRAIN",
        })
    return result


def canon(value):
    return (json.dumps(value,ensure_ascii=False,sort_keys=True,
                       separators=(",",":"),allow_nan=False)+"\n").encode("utf8")


def source_hash(records):
    return hashlib.sha256(b"".join(canon(r) for r in records)).hexdigest()


def normalized(text):
    """Fixed strict short-answer exact metric, never model-specific."""
    check_source(type(text) is str,"string only")
    return re.sub(r"\s+"," ",re.sub(
        r"[.,!?;:«»\"'…—–()]+"," ",text.casefold())).strip()


def validate(records):
    check_source(type(records) is list and records==rows(),
                 "no new generated answers, changed prompts or splits")
    known=set(); golds=[]
    for row in records:
        prompt=row["question"]
        check_source(prompt and len(prompt)<200 and
                     prompt not in known and
                     normalized(prompt)!=normalized(row["canonical_short_answer"]),
                     "distinct no-answer-leak source prompts")
        known.add(prompt);golds.append(row["canonical_short_answer"])
    check_source(Counter(x["source_kind"] for x in records)==
                 Counter({k:2 for k in KINDS}),
                 "two original new cases in each of six semantic domains")
    from research.calibri_nra3.s119q2a_train_source_manifest import (
        build_rows,SPENT_S119N_PROMPTS
    )
    from research.calibri_nra3.s119q2c1_author_counterfactual_pairs import (
        rows as earlier_authored
    )
    previous={x["prompt"] for x in build_rows()}|{
        x["prompt"] for x in earlier_authored()
    }|set(SPENT_S119N_PROMPTS)
    check_source(not previous & known,
                 "fresh source must never repeat prior TRAIN or spent probes")
    digest=source_hash(records)
    if EXPECTED_SOURCE_SHA256 is not None:
        check_source(digest==EXPECTED_SOURCE_SHA256,
                     "pre-registered SHA256 of 12 authored cases unchanged")
    return {
        "stage":STAGE,
        "twelve_source_sha256":digest,
        "source_example_count":12,
        "six_task_areas_each_two":True,
        "not_trained_or_seen_in_prior_traces":True,
        "not_independent_blind_holdout":True,
        "source_stored_before_model_load":True,
        "no_old_EVAL128_S118_B_GOLD_or_J7":True,
        "canonicalization":"casefold_space_punctuation_fixed",
    }
