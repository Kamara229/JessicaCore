"""S56 new HELD24 separately hand-labeled event→transfer-recipient graph.

All manual gold fixed BEFORE S56 architecture/weights are trained and
BEFORE S56 native true Calibri43M+GreenTunnel CPU evaluation. This file
may only be imported by the S56 FIRST-READ evaluation process after
frozen S56 model makes all predictions. No original or external EVAL.

An asserted recipient means a real participant in an OUTER transfer or
communication claim, not any Russian dative, prepositional/benefactive
noun, or an embedded quoted transfer reported without an assertion.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,json
from research.nra3_s20.s56_frozen_new_scope24_plain import (
    SOURCE_ROWS,text_sha256,FAMILIES,
)

@dataclass(frozen=True)
class S56ManualHeld:
    row_id:str
    main_predicate:str
    agent:str|None
    transferred_recipient:str|None
    outer_factuality:str
    directed_transfer_polarity:str

HELD_GOLD=(
    ("Q01", "передал", "преподаватель", "декану", "NEGATED", "REJECTED"),
    ("Q02", "отправили", None, "Заместителю", "NEGATED", "REJECTED"),
    ("Q03", "передан", "архивариусом", "председателю", "NEGATED", "REJECTED"),
    ("Q04", "показала", "сотрудница", "клиентке", "NEGATED", "REJECTED"),
    ("Q05", "выдали", None, "Учителю", "NEGATED", "REJECTED"),
    ("Q06", "сообщили", None, "заявителю", "NEGATED", "REJECTED"),
    ("Q07", "доставил", "курьер", "специалисту", "ASSERTED", "POSITIVE"),
    ("Q08", "вручил", "Почтальон", "пенсионерке", "ASSERTED", "POSITIVE"),
    ("Q09", "переслал", "менеджер", "партнёру", "ASSERTED", "POSITIVE"),
    ("Q10", "передали", None, "Студенту", "ASSERTED", "POSITIVE"),
    ("Q11", "отправил", "Руководитель", "нотариусу", "ASSERTED", "POSITIVE"),
    ("Q12", "выдала", "секретарь", "заказчице", "ASSERTED", "POSITIVE"),
    ("Q13", "подходил", "водитель", None, "NEGATED", "NONE"),
    ("Q14", "доверял", "Врач", None, "NEGATED", "NONE"),
    ("Q15", "готовил", "бухгалтер", None, "NEGATED", "NONE"),
    ("Q16", "связывался", "начальник", None, "NEGATED", "NONE"),
    ("Q17", "нравилось", None, None, "NEGATED", "NONE"),
    ("Q18", "удалось", None, None, "NEGATED", "NONE"),
    ("Q19", "заявил", "Инженер", None, "ASSERTED", "NONE"),
    ("Q20", "написал", "Адвокат", None, "ASSERTED", "NONE"),
    ("Q21", "повторил", "Ведущий", None, "ASSERTED", "NONE"),
    ("Q22", "процитировал", "Репортёр", None, "ASSERTED", "NONE"),
    ("Q23", "прочитал", "Сотрудник", None, "ASSERTED", "NONE"),
    ("Q24", "сказала", "Участница", None, "ASSERTED", "NONE"),
)

def held_gold24():
    assert len(HELD_GOLD)==len(SOURCE_ROWS)==24
    items=tuple(S56ManualHeld(*r) for r in HELD_GOLD)
    assert [g.row_id for g in items]==[r[0] for r in SOURCE_ROWS]
    assert [sum(g.directed_transfer_polarity==p for g in items)
            for p in ("REJECTED","POSITIVE","NONE")]==[6,6,12]
    assert all((g.transferred_recipient is None)==(g.directed_transfer_polarity=="NONE")
               for g in items)
    return items

def gold_sha256():
    return hashlib.sha256(json.dumps(HELD_GOLD,ensure_ascii=False,
        sort_keys=True,separators=(",",":")).encode("utf-8")).hexdigest()
