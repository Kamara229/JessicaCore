"""S57 separately manually frozen NEW HELD24 outer-event gold.

This is a FIRST-READ, researcher-authored, structural 24-case audit.
Gold committed BEFORE event-first algorithm/optimizer or original
Calibri43M+GT CPU HELD prefill. Predicate and outer transferred
recipient are annotated, with REJECTED defined only for event transfer
facts and never for embedded reported quotations. Not real-world EVAL.

CRITICAL: imported only AFTER original43M+GT, old S26, S54, S56 and
new S57 all produce frozen predictions on source raw HELD text.
No S57 weights/threshold are ever selected on this gold.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,json
from research.nra3_s20.s57_frozen_eventfirst_held24_plain import (
    SOURCE_ROWS,text_sha256,
)
@dataclass(frozen=True)
class S57Gold:
    row_id:str
    main_predicate:str
    main_agent:str|None
    transferred_recipient:str|None
    event_factuality:str
    directed_transfer_polarity:str

GOLD24=(
    ("Z01", "выдала", "секретарь", "инспектору", "NEGATED", "REJECTED"),
    ("Z02", "отправили", None, "Наблюдателю", "NEGATED", "REJECTED"),
    ("Z03", "передан", "мастером", "инженеру", "NEGATED", "REJECTED"),
    ("Z04", "вручил", "пристав", "представителю", "NEGATED", "REJECTED"),
    ("Z05", "передавали", None, "Библиотекарю", "NEGATED", "REJECTED"),
    ("Z06", "показала", "заведующая", "проверяющей", "NEGATED", "REJECTED"),
    ("Z07", "доставила", "сотрудница", "архивисту", "ASSERTED", "POSITIVE"),
    ("Z08", "передал", "Регистратор", "студентке", "ASSERTED", "POSITIVE"),
    ("Z09", "предоставили", None, "редакции", "ASSERTED", "POSITIVE"),
    ("Z10", "отправили", None, "подрядчику", "ASSERTED", "POSITIVE"),
    ("Z11", "отправил", "Техник", "оператору", "ASSERTED", "POSITIVE"),
    ("Z12", "передал", "менеджер", "заказчику", "ASSERTED", "POSITIVE"),
    ("Z13", "доверял", "Экскурсовод", None, "NEGATED", "NONE"),
    ("Z14", "подходил", "техник", None, "NEGATED", "NONE"),
    ("Z15", "составлял", "бухгалтер", None, "NEGATED", "NONE"),
    ("Z16", "понравилось", None, None, "NEGATED", "NONE"),
    ("Z17", "проводил", "директор", None, "NEGATED", "NONE"),
    ("Z18", "помогал", "Врач", None, "NEGATED", "NONE"),
    ("Z19", "заявил", "Пресс-секретарь", None, "ASSERTED", "NONE"),
    ("Z20", "записала", "Редактор", None, "ASSERTED", "NONE"),
    ("Z21", "процитировал", "Аналитик", None, "ASSERTED", "NONE"),
    ("Z22", "повторила", "Ведущая", None, "ASSERTED", "NONE"),
    ("Z23", "прочитал", "Инструктор", None, "ASSERTED", "NONE"),
    ("Z24", "добавил", "Юрист", None, "ASSERTED", "NONE"),
)
def held_gold24():
    assert len(GOLD24)==len(SOURCE_ROWS)==24
    gold=tuple(S57Gold(*g) for g in GOLD24)
    assert [g.row_id for g in gold]==[r[0] for r in SOURCE_ROWS]
    assert [sum(g.directed_transfer_polarity==p for g in gold)
            for p in ("POSITIVE","REJECTED","NONE")]==[6,6,12]
    assert [sum(g.event_factuality==p for g in gold)
            for p in ("NEGATED","ASSERTED")]==[12,12]
    return gold
def gold_sha256():
    return hashlib.sha256(json.dumps(GOLD24,ensure_ascii=False,
         sort_keys=True,separators=(",",":")).encode("utf-8")).hexdigest()
