"""S53 separate hand-gold source committed after raw24, before evaluator.

Gold manually declares main clause predicate, agent, dative recipient,
event factuality and directed recipient polarity. Gold is never loaded
into original Calibri43M/GT nor S26 model or quote adapter forward.
All 24 are author-constructed TRAIN-only diagnostic challenges, not a
random independent benchmark, and never fitted on this stage.

For reported quotation rows, main assertive event is the outer
speech verb. Embedded quoted recipient edges are NOT direct factual
assertions; quoted inner words do not constitute main event truth.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json

from research.nra3_s20.s53_precommitted_negation_quote24_plain import (
    SOURCE_ROWS,text_sha256,
)

STAGE="NRA3_S53_SEPARATE_MANUAL_GOLD_NEGATION_QUOTE24"
@dataclass(frozen=True)
class ManualGold53:
    row_id:str
    predicate:str
    agent:str
    recipient:str|None
    event_status:str
    recipient_polarity:str

MANUAL_GOLD24=(
    ("N01","отправила","Нина","Павлу","ASSERTED","POSITIVE"),
    ("N02","передал","Игорь","Марине","ASSERTED","POSITIVE"),
    ("N03","вручила","Тамара","Сергею","ASSERTED","POSITIVE"),
    ("N04","доставил","Пётр","Алине","ASSERTED","POSITIVE"),
    ("N05","показала","Ольга","Ивану","ASSERTED","POSITIVE"),
    ("N06","сообщил","Борис","Анне","ASSERTED","POSITIVE"),
    ("N07","отправила","Нина","Павлу","NEGATED","REJECTED"),
    ("N08","передал","Игорь","Марине","NEGATED","REJECTED"),
    ("N09","вручила","Тамара","Сергею","NEGATED","REJECTED"),
    ("N10","доставил","Пётр","Алине","NEGATED","REJECTED"),
    ("N11","показала","Ольга","Ивану","NEGATED","REJECTED"),
    ("N12","сообщил","Борис","Анне","NEGATED","REJECTED"),
    ("N13","отправила","Нина","Павлу","NEGATED","REJECTED"),
    ("N14","передал","Игорь","Марине","NEGATED","REJECTED"),
    ("N15","вручила","Тамара","Сергею","NEGATED","REJECTED"),
    ("N16","доставил","Пётр","Алине","NEGATED","REJECTED"),
    ("N17","показала","Ольга","Ивану","NEGATED","REJECTED"),
    ("N18","сообщил","Борис","Анне","NEGATED","REJECTED"),
    ("N19","произнесла","Нина",None,"ASSERTED","NONE"),
    ("N20","написал","Игорь",None,"ASSERTED","NONE"),
    ("N21","отметила","Тамара",None,"ASSERTED","NONE"),
    ("N22","прочитал","Пётр",None,"ASSERTED","NONE"),
    ("N23","сказала","Ольга",None,"ASSERTED","NONE"),
    ("N24","добавил","Борис",None,"ASSERTED","NONE"),
)

def gold24():
    assert len(MANUAL_GOLD24)==len(SOURCE_ROWS)==24
    out=tuple(ManualGold53(*row) for row in MANUAL_GOLD24)
    assert [g.row_id for g in out]==[t[0] for t in SOURCE_ROWS]
    assert [sum(g.event_status==k for g in out)
            for k in ("ASSERTED","NEGATED")]==[12,12]
    assert [sum(g.recipient_polarity==k for g in out)
            for k in ("POSITIVE","REJECTED","NONE")]==[6,12,6]
    assert all((g.recipient is None)==(g.recipient_polarity=="NONE") for g in out)
    return out

def gold_sha256():
    return hashlib.sha256(json.dumps(MANUAL_GOLD24,sort_keys=True,
         ensure_ascii=False,separators=(",",":")).encode("utf-8")).hexdigest()

if __name__=="__main__":
    print(json.dumps({"stage":STAGE,"raw_sha256":text_sha256(),
        "gold_sha256":gold_sha256(),"count":len(gold24())},sort_keys=True))
