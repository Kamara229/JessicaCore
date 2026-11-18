"""S54 NEW HELD24 manually authored directed graph gold, independently frozen.

This is the S54 first-read dataset: NEVER train, calibrate, select
hyperparameters or edit candidate based on labels. It is researcher
authored with lexically disjoint dative pairs but same four categories;
not an unbiased real-world benchmark. Only import AFTER frozen S54
training and original S26 graph+residual first HELD forward.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,json
from research.nra3_s20.s54_frozen_new_held24_plain_source import SOURCE_ROWS,text_sha256

@dataclass(frozen=True)
class ManualHeld54:
    row_id:str
    predicate:str
    agent:str
    recipient:str|None
    event_status:str
    recipient_polarity:str

GOLD24=(
    ("H01","отправила","Дарья","Виктору","ASSERTED","POSITIVE"),
    ("H02","переслал","Роман","Светлане","ASSERTED","POSITIVE"),
    ("H03","передала","Лариса","Михаилу","ASSERTED","POSITIVE"),
    ("H04","привёз","Денис","Ксении","ASSERTED","POSITIVE"),
    ("H05","направила","Юлия","Артёму","ASSERTED","POSITIVE"),
    ("H06","выдал","Глеб","Оксане","ASSERTED","POSITIVE"),
    ("H07","отправила","Дарья","Виктору","NEGATED","REJECTED"),
    ("H08","переслал","Роман","Светлане","NEGATED","REJECTED"),
    ("H09","передала","Лариса","Михаилу","NEGATED","REJECTED"),
    ("H10","привёз","Денис","Ксении","NEGATED","REJECTED"),
    ("H11","направила","Юлия","Артёму","NEGATED","REJECTED"),
    ("H12","выдал","Глеб","Оксане","NEGATED","REJECTED"),
    ("H13","отправила","Дарья","Виктору","NEGATED","REJECTED"),
    ("H14","переслал","Роман","Светлане","NEGATED","REJECTED"),
    ("H15","передала","Лариса","Михаилу","NEGATED","REJECTED"),
    ("H16","привёз","Денис","Ксении","NEGATED","REJECTED"),
    ("H17","направила","Юлия","Артёму","NEGATED","REJECTED"),
    ("H18","выдал","Глеб","Оксане","NEGATED","REJECTED"),
    ("H19","сказала","Дарья",None,"ASSERTED","NONE"),
    ("H20","процитировал","Роман",None,"ASSERTED","NONE"),
    ("H21","написала","Лариса",None,"ASSERTED","NONE"),
    ("H22","повторил","Денис",None,"ASSERTED","NONE"),
    ("H23","прочла","Юлия",None,"ASSERTED","NONE"),
    ("H24","заметил","Глеб",None,"ASSERTED","NONE"),
)
def held_gold24():
    assert len(GOLD24)==len(SOURCE_ROWS)==24
    rows=tuple(ManualHeld54(*r) for r in GOLD24)
    assert [r.row_id for r in rows]==[r[0] for r in SOURCE_ROWS]
    assert [sum(x.recipient_polarity==c for x in rows)
            for c in ("POSITIVE","REJECTED","NONE")]==[6,12,6]
    return rows
def gold_sha256():
    return hashlib.sha256(json.dumps(GOLD24,ensure_ascii=False,
        sort_keys=True,separators=(",",":")).encode("utf-8")).hexdigest()
