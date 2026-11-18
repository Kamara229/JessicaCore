"""S55 precommit separate researcher-authored HELD24 directed graph gold.

All labels assigned BEFORE S55 model capture or S54 frozen candidate
inference. Gold enters S55 evaluator AFTER all original 43M/S26/S54
forward, never in original tokenizer or semantic model input.

A directed RECIPIENT here denotes *recipient of a transferred or
communicated object*, NOT any Russian dative object (e.g. 'доверял
Ольге') nor recipient of the statement inside a reported quotation.
Passive/inversion are included; agent may be omitted.
Researcher-specified narrow 24-case structural challenge, NOT unbiased.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from research.nra3_s20.s55_frozen_structural_shift_held24_plain import (
    SOURCE_ROWS,text_sha256,
)

@dataclass(frozen=True)
class S55Gold:
    row_id:str
    main_predicate:str
    main_agent:str|None
    transferred_recipient:str|None
    main_factuality:str
    transfer_polarity:str

GOLD=(
    ("P01","отправил","бухгалтер","директору","NEGATED","REJECTED"),
    ("P02","передала","секретарь","Директору","NEGATED","REJECTED"),
    ("P03","показал","Иван","Ольге","NEGATED","REJECTED"),
    ("P04","вручено","курьером","Анне","NEGATED","REJECTED"),
    ("P05","сообщалось",None,"клиенту","NEGATED","REJECTED"),
    ("P06","передали",None,"Марии","NEGATED","REJECTED"),
    ("P07","передал","инженер","заказчику","ASSERTED","POSITIVE"),
    ("P08","отправили",None,"Светлане","ASSERTED","POSITIVE"),
    ("P09","передала","Татьяна","соседу","ASSERTED","POSITIVE"),
    ("P10","направлен",None,"директору","ASSERTED","POSITIVE"),
    ("P11","доставил","Менеджер","получателю","ASSERTED","POSITIVE"),
    ("P12","переданы",None,"Получателю","ASSERTED","POSITIVE"),
    ("P13","заходил","курьер",None,"NEGATED","NONE"),
    ("P14","подошли","Мы",None,"NEGATED","NONE"),
    ("P15","доверял","Сергей",None,"NEGATED","NONE"),
    ("P16","приготовил","директор",None,"NEGATED","NONE"),
    ("P17","направлял","судья",None,"NEGATED","NONE"),
    ("P18","могла","Катя",None,"NEGATED","NONE"),
    ("P19","сказала","Секретарь",None,"ASSERTED","NONE"),
    ("P20","написано",None,None,"ASSERTED","NONE"),
    ("P21","повторил","Руководитель",None,"ASSERTED","NONE"),
    ("P22","процитировал","Эксперт",None,"ASSERTED","NONE"),
    ("P23","прочитал","Директор",None,"ASSERTED","NONE"),
    ("P24","добавила","Она",None,"ASSERTED","NONE"),
)

def held_gold24():
    assert len(GOLD)==len(SOURCE_ROWS)==24
    rows=tuple(S55Gold(*r) for r in GOLD)
    assert [r.row_id for r in rows]==[x[0] for x in SOURCE_ROWS]
    assert [sum(r.transfer_polarity==s for r in rows)
            for s in ("REJECTED","POSITIVE","NONE")]==[6,6,12]
    assert all((r.transferred_recipient is None)==(r.transfer_polarity=="NONE")
               for r in rows)
    assert [sum(r.main_factuality==s for r in rows)
            for s in ("NEGATED","ASSERTED")]==[12,12]
    return rows

def gold_sha256():
    return hashlib.sha256(json.dumps(GOLD,ensure_ascii=False,
        sort_keys=True,separators=(",",":")).encode("utf-8")).hexdigest()
