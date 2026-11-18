"""S66 strictly separate precommitted first NEW32 manual gold.

GOLD must NOT enter CPU real43M GT baseline replay, full/prefix
native hidden parity, source-only tokenizer audit, candidate training,
any frozen S57-S65 prediction, threshold fitting or model selection.
Author-created challenge only; NOT representative population evaluation.
"""
from __future__ import annotations
import json,hashlib
from dataclasses import dataclass
from research.nra3_s20.s66_frozen_new32_raw_only_DO_NOT_TRAIN import SOURCE_ROWS
@dataclass(frozen=True)
class Manual:
    row_id:str
    main_predicate:str
    transferred_recipient:str|None
    event_factuality:str
    directed_transfer_polarity:str
GOLD=(
 ("T01","передать","дежурному","ASSERTED","ASSERTED"),
 ("T02","вручил","бухгалтеру","ASSERTED","ASSERTED"),
 ("T03","направил","руководителю","ASSERTED","ASSERTED"),
 ("T04","показал","инспектору","ASSERTED","ASSERTED"),
 ("T05","отправила","заказчику","ASSERTED","ASSERTED"),
 ("T06","предоставил","эксперту","ASSERTED","ASSERTED"),
 ("T07","сообщил","клиенту","ASSERTED","ASSERTED"),
 ("T08","переслал","ревизору","ASSERTED","ASSERTED"),
 ("T09","вручили","водителю","NEGATED","REJECTED"),
 ("T10","передали","контролёру","NEGATED","REJECTED"),
 ("T11","выдана","оператору","NEGATED","REJECTED"),
 ("T12","сообщили","ему","NEGATED","REJECTED"),
 ("T13","отправили","инспектору","NEGATED","REJECTED"),
 ("T14","показали","Сотруднику","NEGATED","REJECTED"),
 ("T15","предоставили","заведующей","NEGATED","REJECTED"),
 ("T16","доставили","врачу","NEGATED","REJECTED"),
 ("T17","усмехнулся",None,"ASSERTED","NONE"),
 ("T18","снилось",None,"ASSERTED","NONE"),
 ("T19","подтвердил",None,"NEGATED","NONE"),
 ("T20","выключил",None,"ASSERTED","NONE"),
 ("T21","проверил",None,"NEGATED","NONE"),
 ("T22","показалось",None,"ASSERTED","NONE"),
 ("T23","написали",None,"NEGATED","NONE"),
 ("T24","пожал",None,"ASSERTED","NONE"),
 ("T25","передал","врачу","ASSERTED","ASSERTED"),
 ("T26","направила","юристу","ASSERTED","ASSERTED"),
 ("T27","показали","контролёру","ASSERTED","ASSERTED"),
 ("T28","вручил","архивариусу","ASSERTED","ASSERTED"),
 ("T29","направила","покупателю","ASSERTED","ASSERTED"),
 ("T30","отправила","консультанту","ASSERTED","ASSERTED"),
 ("T31","предоставил","ревизору","ASSERTED","ASSERTED"),
 ("T32","передал","бухгалтеру","ASSERTED","ASSERTED"),
)
def held_gold32():
    assert len(GOLD)==len(SOURCE_ROWS)==32
    assert [g[0] for g in GOLD]==[r[0] for r in SOURCE_ROWS]
    assert sum(g[2] is not None for g in GOLD)==24
    assert sum(g[4]=="REJECTED" for g in GOLD)==8
    return tuple(Manual(*x) for x in GOLD)
def gold_sha256():
    return hashlib.sha256(json.dumps(GOLD,ensure_ascii=False,
                   sort_keys=True,separators=(",",":")).encode()).hexdigest()
