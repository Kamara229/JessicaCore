"""S62 *separate* frozen manual NEW32 reference; DO NOT import during
original43M GT native capture, old248 fitting or first new32 forward.

The hand-authored challenge deliberately distinguishes outside quoted
speech, subordinate transfer, passive/rejection and true no-transfer.
No population independent/random evaluation is claimed.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,json
from research.nra3_s20.s62_new32_preregistered_raw_only import SOURCE_ROWS
@dataclass(frozen=True)
class Manual:
    row_id:str
    main_predicate:str
    transferred_recipient:str|None
    event_factuality:str
    directed_transfer_polarity:str
GOLD=(
("N01","передал","секретарю","ASSERTED","ASSERTED"),
("N02","направил","ревизору","ASSERTED","ASSERTED"),
("N03","вручила","руководителю","ASSERTED","ASSERTED"),
("N04","передал","оператору","ASSERTED","ASSERTED"),
("N05","отправил","архивариусу","ASSERTED","ASSERTED"),
("N06","вручила","директору","ASSERTED","ASSERTED"),
("N07","показал","лаборанту","ASSERTED","ASSERTED"),
("N08","переслал","диспетчеру","ASSERTED","ASSERTED"),
("N09","передала","ревизору","ASSERTED","ASSERTED"),
("N10","отправил","координатору","ASSERTED","ASSERTED"),
("N11","вручил","инспектору","ASSERTED","ASSERTED"),
("N12","сообщила","заведующей","ASSERTED","ASSERTED"),
("N13","направил","заказчику","ASSERTED","ASSERTED"),
("N14","отправил","ревизору","ASSERTED","ASSERTED"),
("N15","выдал","клиенту","ASSERTED","ASSERTED"),
("N16","предоставил","заказчику","ASSERTED","ASSERTED"),
("N17","выдан","инженеру","NEGATED","REJECTED"),
("N18","направлена","ревизору","NEGATED","REJECTED"),
("N19","вручено","секретарю","NEGATED","REJECTED"),
("N20","передали","Представителю","NEGATED","REJECTED"),
("N21","отправлена","оператору","NEGATED","REJECTED"),
("N22","показали","Контролёру","NEGATED","REJECTED"),
("N23","предоставили","Архивариусу","NEGATED","REJECTED"),
("N24","доставлена","водителю","NEGATED","REJECTED"),
("N25","обращались",None,"NEGATED","NONE"),
("N26","подготовили",None,"NEGATED","NONE"),
("N27","хватило",None,"NEGATED","NONE"),
("N28","проводили",None,"NEGATED","NONE"),
("N29","понравилось",None,"NEGATED","NONE"),
("N30","подходил",None,"NEGATED","NONE"),
("N31","удалось",None,"NEGATED","NONE"),
("N32","оборудовали",None,"NEGATED","NONE"),
)
def held_gold32():
    assert len(GOLD)==len(SOURCE_ROWS)==32
    assert [g[0] for g in GOLD]==[r[0] for r in SOURCE_ROWS]
    assert sum(g[2] is not None for g in GOLD)==24
    assert sum(g[4]=="REJECTED" for g in GOLD)==8
    return tuple(Manual(*row) for row in GOLD)
def gold_sha256():
    return hashlib.sha256(json.dumps(GOLD,ensure_ascii=False,
             sort_keys=True,separators=(",",":")).encode()).hexdigest()
