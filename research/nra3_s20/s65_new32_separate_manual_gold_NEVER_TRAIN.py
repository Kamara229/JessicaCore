"""S65 separate immutable MANUAL NEW32 gold; NEVER used in OLD312 fitting.

No generated model output may use this before all frozen 43M GT forwards.
Author-written structural challenge, not external or representative eval.
"""
from __future__ import annotations
from dataclasses import dataclass
import json,hashlib
from research.nra3_s20.s65_new32_raw_only_frozen_before_train import SOURCE_ROWS
@dataclass(frozen=True)
class Manual:
    row_id:str
    main_predicate:str
    transferred_recipient:str|None
    event_factuality:str
    directed_transfer_polarity:str
GOLD=(
 ("R01","передала","диспетчеру","ASSERTED","ASSERTED"),
 ("R02","вручил","механику","ASSERTED","ASSERTED"),
 ("R03","направил","бухгалтеру","ASSERTED","ASSERTED"),
 ("R04","выдал","архивариусу","ASSERTED","ASSERTED"),
 ("R05","передал","врачу","ASSERTED","ASSERTED"),
 ("R06","отправила","контролёру","ASSERTED","ASSERTED"),
 ("R07","предоставил","консультанту","ASSERTED","ASSERTED"),
 ("R08","сообщил","заведующей","ASSERTED","ASSERTED"),
 ("R09","подтверждал",None,"NEGATED","NONE"),
 ("R10","повторила",None,"ASSERTED","NONE"),
 ("R11","лежали",None,"ASSERTED","NONE"),
 ("R12","выписал",None,"ASSERTED","NONE"),
 ("R13","нашлось",None,"NEGATED","NONE"),
 ("R14","закончилось",None,"ASSERTED","NONE"),
 ("R15","молчали",None,"ASSERTED","NONE"),
 ("R16","приняли",None,"NEGATED","NONE"),
 ("R17","передан","оператору","NEGATED","REJECTED"),
 ("R18","вручили","ревизору","NEGATED","REJECTED"),
 ("R19","направили","курьеру","NEGATED","REJECTED"),
 ("R20","предоставили","эксперту","NEGATED","REJECTED"),
 ("R21","выдали","секретарю","NEGATED","REJECTED"),
 ("R22","отправил","бухгалтеру","NEGATED","REJECTED"),
 ("R23","переданы","архивариусу","NEGATED","REJECTED"),
 ("R24","показали","Руководителю","NEGATED","REJECTED"),
 ("R25","понравился",None,"NEGATED","NONE"),
 ("R26","удалось",None,"NEGATED","NONE"),
 ("R27","приготовили",None,"NEGATED","NONE"),
 ("R28","подходили",None,"NEGATED","NONE"),
 ("R29","передал","врачу","ASSERTED","ASSERTED"),
 ("R30","отправил","инженеру","ASSERTED","ASSERTED"),
 ("R31","направил","ревизору","ASSERTED","ASSERTED"),
 ("R32","предоставила","бухгалтеру","ASSERTED","ASSERTED"),
)
def held_gold32():
    assert len(GOLD)==len(SOURCE_ROWS)==32
    assert [x[0] for x in GOLD]==[z[0] for z in SOURCE_ROWS]
    assert sum(x[2] is not None for x in GOLD)==20
    assert sum(x[4]=="REJECTED" for x in GOLD)==8
    return tuple(Manual(*x) for x in GOLD)
def gold_sha256():
    return hashlib.sha256(json.dumps(GOLD,ensure_ascii=False,
             sort_keys=True,separators=(",",":")).encode()).hexdigest()
