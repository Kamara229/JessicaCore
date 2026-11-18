"""S60 separately pinned human gold; NEVER import before first-read S60 forward.

Annotated exact raw surface word form. This 32 is researcher-built not
population random evaluation; after first score becomes permanently SPENT.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,json
from research.nra3_s20.s60_new32_precommitted_raw_only import SOURCE_ROWS
@dataclass(frozen=True)
class Gold:
    row_id:str
    main_predicate:str
    transferred_recipient:str|None
    event_factuality:str
    directed_transfer_polarity:str
GOLD=(
 ("L01","выдан","кладовщику","NEGATED","REJECTED"),
 ("L02","передана","экспедитору","NEGATED","REJECTED"),
 ("L03","направлено","контролёру","NEGATED","REJECTED"),
 ("L04","показан","инженеру","NEGATED","REJECTED"),
 ("L05","вручены","лаборанту","NEGATED","REJECTED"),
 ("L06","предоставили","переводчику","NEGATED","REJECTED"),
 ("L07","отправили","секретарю","NEGATED","REJECTED"),
 ("L08","доставили","заказчику","NEGATED","REJECTED"),
 ("L09","передал","мастеру","ASSERTED","ASSERTED"),
 ("L10","выдала","заведующей","ASSERTED","ASSERTED"),
 ("L11","сообщил","бригадиру","ASSERTED","ASSERTED"),
 ("L12","переслал","автору","ASSERTED","ASSERTED"),
 ("L13","вручила","клиенту","ASSERTED","ASSERTED"),
 ("L14","показал","начальнику","ASSERTED","ASSERTED"),
 ("L15","направила","водителю","ASSERTED","ASSERTED"),
 ("L16","передал","ревизору","ASSERTED","ASSERTED"),
 ("L17","хватило",None,"NEGATED","NONE"),
 ("L18","обратились",None,"NEGATED","NONE"),
 ("L19","предусмотрели",None,"NEGATED","NONE"),
 ("L20","понравилась",None,"NEGATED","NONE"),
 ("L21","проводили",None,"NEGATED","NONE"),
 ("L22","удалось",None,"NEGATED","NONE"),
 ("L23","подошли",None,"NEGATED","NONE"),
 ("L24","стало",None,"NEGATED","NONE"),
 ("L25","произнёс",None,"ASSERTED","NONE"),
 ("L26","записала",None,"ASSERTED","NONE"),
 ("L27","прочитал",None,"ASSERTED","NONE"),
 ("L28","повторил",None,"ASSERTED","NONE"),
 ("L29","процитировала",None,"ASSERTED","NONE"),
 ("L30","зачитал",None,"ASSERTED","NONE"),
 ("L31","отметил",None,"ASSERTED","NONE"),
 ("L32","произнесла",None,"ASSERTED","NONE"),
)
def held_gold32():
    assert len(GOLD)==len(SOURCE_ROWS)==32
    assert [g[0] for g in GOLD]==[r[0] for r in SOURCE_ROWS]
    assert sum(g[2] is not None for g in GOLD)==16
    assert sum(g[4]=="REJECTED" for g in GOLD)==8
    return tuple(Gold(*g) for g in GOLD)
def gold_sha256():
    return hashlib.sha256(json.dumps(GOLD,ensure_ascii=False,
        separators=(",",":"),sort_keys=True).encode()).hexdigest()
