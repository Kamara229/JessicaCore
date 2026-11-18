"""S61 precommitted separate author MANUAL gold, never imported before forward.

Exact S18 word surface forms. Main asserted predicate is annotated rather
than every event; M25 is a deliberate outer actual recipient + embedded
quoted transfer, which should count outer recipient, not quoted recipient.
Tiny author-written challenge, not independent population benchmark.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,json
from research.nra3_s20.s61_new32_raw_only_preregistered import SOURCE_ROWS
@dataclass(frozen=True)
class Gold:
    row_id:str
    main_predicate:str
    transferred_recipient:str|None
    event_factuality:str
    directed_transfer_polarity:str
GOLD=(
 ("M01","выслан","ревизору","NEGATED","REJECTED"),
 ("M02","передали","инженеру","NEGATED","REJECTED"),
 ("M03","вручена","архивариусу","NEGATED","REJECTED"),
 ("M04","сообщили","Секретарю","NEGATED","REJECTED"),
 ("M05","направили","Поставщику","NEGATED","REJECTED"),
 ("M06","предоставили","Аудитору","NEGATED","REJECTED"),
 ("M07","показали","Технику","NEGATED","REJECTED"),
 ("M08","доставлен","представителю","NEGATED","REJECTED"),
 ("M09","передала","водителю","ASSERTED","ASSERTED"),
 ("M10","направил","инспектору","ASSERTED","ASSERTED"),
 ("M11","передал","ревизору","ASSERTED","ASSERTED"),
 ("M12","отправил","редактору","ASSERTED","ASSERTED"),
 ("M13","показал","врачу","ASSERTED","ASSERTED"),
 ("M14","выдал","контролёру","ASSERTED","ASSERTED"),
 ("M15","сообщил","технику","ASSERTED","ASSERTED"),
 ("M16","доставил","консультанту","ASSERTED","ASSERTED"),
 ("M17","хватило",None,"NEGATED","NONE"),
 ("M18","обращались",None,"NEGATED","NONE"),
 ("M19","предусмотрели",None,"NEGATED","NONE"),
 ("M20","понравилось",None,"NEGATED","NONE"),
 ("M21","проводили",None,"NEGATED","NONE"),
 ("M22","удалось",None,"NEGATED","NONE"),
 ("M23","подошёл",None,"NEGATED","NONE"),
 ("M24","стало",None,"NEGATED","NONE"),
 ("M25","передал","редактору","ASSERTED","ASSERTED"),
 ("M26","процитировал",None,"ASSERTED","NONE"),
 ("M27","повторила",None,"ASSERTED","NONE"),
 ("M28","зачитал",None,"ASSERTED","NONE"),
 ("M29","произнесла",None,"ASSERTED","NONE"),
 ("M30","отметил",None,"ASSERTED","NONE"),
 ("M31","записала",None,"ASSERTED","NONE"),
 ("M32","повторил",None,"ASSERTED","NONE"),
)
def held_gold32():
    assert len(GOLD)==len(SOURCE_ROWS)==32
    assert [x[0] for x in GOLD]==[x[0] for x in SOURCE_ROWS]
    assert sum(x[2] is not None for x in GOLD)==17
    assert sum(x[4]=="REJECTED" for x in GOLD)==8
    return tuple(Gold(*x) for x in GOLD)
def gold_sha256():
    return hashlib.sha256(json.dumps(GOLD,ensure_ascii=False,
             sort_keys=True,separators=(",",":")).encode()).hexdigest()
