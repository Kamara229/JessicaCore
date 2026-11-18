"""S63 distinct manual NEW32 gold, separately precommitted before S63 TRAIN.

Do not import during original 43M CPU, goldblind BPE capture, old280
fitting, fixed forward/thresholds. All reference words are S18 raw forms.
This is researcher-annotated challenge, not representative external EVAL.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,json
from research.nra3_s20.s63_new32_raw_only_precommitted import SOURCE_ROWS
@dataclass(frozen=True)
class Manual:
    row_id:str
    main_predicate:str
    transferred_recipient:str|None
    event_factuality:str
    directed_transfer_polarity:str
GOLD=(
 ("P01","вручил","ревизору","ASSERTED","ASSERTED"),
 ("P02","передал","секретарю","ASSERTED","ASSERTED"),
 ("P03","направил","заказчику","ASSERTED","ASSERTED"),
 ("P04","выдал","оператору","ASSERTED","ASSERTED"),
 ("P05","предоставил","инспектору","ASSERTED","ASSERTED"),
 ("P06","отправил","заведующей","ASSERTED","ASSERTED"),
 ("P07","показала","директору","ASSERTED","ASSERTED"),
 ("P08","переслал","архивариусу","ASSERTED","ASSERTED"),
 ("P09","передал","начальнику","NEGATED","REJECTED"),
 ("P10","вручила","контролёру","NEGATED","REJECTED"),
 ("P11","направил","ревизору","NEGATED","REJECTED"),
 ("P12","выдал","водителю","NEGATED","REJECTED"),
 ("P13","предоставил","директору","NEGATED","REJECTED"),
 ("P14","отправил","помощнику","NEGATED","REJECTED"),
 ("P15","сообщил","менеджеру","NEGATED","REJECTED"),
 ("P16","показала","ревизору","NEGATED","REJECTED"),
 ("P17","записал",None,"ASSERTED","NONE"),
 ("P18","повторил",None,"ASSERTED","NONE"),
 ("P19","прочитал",None,"ASSERTED","NONE"),
 ("P20","процитировала",None,"ASSERTED","NONE"),
 ("P21","произнёс",None,"ASSERTED","NONE"),
 ("P22","зачитал",None,"ASSERTED","NONE"),
 ("P23","отметил",None,"ASSERTED","NONE"),
 ("P24","записала",None,"ASSERTED","NONE"),
 ("P25","отправила","инженеру","ASSERTED","ASSERTED"),
 ("P26","вручил","ревизору","ASSERTED","ASSERTED"),
 ("P27","передал","водителю","ASSERTED","ASSERTED"),
 ("P28","выдал","контролёру","ASSERTED","ASSERTED"),
 ("P29","отправила","инспектору","ASSERTED","ASSERTED"),
 ("P30","предоставил","директору","ASSERTED","ASSERTED"),
 ("P31","показала","водителю","ASSERTED","ASSERTED"),
 ("P32","направил","эксперту","ASSERTED","ASSERTED"),
)
def held_gold32():
    assert len(GOLD)==len(SOURCE_ROWS)==32
    assert [g[0] for g in GOLD]==[r[0] for r in SOURCE_ROWS]
    assert sum(g[2] is not None for g in GOLD)==24
    assert sum(g[4]=="REJECTED" for g in GOLD)==8
    return tuple(Manual(*z) for z in GOLD)
def gold_sha256():
    return hashlib.sha256(json.dumps(GOLD,ensure_ascii=False,
           sort_keys=True,separators=(",",":")).encode()).hexdigest()
