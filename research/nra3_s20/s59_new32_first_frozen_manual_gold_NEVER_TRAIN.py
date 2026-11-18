"""S59 precommitted first NEW32 manual gold, separate from raw-only capture.

No imports of this module may occur in original 43M+GT CPU capture,
old152 semantic TRAIN, old S26/S57, or S59 frozen first-pass prediction.
Import this gold only AFTER all first-read new32 predictions are frozen.
Synthetic author-built contrast, not population-independent EVAL.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,json
from research.nra3_s20.s59_new32_first_frozen_raw_only_NEVER_TRAIN import SOURCE_ROWS

@dataclass(frozen=True)
class Manual:
    row_id:str
    main_predicate:str
    transferred_recipient:str|None
    event_factuality:str
    directed_transfer_polarity:str

GOLD=(
 ("K01","передали","охраннику","NEGATED","REJECTED"),
 ("K02","вручён","ревизору","NEGATED","REJECTED"),
 ("K03","отправили","проектировщику","NEGATED","REJECTED"),
 ("K04","сообщили","заказчику","NEGATED","REJECTED"),
 ("K05","выдала","машинисту","NEGATED","REJECTED"),
 ("K06","предоставили","партнёру","NEGATED","REJECTED"),
 ("K07","показали","аудитору","NEGATED","REJECTED"),
 ("K08","передал","новичку","NEGATED","REJECTED"),
 ("K09","переслал","директору","ASSERTED","ASSERTED"),
 ("K10","выдал","исследовательнице","ASSERTED","ASSERTED"),
 ("K11","сообщила","инженеру","ASSERTED","ASSERTED"),
 ("K12","показал","сметчику","ASSERTED","ASSERTED"),
 ("K13","направил","переводчику","ASSERTED","ASSERTED"),
 ("K14","вручил","адвокату","ASSERTED","ASSERTED"),
 ("K15","доставил","заведующей","ASSERTED","ASSERTED"),
 ("K16","отправил","оператору","ASSERTED","ASSERTED"),
 ("K17","хватало",None,"NEGATED","NONE"),
 ("K18","обращались",None,"NEGATED","NONE"),
 ("K19","оборудовали",None,"NEGATED","NONE"),
 ("K20","понравился",None,"NEGATED","NONE"),
 ("K21","удалось",None,"NEGATED","NONE"),
 ("K22","проводили",None,"NEGATED","NONE"),
 ("K23","было",None,"NEGATED","NONE"),
 ("K24","подходил",None,"NEGATED","NONE"),
 ("K25","произнёс",None,"ASSERTED","NONE"),
 ("K26","повторила",None,"ASSERTED","NONE"),
 ("K27","отметил",None,"ASSERTED","NONE"),
 ("K28","зачитал",None,"ASSERTED","NONE"),
 ("K29","произнесла",None,"ASSERTED","NONE"),
 ("K30","записал",None,"ASSERTED","NONE"),
 ("K31","процитировал",None,"ASSERTED","NONE"),
 ("K32","прочитала",None,"ASSERTED","NONE"),
)
def held_gold32():
    assert len(GOLD)==len(SOURCE_ROWS)==32
    assert [g[0] for g in GOLD]==[r[0] for r in SOURCE_ROWS]
    assert sum(g[2] is not None for g in GOLD)==16
    assert sum(g[4]=="REJECTED" for g in GOLD)==8
    return tuple(Manual(*g) for g in GOLD)
def gold_sha256():
    return hashlib.sha256(json.dumps(GOLD,ensure_ascii=False,
          sort_keys=True,separators=(",",":")).encode("utf-8")).hexdigest()
