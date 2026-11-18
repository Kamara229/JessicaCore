"""S64 NEW32 manually annotated gold, physically preregistered separately.

DO NOT import into source-only Calibri tokenizer / post-GT read, before
all frozen S57..S63 prediction forward, nor in thresholds or optimization.
S64 is an author-created stress set, NOT population/multi-author EVAL.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,json
from research.nra3_s20.s64_new32_raw_only_preregistered import SOURCE_ROWS
@dataclass(frozen=True)
class Manual:
    row_id:str
    main_predicate:str
    transferred_recipient:str|None
    event_factuality:str
    directed_transfer_polarity:str
GOLD=(
("Q01","передала","лаборанту","ASSERTED","ASSERTED"),
("Q02","вручил","инспектору","ASSERTED","ASSERTED"),
("Q03","показал","руководителю","ASSERTED","ASSERTED"),
("Q04","отправила","бухгалтеру","ASSERTED","ASSERTED"),
("Q05","переслал","архивариусу","ASSERTED","ASSERTED"),
("Q06","выдал","оператору","ASSERTED","ASSERTED"),
("Q07","доставила","юристу","ASSERTED","ASSERTED"),
("Q08","сообщил","диспетчеру","ASSERTED","ASSERTED"),
("Q09","передал","контролёру","ASSERTED","ASSERTED"),
("Q10","направил","куратору","ASSERTED","ASSERTED"),
("Q11","вручил","механику","ASSERTED","ASSERTED"),
("Q12","отправила","инспектору","ASSERTED","ASSERTED"),
("Q13","предоставил","эксперту","ASSERTED","ASSERTED"),
("Q14","переслал","бухгалтеру","ASSERTED","ASSERTED"),
("Q15","выдал","оператору","ASSERTED","ASSERTED"),
("Q16","сообщила","директору","ASSERTED","ASSERTED"),
("Q17","вышел",None,"ASSERTED","NONE"),
("Q18","повторил",None,"ASSERTED","NONE"),
("Q19","оставил",None,"ASSERTED","NONE"),
("Q20","засмеялась",None,"ASSERTED","NONE"),
("Q21","подписал",None,"NEGATED","NONE"),
("Q22","замолчал",None,"ASSERTED","NONE"),
("Q23","заметил",None,"ASSERTED","NONE"),
("Q24","добавил",None,"NEGATED","NONE"),
("Q25","отправили","консультанту","NEGATED","REJECTED"),
("Q26","передал","архивисту","ASSERTED","ASSERTED"),
("Q27","вручена","механику","NEGATED","REJECTED"),
("Q28","показали","Ревизору","NEGATED","REJECTED"),
("Q29","помогло",None,"NEGATED","NONE"),
("Q30","хватало",None,"NEGATED","NONE"),
("Q31","подошли",None,"NEGATED","NONE"),
("Q32","подготовили",None,"NEGATED","NONE"),
)
def held_gold32():
    assert len(GOLD)==len(SOURCE_ROWS)==32
    assert [z[0] for z in GOLD]==[z[0] for z in SOURCE_ROWS]
    assert sum(z[2] is not None for z in GOLD)==20
    assert sum(z[4]=="REJECTED" for z in GOLD)==3
    return tuple(Manual(*z) for z in GOLD)
def gold_sha256():
    return hashlib.sha256(json.dumps(GOLD,sort_keys=True,
             ensure_ascii=False,separators=(",",":")).encode()).hexdigest()
