"""S58 separately precommitted manual GOLD for NEW32 researcher-built corpus.

Do not import from any TRAIN, feature extraction, native43M CPU prefill,
model selection, or first S58 frozen model forward script. First read
allowed ONLY after all frozen candidate predictions have been committed.
Corpus is synthetic/challenging, NOT unbiased real-world EVAL.
S58 result on these rows cannot be reused as independent HELD again.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,json
from research.nra3_s20.s58_new32_preregistered_raw_only_do_not_train import SOURCE_ROWS

@dataclass(frozen=True)
class Manual:
    row_id: str
    main_predicate: str
    transferred_recipient: str|None
    event_factuality: str
    directed_transfer_polarity: str

# Annotated exact RAW surface-token FORMS, not generated BPE/token IDs.
# REJECTED is a semantic negated transfer with a recipient argument;
# quoted/report-only sentences have no asserted outer transfer.
GOLD=(
  ("H01","передали","курьеру","NEGATED","REJECTED"),
  ("H02","вручено","Представителю","NEGATED","REJECTED"),
  ("H03","отправила","технику","NEGATED","REJECTED"),
  ("H04","доставили","Архивариусу","NEGATED","REJECTED"),
  ("H05","выдали","переводчику","NEGATED","REJECTED"),
  ("H06","показали","участнику","NEGATED","REJECTED"),
  ("H07","переслали","инженеру","NEGATED","REJECTED"),
  ("H08","предоставил","инспектору","NEGATED","REJECTED"),
  ("H09","передал","бригадиру","ASSERTED","ASSERTED"),
  ("H10","направил","координатору","ASSERTED","ASSERTED"),
  ("H11","выдала","проводнику","ASSERTED","ASSERTED"),
  ("H12","показала","клиенту","ASSERTED","ASSERTED"),
  ("H13","отправила","исследователю","ASSERTED","ASSERTED"),
  ("H14","вручил","консультанту","ASSERTED","ASSERTED"),
  ("H15","передал","сотруднице","ASSERTED","ASSERTED"),
  ("H16","предоставил","подрядчику","ASSERTED","ASSERTED"),
  ("H17","понравились",None,"NEGATED","NONE"),
  ("H18","обращались",None,"NEGATED","NONE"),
  ("H19","подготовили",None,"NEGATED","NONE"),
  ("H20","хватало",None,"NEGATED","NONE"),
  ("H21","проводили",None,"NEGATED","NONE"),
  ("H22","удалось",None,"NEGATED","NONE"),
  ("H23","было",None,"NEGATED","NONE"),
  ("H24","подошли",None,"NEGATED","NONE"),
  ("H25","прочитала",None,"ASSERTED","NONE"),
  ("H26","записал",None,"ASSERTED","NONE"),
  ("H27","процитировала",None,"ASSERTED","NONE"),
  ("H28","повторил",None,"ASSERTED","NONE"),
  ("H29","зачитал",None,"ASSERTED","NONE"),
  ("H30","записала",None,"ASSERTED","NONE"),
  ("H31","показал",None,"ASSERTED","NONE"),
  ("H32","произнёс",None,"ASSERTED","NONE"),
)

def held_gold32():
    assert len(GOLD)==len(SOURCE_ROWS)==32
    assert [g[0] for g in GOLD]==[r[0] for r in SOURCE_ROWS]
    assert sum(g[2] is not None for g in GOLD)==16
    assert sum(g[4]=="REJECTED" for g in GOLD)==8
    assert sum(g[3]=="NEGATED" for g in GOLD)==16
    return tuple(Manual(*g) for g in GOLD)

def gold_sha256():
    return hashlib.sha256(json.dumps(GOLD,ensure_ascii=False,sort_keys=True,
             separators=(",",":")).encode("utf-8")).hexdigest()
