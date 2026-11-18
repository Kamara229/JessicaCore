"""S57 NEW HELD24 manual event/predicate/recipient gold, separately frozen.

The manually tagged main predicate and outer transferred recipient are
never supplied to S57 forward/feature construction. Gold ONLY imported
after original 43M, S26, S54, S56 and NEW S57 candidate predictions
have been computed and frozen in the first HELD evaluation script.

One external transfer-recipient link per positive/negated-transfer
sentence. Negative dative and quoted inner claims have NO outer
transfer-recipient link. Author-designed small precommitted HELD.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,json
from research.nra3_s20.s57_frozen_new_held24_plain import SOURCE_ROWS

@dataclass(frozen=True)
class Gold57:
    row_id:str
    main_predicate:str
    main_agent:str|None
    outer_recipient:str|None
    outer_factuality:str
    directed_recipient_polarity:str

GOLD_ROWS=(
    ("R01", "вручил", "архивариус", "председательнице", "NEGATED", "REJECTED"),
    ("R02", "передан", "сотрудником", "учителю", "NEGATED", "REJECTED"),
    ("R03", "направил", "управляющий", "инженеру", "NEGATED", "REJECTED"),
    ("R04", "сообщили", None, "Заявительнице", "NEGATED", "REJECTED"),
    ("R05", "показала", "секретарь", "адвокату", "NEGATED", "REJECTED"),
    ("R06", "доставили", None, "представителю", "NEGATED", "REJECTED"),
    ("R07", "отправила", "администратор", "директору", "ASSERTED", "POSITIVE"),
    ("R08", "передал", "он", "консьержке", "ASSERTED", "POSITIVE"),
    ("R09", "выдала", "Наставница", "студентке", "ASSERTED", "POSITIVE"),
    ("R10", "доставили", None, "Клиенту", "ASSERTED", "POSITIVE"),
    ("R11", "вручил", "Михаил", "соседке", "ASSERTED", "POSITIVE"),
    ("R12", "передал", "помощник", "эксперту", "ASSERTED", "POSITIVE"),
    ("R13", "нравился", None, None, "NEGATED", "NONE"),
    ("R14", "заходила", "помощница", None, "NEGATED", "NONE"),
    ("R15", "верил", "Врач", None, "NEGATED", "NONE"),
    ("R16", "готовил", "инженер", None, "NEGATED", "NONE"),
    ("R17", "присутствовал", "бухгалтер", None, "NEGATED", "NONE"),
    ("R18", "помогал", "Юрист", None, "NEGATED", "NONE"),
    ("R19", "сказала", "Библиотекарь", None, "ASSERTED", "NONE"),
    ("R20", "записал", "Сотрудник", None, "ASSERTED", "NONE"),
    ("R21", "повторил", "Докладчик", None, "ASSERTED", "NONE"),
    ("R22", "процитировал", "Репортёр", None, "ASSERTED", "NONE"),
    ("R23", "прочёл", "Начальник", None, "ASSERTED", "NONE"),
    ("R24", "добавил", "Журналист", None, "ASSERTED", "NONE"),
)

def held_gold24():
    assert len(SOURCE_ROWS)==len(GOLD_ROWS)==24
    val=tuple(Gold57(*row) for row in GOLD_ROWS)
    assert [r.row_id for r in val]==[x[0] for x in SOURCE_ROWS]
    assert [sum(x.directed_recipient_polarity==v for x in val)
            for v in ("REJECTED","POSITIVE","NONE")]==[6,6,12]
    assert all((x.outer_recipient is None)==
               (x.directed_recipient_polarity=="NONE") for x in val)
    return val

def gold_sha256():
    return hashlib.sha256(json.dumps(GOLD_ROWS,ensure_ascii=False,
        sort_keys=True,separators=(",",":")).encode("utf-8")).hexdigest()
