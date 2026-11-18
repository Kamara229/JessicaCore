"""S53 sealed raw Russian negation / quote 24-case source, TRAIN-development audit.

Manual plain source committed before S53 true-original Calibri43M capture
or graph evaluator exists. Exactly 24 author-designed short Russian
sentence contrasts; NOT a random independent EVAL. All data are used
only for research-only read-only CPU diagnostics (no gradient).
No protected S38, previously examined S47, or S48 HELD26 imported.
No labels here: safe to load raw source into original 43M/GT tokenizer.
"""
from __future__ import annotations
import hashlib
import json

STAGE="NRA3_S53_PRECOMMITTED_NEW24_PLAINTEXT_ONLY"
SOURCE_ROWS=(
    ("N01","positive","Нина отправила отчёт Павлу."),
    ("N02","positive","Игорь передал ключи Марине."),
    ("N03","positive","Тамара вручила документ Сергею."),
    ("N04","positive","Пётр доставил посылку Алине."),
    ("N05","positive","Ольга показала схему Ивану."),
    ("N06","positive","Борис сообщил решение Анне."),
    ("N07","negated_postposed","Нина не отправила отчёт Павлу."),
    ("N08","negated_postposed","Игорь не передал ключи Марине."),
    ("N09","negated_postposed","Тамара не вручила документ Сергею."),
    ("N10","negated_postposed","Пётр не доставил посылку Алине."),
    ("N11","negated_postposed","Ольга не показала схему Ивану."),
    ("N12","negated_postposed","Борис не сообщил решение Анне."),
    ("N13","negated_preposed","Павлу Нина не отправила отчёт."),
    ("N14","negated_preposed","Марине Игорь не передал ключи."),
    ("N15","negated_preposed","Сергею Тамара не вручила документ."),
    ("N16","negated_preposed","Алине Пётр не доставил посылку."),
    ("N17","negated_preposed","Ивану Ольга не показала схему."),
    ("N18","negated_preposed","Анне Борис не сообщил решение."),
    ("N19","quoted_report","Нина произнесла «Павлу не отправили отчёт» вслух."),
    ("N20","quoted_report","Игорь написал “Марине передали ключи” на доске."),
    ("N21","quoted_report","Тамара отметила „Сергею не вручили документ“ в журнале."),
    ("N22","quoted_report","Пётр прочитал ‘Алине доставили посылку’ в сообщении."),
    ("N23","quoted_report","Ольга сказала \"Ивану не показали схему\" вчера."),
    ("N24","quoted_report","Борис добавил «Анне сообщили решение» в протокол."),
)

def text_only_inputs():
    assert len(SOURCE_ROWS)==24
    assert len(set(row[0] for row in SOURCE_ROWS))==24
    assert len(set(row[2] for row in SOURCE_ROWS))==24
    assert all(row[0]==f"N{i:02d}" for i,row in enumerate(SOURCE_ROWS,1))
    assert [sum(r[1]==c for r in SOURCE_ROWS) for c in
            ("positive","negated_postposed","negated_preposed","quoted_report")]==[6,6,6,6]
    return tuple(row[2] for row in SOURCE_ROWS)

def text_sha256():
    return hashlib.sha256(json.dumps(SOURCE_ROWS,
        ensure_ascii=False,sort_keys=True,separators=(",",":")).encode("utf-8")).hexdigest()

if __name__=="__main__":
    print(json.dumps({"stage":STAGE,"count":len(text_only_inputs()),
        "sha256":text_sha256()},sort_keys=True))
