"""S54 new frozen author-held quote/recipient24 source; plain-text only.

First committed BEFORE S54 recipient-rescue algorithm and original 43M
capture. Do not use for training, threshold choice, feature selection,
or modifying candidate based on results. The examples are author-designed
near-template paired variants; first-read held-out from S53 training but
NOT a representative independent real-world quality benchmark.
"""
from __future__ import annotations
import hashlib,json
SOURCE_ROWS=(
    ("H01","positive","Дарья отправила заявку Виктору."),
    ("H02","positive","Роман переслал файл Светлане."),
    ("H03","positive","Лариса передала конверт Михаилу."),
    ("H04","positive","Денис привёз заказ Ксении."),
    ("H05","positive","Юлия направила письмо Артёму."),
    ("H06","positive","Глеб выдал справку Оксане."),
    ("H07","negative_post","Дарья не отправила заявку Виктору."),
    ("H08","negative_post","Роман не переслал файл Светлане."),
    ("H09","negative_post","Лариса не передала конверт Михаилу."),
    ("H10","negative_post","Денис не привёз заказ Ксении."),
    ("H11","negative_post","Юлия не направила письмо Артёму."),
    ("H12","negative_post","Глеб не выдал справку Оксане."),
    ("H13","negative_pre","Виктору Дарья не отправила заявку."),
    ("H14","negative_pre","Светлане Роман не переслал файл."),
    ("H15","negative_pre","Михаилу Лариса не передала конверт."),
    ("H16","negative_pre","Ксении Денис не привёз заказ."),
    ("H17","negative_pre","Артёму Юлия не направила письмо."),
    ("H18","negative_pre","Оксане Глеб не выдал справку."),
    ("H19","quoted_report","Дарья сказала «Виктору не отправили заявку» вслух."),
    ("H20","quoted_report","Роман процитировал “Светлане переслали файл” утром."),
    ("H21","quoted_report","Лариса написала „Михаилу не передали конверт“ в чате."),
    ("H22","quoted_report","Денис повторил ‘Ксении привезли заказ’ сегодня."),
    ("H23","quoted_report","Юлия прочла \"Артёму не направили письмо\" вслух."),
    ("H24","quoted_report","Глеб заметил «Оксане выдали справку» поздно."),
)
def text_only_inputs():
    rows=SOURCE_ROWS
    assert len(rows)==24 and len(set(x[2] for x in rows))==24
    assert [x[0] for x in rows]==[f"H{i:02d}" for i in range(1,25)]
    assert [sum(r[1]==name for r in rows) for name in (
        "positive","negative_post","negative_pre","quoted_report")]==[6,6,6,6]
    return tuple(r[2] for r in rows)
def text_sha256():
    return hashlib.sha256(json.dumps(SOURCE_ROWS,ensure_ascii=False,
                  sort_keys=True,separators=(",",":")).encode("utf-8")).hexdigest()
