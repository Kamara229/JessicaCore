"""S57 FIRST-READ structural event-first HELD24 RAW texts, independently frozen.

24 manually authored contrasts: complex negated external transfer,
positive transfer despite unrelated negation, negative non-transfer
dative case confounds and reported-only quotes. Source committed BEFORE
S57 classifier, event-first decoder, and physical Calibri43M+GT CPU
first-read. NOT random/population independent assessment. All S53/S54/
S55/S56 examples already spent and available only as TRAIN.
"""
from __future__ import annotations
import hashlib,json

STAGE="NRA3_S57_FROZEN_NEW_FIRST_READ_HELD24_RAW_ONLY"
FAMILIES=("complex_negated_transfer","positive_other_negation",
          "nontransfer_dative","quoted_nonasserted")
SOURCE_ROWS=(
    ("R01", "complex_negated_transfer", "Несмотря на письмо от комиссии, архивариус не вручил председательнице подписанный протокол до пятницы."),
    ("R02", "complex_negated_transfer", "Диплом учителю после церемонии так и не был передан ответственным сотрудником."),
    ("R03", "complex_negated_transfer", "Когда началась проверка, управляющий не направил инженеру исправленный план."),
    ("R04", "complex_negated_transfer", "Заявительнице не сообщили решение, хотя документы сотрудники изучили заранее."),
    ("R05", "complex_negated_transfer", "Пока свидетели обсуждали дело, секретарь не показала адвокату запись."),
    ("R06", "complex_negated_transfer", "Вчера представителю заказчика не доставили новую версию сметы."),
    ("R07", "positive_other_negation", "Хотя исполнитель не подписал акт, администратор отправила копию директору утром."),
    ("R08", "positive_other_negation", "После того как курьер не нашёл подъезд, он передал пакет консьержке."),
    ("R09", "positive_other_negation", "Наставница выдала справку студентке, не сообщая об этом коллегам."),
    ("R10", "positive_other_negation", "Клиенту доставили заказ, несмотря на то что машина не завелась сразу."),
    ("R11", "positive_other_negation", "Михаил не ответил на вопрос, но затем вручил подарок соседке."),
    ("R12", "positive_other_negation", "Хотя комиссия не утвердила решение, помощник передал черновик эксперту."),
    ("R13", "nontransfer_dative", "Мастеру не нравился результат ремонта после осмотра."),
    ("R14", "nontransfer_dative", "К руководителю помощница не заходила с утра."),
    ("R15", "nontransfer_dative", "Врач не верил пациентке, даже после беседы."),
    ("R16", "nontransfer_dative", "Для директора инженер не готовил новых чертежей."),
    ("R17", "nontransfer_dative", "По просьбе клиента бухгалтер не присутствовал на собрании."),
    ("R18", "nontransfer_dative", "Юрист не помогал соседу из-за конфликта."),
    ("R19", "quoted_nonasserted", "Библиотекарь сказала: «Читателю не выдали книгу», — и завершила разговор."),
    ("R20", "quoted_nonasserted", "Сотрудник записал “Покупателю доставили товар” в блокнот без проверки."),
    ("R21", "quoted_nonasserted", "Докладчик повторил „Заявителю не направили письмо“ в конце выступления."),
    ("R22", "quoted_nonasserted", "Репортёр процитировал ‘Декану передали документы’ на пресс-конференции."),
    ("R23", "quoted_nonasserted", "Начальник прочёл \"Клиентке не показали договор\" и закрыл папку."),
    ("R24", "quoted_nonasserted", "Журналист добавил «Преподавателю сообщили результат» и умолк."),
)

def text_only_inputs():
    assert len(SOURCE_ROWS)==24 and len(set(x[2] for x in SOURCE_ROWS))==24
    assert [x[0] for x in SOURCE_ROWS]==[f"R{i:02d}" for i in range(1,25)]
    assert [sum(x[1]==family for x in SOURCE_ROWS)
            for family in FAMILIES]==[6,6,6,6]
    assert all("⟦" not in x[2] and "⟧" not in x[2] for x in SOURCE_ROWS)
    return tuple(x[2] for x in SOURCE_ROWS)

def text_sha256():
    return hashlib.sha256(json.dumps(SOURCE_ROWS,ensure_ascii=False,
       sort_keys=True,separators=(",",":")).encode("utf-8")).hexdigest()
