"""S55 freeze a genuinely more diverse structural-shift HELD24 RAW text source.

Six negated complex clauses incl passive and preposed recipients, six
affirmative main transfers in contexts with irrelevant negation, six
negative non-recipient/indirect case/prepositional confounds, six
reported/quoted inner transfers that are NOT externally asserted.

Precommitted BEFORE S55 43M capture or evaluator implementation.
Researcher-authored and NOT an independent real-world benchmark.
FIRST-READ ONCE: NEVER FIT S54 weights or select thresholds after reading
these items. Protected S38/S47, spent S48 HELD26 and EVAL128 unused.
"""
from __future__ import annotations
import hashlib
import json

STAGE="NRA3_S55_STRUCTURAL_SHIFT_HELD24_PRECOMMITTED_RAW_ONLY"
SOURCE_ROWS=(
    ("P01","negated_main_complex","После долгого совещания бухгалтер не отправил отчёт директору, хотя обещал сделать это вчера."),
    ("P02","negated_main_complex","Директору в понедельник секретарь всё-таки не передала подписанный документ."),
    ("P03","negated_main_complex","Пока инженеры проверяли чертежи, Иван не показал готовую схему Ольге."),
    ("P04","negated_main_complex","Письмо Анне так и не было вручено курьером к вечеру."),
    ("P05","negated_main_complex","При проверке заявки решение клиенту не сообщалось никем."),
    ("P06","negated_main_complex","Марии договор не передали, несмотря на срочную просьбу руководителя."),
    ("P07","affirmative_with_other_negation","Хотя начальник не согласился с условиями, инженер передал проект заказчику."),
    ("P08","affirmative_with_other_negation","Заявку Светлане отправили после того, как проверка не выявила ошибок."),
    ("P09","affirmative_with_other_negation","Татьяна не ответила на звонок, но всё-таки передала ключи соседу."),
    ("P10","affirmative_with_other_negation","Отчёт директору был направлен вовремя, хотя исполнитель не успел на встречу."),
    ("P11","affirmative_with_other_negation","Менеджер доставил посылку получателю, не проверив заранее адрес."),
    ("P12","affirmative_with_other_negation","Получателю документы переданы после завершения проверки, несмотря на возражения."),
    ("P13","negated_dative_nonrecipient","К Ивану курьер не заходил, хотя письмо осталось у секретаря."),
    ("P14","negated_dative_nonrecipient","Мы не подошли к Марине из-за закрытой двери."),
    ("P15","negated_dative_nonrecipient","Сергей не доверял Ольге, несмотря на долгую совместную работу."),
    ("P16","negated_dative_nonrecipient","Для Анны директор не приготовил никаких документов."),
    ("P17","negated_dative_nonrecipient","По делу клиента судья не направлял никаких писем."),
    ("P18","negated_dative_nonrecipient","Катя не могла говорить об Олеге после собрания."),
    ("P19","reported_quote_no_assertion","Секретарь сказала: «Клиенту не отправили письмо», — и закрыла журнал."),
    ("P20","reported_quote_no_assertion","В отчёте написано “Договор Марине не передан”, но фактов проверки нет."),
    ("P21","reported_quote_no_assertion","Руководитель повторил „Алине уже выдали справку“, не подтверждая сообщение."),
    ("P22","reported_quote_no_assertion","Эксперт процитировал ‘Заказчику не направили проект’ на заседании."),
    ("P23","reported_quote_no_assertion","Директор прочитал \"Клиенту отправили письмо\" и ничего не ответил."),
    ("P24","reported_quote_no_assertion","Она добавила «Ивану ничего не передавали» и вышла."),
)

def text_only_inputs():
    rows=SOURCE_ROWS
    assert len(rows)==24 and len({r[0] for r in rows})==24
    assert [r[0] for r in rows]==[f"P{i:02d}" for i in range(1,25)]
    assert len({r[2] for r in rows})==24
    assert [sum(r[1]==tag for r in rows) for tag in (
      "negated_main_complex","affirmative_with_other_negation",
      "negated_dative_nonrecipient","reported_quote_no_assertion",
    )]==[6,6,6,6]
    assert sum("не " in text for _,_,text in rows)>=14
    assert any("«" in text for _,_,text in rows)
    return tuple(r[2] for r in rows)

def text_sha256():
    return hashlib.sha256(json.dumps(
        SOURCE_ROWS,ensure_ascii=False,sort_keys=True,
        separators=(",",":")).encode("utf-8")).hexdigest()
