"""S56 new structurally diverse 24-case HELD raw Russian corpus.

Frozen BEFORE implementation of S56 architecture or its real original
43M+GT CPU first-read. Strict 6 x (complex negated transfers, asserted
transfers with unrelated negation, negative nontransfer/dative case traps,
quoted attributed actions with no direct outside transfer claim).

Researcher-authored synthetic HELD for a single first-read diagnostic.
Not a random/population-independent EVAL. No S38/S47/S48 HELD26 or
independent EVAL32/EVAL128 may be read.
"""
from __future__ import annotations
import hashlib
import json

STAGE="NRA3_S56_NEW_STRUCTURAL_SCOPE_HELD24_FROZEN_RAW_ONLY"
SOURCE_ROWS=(
    ("Q01", "complex_negated_transfer", "После собрания преподаватель так и не передал ведомость декану, хотя помощник уже подготовил бумаги."),
    ("Q02", "complex_negated_transfer", "Заместителю руководителя вчера не отправили подписанное уведомление, несмотря на требование отдела."),
    ("Q03", "complex_negated_transfer", "К утру протокол комиссии не был передан председателю архивариусом."),
    ("Q04", "complex_negated_transfer", "Пока юристы обсуждали условия, сотрудница не показала клиентке итоговую версию договора."),
    ("Q05", "complex_negated_transfer", "Учителю после проверки домашней работы не выдали обещанный сертификат."),
    ("Q06", "complex_negated_transfer", "Решение по жалобе заявителю до конца месяца так и не сообщили."),
    ("Q07", "asserted_other_negation", "Хотя бухгалтер не получил подпись, курьер доставил папку специалисту."),
    ("Q08", "asserted_other_negation", "Почтальон не нашёл адрес сразу, но утром вручил открытку пенсионерке."),
    ("Q09", "asserted_other_negation", "Аналитики не завершили доклад, однако менеджер переслал таблицу партнёру."),
    ("Q10", "asserted_other_negation", "Студенту уже передали зачётную книжку, хотя экзаменатор не явился."),
    ("Q11", "asserted_other_negation", "Руководитель отправил документы нотариусу, не проверяя остаток приложений."),
    ("Q12", "asserted_other_negation", "Несмотря на то что отдел не согласовал смету, секретарь выдала копию заказчице."),
    ("Q13", "negated_nontransfer_dative", "К соседу водитель не подходил после работы."),
    ("Q14", "negated_nontransfer_dative", "Врач не доверял ассистентке, несмотря на советы коллег."),
    ("Q15", "negated_nontransfer_dative", "Для начальницы бухгалтер не готовил никаких отчётов."),
    ("Q16", "negated_nontransfer_dative", "По поручению директора начальник не связывался с поставщиком."),
    ("Q17", "negated_nontransfer_dative", "Партнёру не нравилось предложение после совещания."),
    ("Q18", "negated_nontransfer_dative", "Сотруднику не удалось дозвониться до клиента из-за сбоя."),
    ("Q19", "quoted_not_asserted", "Инженер заявил: «Проверяющему не показали чертёж», — и продолжил совещание."),
    ("Q20", "quoted_not_asserted", "Адвокат написал “Судье передали заявление” в заметках без проверки."),
    ("Q21", "quoted_not_asserted", "Ведущий повторил „Пациентке не выдали справку“ во время заседания."),
    ("Q22", "quoted_not_asserted", "Репортёр процитировал ‘Получателю доставили письмо’ в эфире."),
    ("Q23", "quoted_not_asserted", "Сотрудник прочитал \"Директору не направили отчёт\" и вздохнул."),
    ("Q24", "quoted_not_asserted", "Участница сказала «Наставнику передали материалы» и замолчала."),
)
FAMILIES=("complex_negated_transfer","asserted_other_negation",
          "negated_nontransfer_dative","quoted_not_asserted")

def text_only_inputs():
    rows=SOURCE_ROWS
    assert len(rows)==24 and len(set(x[2] for x in rows))==24
    assert [r[0] for r in rows]==[f"Q{i:02d}" for i in range(1,25)]
    assert [sum(r[1]==k for r in rows) for k in FAMILIES]==[6,6,6,6]
    assert all("⟦" not in r[2] and "⟧" not in r[2] for r in rows)
    return tuple(r[2] for r in rows)

def text_sha256():
    return hashlib.sha256(json.dumps(SOURCE_ROWS,
        ensure_ascii=False,sort_keys=True,separators=(",",":")).encode("utf-8")).hexdigest()
