"""S57 new frozen first-read structural HELD24 raw texts.

Authored and git committed BEFORE S57 event-first model architecture,
training, real43M native capture or manual evaluation. Four predefined
groups, six each: negated transferred event in complex/passive clauses,
positive transferred event with negation elsewhere, negative dative/
prepositional nontransfer confounds, and nonasserted reported quote.

This is a small author-generated challenge, NOT general/population EVAL.
Neither held text nor manual gold is available in S57 TRAIN process;
spent previously scored S53..S56 only. No protected S38, S47, spent
S48 HELD26, or independent EVAL128/32 accessed.
"""
from __future__ import annotations
import hashlib,json

STAGE="NRA3_S57_FIRST_EVENTFIRST_STRUCTURAL_HELD24_PLAINTEXT"
FAMILIES=("negated_main_complex","asserted_other_negation",
          "negative_nontransfer_dative","quoted_report")
SOURCE_ROWS=(
    ("Z01", "negated_main_complex", "Когда комиссия завершила заседание, секретарь всё же не выдала протокол инспектору."),
    ("Z02", "negated_main_complex", "Наблюдателю после проверки оборудования не отправили итоговое заключение, хотя эксперт его подписал."),
    ("Z03", "negated_main_complex", "После замены деталей инженеру не был передан обновлённый чертёж мастером."),
    ("Z04", "negated_main_complex", "Пока судья оглашал решение, пристав не вручил постановление представителю."),
    ("Z05", "negated_main_complex", "Библиотекарю акт приёма не передавали до завершения инвентаризации."),
    ("Z06", "negated_main_complex", "Вчера заведующая не показала отчёт проверяющей, хотя документы уже были готовы."),
    ("Z07", "asserted_other_negation", "Поскольку водитель не успел к открытию, документы доставила архивисту сотрудница."),
    ("Z08", "asserted_other_negation", "Регистратор передал талон студентке, не проверив расписание."),
    ("Z09", "asserted_other_negation", "Несмотря на то что редактор не одобрил статью, редакции предоставили копию рукописи."),
    ("Z10", "asserted_other_negation", "Акт подрядчику всё-таки отправили после того, как директор не подписал приложение."),
    ("Z11", "asserted_other_negation", "Техник не стал спорить и отправил код оператору."),
    ("Z12", "asserted_other_negation", "Из-за того что никто не ответил, менеджер передал сообщение заказчику."),
    ("Z13", "negative_nontransfer_dative", "Экскурсовод не доверял стажёру во время экскурсии."),
    ("Z14", "negative_nontransfer_dative", "К инженеру техник не подходил после аварии."),
    ("Z15", "negative_nontransfer_dative", "Для аудитора бухгалтер не составлял пояснений."),
    ("Z16", "negative_nontransfer_dative", "Учителю не понравилось объяснение ученика."),
    ("Z17", "negative_nontransfer_dative", "По заявлению работника директор не проводил встречу."),
    ("Z18", "negative_nontransfer_dative", "Врач не помогал медсестре при подготовке процедуры."),
    ("Z19", "quoted_report", "Пресс-секретарь заявил: «Инспектору отправили распоряжение», — после окончания встречи."),
    ("Z20", "quoted_report", "Редактор записала “Эксперту не вручили заключение” и убрала блокнот."),
    ("Z21", "quoted_report", "Аналитик процитировал „Поставщику доставили образцы“ перед обсуждением."),
    ("Z22", "quoted_report", "Ведущая повторила ‘Свидетелю не передали повестку’ в эфире."),
    ("Z23", "quoted_report", "Инструктор прочитал «Оператору направили описание» и переключил слайд."),
    ("Z24", "quoted_report", "Юрист добавил «Подрядчику не сообщили сумму» в протокол."),
)
def text_only_inputs():
    assert len(SOURCE_ROWS)==24
    assert [r[0] for r in SOURCE_ROWS]==[f"Z{i:02d}" for i in range(1,25)]
    assert len({r[2] for r in SOURCE_ROWS})==24
    assert [sum(r[1]==c for r in SOURCE_ROWS) for c in FAMILIES]==[6,6,6,6]
    assert all("⟦" not in t and "⟧" not in t for _,_,t in SOURCE_ROWS)
    return tuple(r[2] for r in SOURCE_ROWS)
def text_sha256():
    return hashlib.sha256(json.dumps(SOURCE_ROWS,ensure_ascii=False,
           sort_keys=True,separators=(",",":")).encode("utf-8")).hexdigest()
