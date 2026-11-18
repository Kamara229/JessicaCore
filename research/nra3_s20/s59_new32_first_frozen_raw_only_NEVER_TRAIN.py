"""S59 fresh author-created challenge32 source ONLY. No manual labels here.

Precommitted BEFORE first S59 semantic native320 model TRAIN/EVAL actions.
Four 8-case strata, all unique and disjoint from old inspected groups.
No split tuning or model selection on these NEW32 texts. Not a random
population benchmark; FIRST-READ results become permanently SPENT.
"""
from __future__ import annotations
import hashlib,json
STAGE="NRA3_S59_PREREG_RAW_NEW32_BEFORE_NATIVE_SEMANTIC_PAID2046"
FAMILIES=(
 "negated_transfer_event",
 "asserted_transfer_negation_elsewhere",
 "nontransfer_dative_confound",
 "quotation_report_not_asserted",
)
SOURCE_ROWS=(
 ("K01","negated_transfer_event","После ночной проверки охраннику не передали ключ, хотя список посетителей уже утвердили."),
 ("K02","negated_transfer_event","Отчёт по расходам не был вручён ревизору до начала совещания."),
 ("K03","negated_transfer_event","Только после замены схемы проектировщику не отправили новую спецификацию."),
 ("K04","negated_transfer_event","Пока оператор вызывал мастера, заказчику не сообщили итоговую стоимость."),
 ("K05","negated_transfer_event","Диспетчер так и не выдала машинисту исправленный маршрут."),
 ("K06","negated_transfer_event","Поскольку директор отсутствовал, партнёру не предоставили копию приложения."),
 ("K07","negated_transfer_event","Результаты измерений не показали аудитору после повторного замера."),
 ("K08","negated_transfer_event","Инструктор не передал новичку пропуск, несмотря на письмо отдела."),
 ("K09","asserted_transfer_negation_elsewhere","Хотя техник не прочитал замечание, секретарь переслал директору итоговый акт."),
 ("K10","asserted_transfer_negation_elsewhere","Не закончив проверку, лаборант выдал исследовательнице протокол."),
 ("K11","asserted_transfer_negation_elsewhere","Когда связист не ответил, дежурная сообщила инженеру код."),
 ("K12","asserted_transfer_negation_elsewhere","Прораб не задержался и показал сметчику чертёж."),
 ("K13","asserted_transfer_negation_elsewhere","Редактор не спорил, а направил переводчику новую версию."),
 ("K14","asserted_transfer_negation_elsewhere","Несмотря на то что судья не согласился, помощник вручил адвокату копию."),
 ("K15","asserted_transfer_negation_elsewhere","Водитель не подписал ведомость, но доставил заведующей коробку."),
 ("K16","asserted_transfer_negation_elsewhere","Старший не проверил номера и отправил оператору инструкцию."),
 ("K17","nontransfer_dative_confound","Наблюдателю не хватало подтверждений после проверки."),
 ("K18","nontransfer_dative_confound","К ревизору не обращались во время обеда."),
 ("K19","nontransfer_dative_confound","Для секретаря не оборудовали рабочее место."),
 ("K20","nontransfer_dative_confound","Ученице не понравился предложенный маршрут."),
 ("K21","nontransfer_dative_confound","Управляющему не удалось закончить смету к пятнице."),
 ("K22","nontransfer_dative_confound","По распоряжению инженера не проводили повторное испытание."),
 ("K23","nontransfer_dative_confound","Оператору не было понятно решение комиссии."),
 ("K24","nontransfer_dative_confound","К проводнику не подходил пассажир после посадки."),
 ("K25","quotation_report_not_asserted","Писатель произнёс «Бригадиру сообщили дату» и закрыл блокнот."),
 ("K26","quotation_report_not_asserted","Модератор повторила “Контролёру не вручили лист” перед аудиторией."),
 ("K27","quotation_report_not_asserted","Редактор отметил „Сотруднику передали пакет“ на полях."),
 ("K28","quotation_report_not_asserted","Преподаватель зачитал ‘Мастеру не показали данные’ перед группой."),
 ("K29","quotation_report_not_asserted","Актриса произнесла «Режиссёру отправили афишу» и улыбнулась."),
 ("K30","quotation_report_not_asserted","Юрист записал «Клиенту не направили ответ» в заметках."),
 ("K31","quotation_report_not_asserted","Докладчик процитировал “Эксперту вручили материалы” в начале выступления."),
 ("K32","quotation_report_not_asserted","Сотрудница прочитала „Начальнику не сообщили время“ и отложила письмо."),
)
def raw_only_inputs():
    assert len(SOURCE_ROWS)==32
    assert len(set(x[2] for x in SOURCE_ROWS))==32
    assert [x[0] for x in SOURCE_ROWS]==[f"K{i:02d}" for i in range(1,33)]
    assert [sum(row[1]==f for row in SOURCE_ROWS) for f in FAMILIES]==[8]*4
    assert all("⟦" not in row[2] and "⟧" not in row[2] for row in SOURCE_ROWS)
    return tuple(x[2] for x in SOURCE_ROWS)
def raw_sha256():
    return hashlib.sha256(json.dumps(SOURCE_ROWS,ensure_ascii=False,
           sort_keys=True,separators=(",",":")).encode("utf-8")).hexdigest()
