"""S58 preregistered NEW raw-only author-built challenge32 (UNSCORED).

This file contains ONLY raw input strings, no role/event labels.
Committed on the S58 research branch BEFORE any new S58 TRAIN fit
or first-read S58 model scoring. New candidate development must NOT
read, inspect or use these texts as TRAIN or dev. A separately committed
gold file is required before any first-read evaluation. Precommitting
raw text alone does not certify independent sampling or generalization.

Original S38/S47/S48 and independent EVAL32/128 remain inaccessible.
"""
from __future__ import annotations
import hashlib,json
STAGE="NRA3_S58_PREREG_NEW_RAW_ONLY_CHALLENGE32_NO_GOLD"
FAMILIES=(
  "negative_actual_transfer_complex",
  "positive_transfer_different_negation",
  "dative_nontransfer_false_recipient",
  "reported_quote_not_asserted",
)
SOURCE_ROWS=(
  ("H01","negative_actual_transfer_complex","После сверки реестра курьеру так и не передали опечатанный пакет, хотя накладную уже составили."),
  ("H02","negative_actual_transfer_complex","Представителю подрядчика не было вручено окончательное уведомление перед закрытием смены."),
  ("H03","negative_actual_transfer_complex","Пока мастер осматривал детали, помощница не отправила технику исправленную ведомость."),
  ("H04","negative_actual_transfer_complex","Архивариусу так и не доставили копию журнала после переноса заседания."),
  ("H05","negative_actual_transfer_complex","Подготовленное письмо не выдали переводчику, несмотря на подпись заведующего."),
  ("H06","negative_actual_transfer_complex","Вопреки обещанию организатора участнику не показали новый план рассадки."),
  ("H07","negative_actual_transfer_complex","Заявку инженеру не переслали до проверки оборудования второй комиссией."),
  ("H08","negative_actual_transfer_complex","Наблюдатель не предоставил инспектору снимки, пока секретарь уточняла время."),
  ("H09","positive_transfer_different_negation","Хотя эксперт не согласился с расчётом, техник передал бригадиру новую инструкцию."),
  ("H10","positive_transfer_different_negation","Дежурный не стал объяснять причину и направил координатору код доступа."),
  ("H11","positive_transfer_different_negation","Пока водитель не выходил из машины, диспетчер выдала проводнику маршрутный лист."),
  ("H12","positive_transfer_different_negation","Несмотря на то что переводчик не успел закончить текст, редактор показала клиенту черновик."),
  ("H13","positive_transfer_different_negation","Никто не возражал, когда лаборантка отправила исследователю таблицу результатов."),
  ("H14","positive_transfer_different_negation","После того как начальник не принял отчёт, оператор вручил консультанту квитанцию."),
  ("H15","positive_transfer_different_negation","Стажёр не открыл вложение, но передал сотруднице ключ от кабинета."),
  ("H16","positive_transfer_different_negation","Пока смена не завершилась, контролёр предоставил подрядчику акт осмотра."),
  ("H17","dative_nontransfer_false_recipient","Наставнику не понравились поправки к проекту."),
  ("H18","dative_nontransfer_false_recipient","К консультанту не обращались во время совещания."),
  ("H19","dative_nontransfer_false_recipient","Для переводчика не подготовили отдельное рабочее место."),
  ("H20","dative_nontransfer_false_recipient","Аудитору не хватало исходных данных для проверки."),
  ("H21","dative_nontransfer_false_recipient","По просьбе диспетчера не проводили повторный инструктаж."),
  ("H22","dative_nontransfer_false_recipient","Медсестре не удалось завершить осмотр вовремя."),
  ("H23","dative_nontransfer_false_recipient","Оператору не было известно о новом расписании."),
  ("H24","dative_nontransfer_false_recipient","К мастеру не подошли после объявления перерыва."),
  ("H25","reported_quote_not_asserted","Заведующая прочитала «Помощнику передали черновой отчёт» и закрыла папку."),
  ("H26","reported_quote_not_asserted","Стенографист записал: «Редактору не отправили рукопись», — затем выключил диктофон."),
  ("H27","reported_quote_not_asserted","Сотрудница процитировала “Инженеру выдали пропуск” во время обсуждения."),
  ("H28","reported_quote_not_asserted","Ведущий повторил „Участнику не вручили билет“ перед перерывом."),
  ("H29","reported_quote_not_asserted","Инспектор зачитал ‘Координатору доставили документы’ и отложил лист."),
  ("H30","reported_quote_not_asserted","Юрист записала «Клиенту не передали экземпляр договора» в протокол."),
  ("H31","reported_quote_not_asserted","Аналитик показал надпись «Проводнику сообщили время» на экране."),
  ("H32","reported_quote_not_asserted","Преподаватель произнёс «Слушателю не выдали сертификат» и сменил тему."),
)
def raw_only_inputs():
    assert len(SOURCE_ROWS)==32
    assert [x[0] for x in SOURCE_ROWS]==[f"H{i:02d}" for i in range(1,33)]
    assert len(set(x[2] for x in SOURCE_ROWS))==32
    assert [sum(x[1]==f for x in SOURCE_ROWS) for f in FAMILIES]==[8]*4
    assert all("⟦" not in row[2] and "⟧" not in row[2] for row in SOURCE_ROWS)
    return tuple(r[2] for r in SOURCE_ROWS)
def raw_sha256():
    return hashlib.sha256(json.dumps(SOURCE_ROWS,ensure_ascii=False,
            separators=(",",":"),sort_keys=True).encode("utf-8")).hexdigest()
