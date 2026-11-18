"""S48 precommitted manually marked HELD26 quote structure audit source.

Committed before S48 evaluator, algorithm implementation, or first
model fitting. Rows and manual ⟦...⟧ gold were authored separately
from S48 TRAIN36 rows. They are held frozen from training, selection
and tuning, and are NOT a random external general-language benchmark.
Must never be used for model.fit, prompt selection or gradients.
S47 and S38 protected challenges not imported.
"""
from __future__ import annotations
import hashlib
import json

from research.nra3_s20.s48_frozen_manual_train36_source import (
    ManualQuote, remove_gold_markers,
)

STAGE="NRA3_S48_SEPARATELY_COMMITTED_MANUAL_HELD26_ONLY"
MANUAL_HELD_MARKUP=(
    ("H01", "guillemet", "Дежурный вписал «⟦проверка завершилась⟧» в акт."),
    ("H02", "curly_double", "Татьяна произнесла “⟦передайте образцы завтра⟧” без шуток."),
    ("H03", "german", "В макете выделили „⟦срок выполнения⟧“ красным цветом."),
    ("H04", "curly_single", "Судья отметил ‘⟦существенная деталь⟧’ в тексте."),
    ("H05", "nested_single", "На карточке было “⟦указание ⟧‘⟦не открывать⟧’⟦ выполнено⟧”."),
    ("H06", "nested_german", "Проверяющий сказал «⟦здесь написано ⟧„⟦допуск закрыт⟧“⟦ верно⟧»."),
    ("H07", "ascii_double", "Инспектор переспросил \"⟦готово к отправке⟧\" перед выездом."),
    ("H08", "ascii_single", "Секретарь обвёл '⟦только сегодня⟧' возле даты."),
    ("H09", "name_apostrophe", "Фамилию D'Angelo перенесли в другой столбец."),
    ("H10", "two_apostrophes", "Название l'esprit d'équipe осталось в заголовке."),
    ("H11", "escaped_double", "Разработчик написал \\\"пауза\\\" и продолжил запуск."),
    ("H12", "escaped_single", "Стажёр ввёл \\'запрет\\' в командную строку."),
    ("H13", "unclosed_guillemet", "Журналист оставил «длинный отрывок без окончания."),
    ("H14", "unclosed_curly", "Кладовщик увидел “забытую цитату в тетради."),
    ("H15", "unclosed_ascii", "Покупатель начал \"сообщение без окончания."),
    ("H16", "stray_closer", "В документе напечатали лишнюю » посередине строки."),
    ("H17", "mismatched", "Марк прочёл «текст вопроса” и закрыл папку."),
    ("H18", "orphan_after_closed", "Ранее сказали «⟦правильно⟧», затем «всё оборвалось."),
    ("H19", "nested_unbalanced", "В отчёте осталось «внутри “незаконченная речь»."),
    ("H20", "adjacent", "Два ответа «⟦вход разрешён⟧»«⟦выход свободен⟧» появились рядом."),
    ("H21", "different_sequential", "Дизайнер подписал „⟦версия первая⟧“ и “⟦версия вторая⟧” в файле."),
    ("H22", "quote_with_negation", "Она отказала в доставке, но прочла «⟦не выдавать лист⟧»."),
    ("H23", "multiline", "В протоколе «⟦первый адрес\nвторой адрес⟧» записали одной строкой."),
    ("H24", "escaped_plus_real", "Он напечатал \\\"условно\\\" и затем сказал ‘⟦можно⟧’ вслух."),
    ("H25", "quote_at_start", "“⟦Подписать сегодня⟧” — такое требование встретилось в письме."),
    ("H26", "quote_at_end", "На коробке указано: „⟦не кантовать⟧“"),
)

def held26():
    data=[]
    for name,category,marked in MANUAL_HELD_MARKUP:
        text,spans=remove_gold_markers(marked)
        data.append(ManualQuote(name,category,text,spans))
    assert len(data)==26 and len({r.row_id for r in data})==26
    assert len({r.text for r in data})==26
    assert all(r.row_id.startswith("H") for r in data)
    return tuple(data)

def digest(rows=None):
    rows=rows if rows is not None else held26()
    data=[(r.row_id,r.category,r.text,r.gold_char_spans) for r in rows]
    return hashlib.sha256(json.dumps(data,ensure_ascii=False,
        sort_keys=True,separators=(",",":")).encode("utf-8")).hexdigest()

if __name__=="__main__":
    print(json.dumps({"stage":STAGE,"held":len(held26()),"digest":digest()},
        sort_keys=True,ensure_ascii=False))
