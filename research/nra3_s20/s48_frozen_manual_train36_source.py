"""S48 separately PRECOMMITTED manually marked TRAIN-only quote examples.

Frozen before any S48 parser implementation or held scoring. The manual
⟦...⟧ markup assigns gold character spans independently of the quote
scanner and S39/S46 q[...,0] feature. No S47 previously audited texts
may be imported or used to tune; S38 protected challenge remains sealed.
S48 labels are for TRAIN loss only, never inference or model.forward.
This is a synthetic author-built training set, not general Russian data.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json

STAGE="NRA3_S48_SEPARATELY_COMMITTED_MANUAL_QUOTE_TRAIN36"
OPEN="⟦"
CLOSE="⟧"

@dataclass(frozen=True)
class ManualQuote:
    row_id:str
    category:str
    text:str
    gold_char_spans:tuple[tuple[int,int],...]

MANUAL_TRAIN_MARKUP=(
    ("T01", "guillemet", "Мастер отметил «⟦измерение завершено⟧» и закрыл журнал."),
    ("T02", "curly_double", "Тамара написала “⟦нужно обновить чертёж⟧” на листе."),
    ("T03", "german", "На стене висело „⟦Проверка окончена⟧“ без подписи."),
    ("T04", "ascii_double", "Павел ответил \"⟦заказ принят⟧\" и улыбнулся."),
    ("T05", "ascii_single", "Слово '⟦монтаж⟧' выделили курсивом."),
    ("T06", "curly_single", "Эксперт назвал ‘⟦важную оговорку⟧’ отдельно."),
    ("T07", "nested_curly", "Ирина сообщила «⟦мастер сказал ⟧“⟦переделать крышу⟧”⟦ вчера⟧»."),
    ("T08", "nested_german", "Автор добавил «⟦встречается термин ⟧„⟦учёт работ⟧“⟦ в отчёте⟧»."),
    ("T09", "nested_single", "Курьер заметил “⟦внутри есть ⟧‘⟦срочное письмо⟧’⟦ и номер⟧”."),
    ("T10", "nested_ascii", "В списке было «⟦надпись ⟧\"⟦нельзя входить⟧\"⟦ на двери⟧»."),
    ("T11", "two_quotes", "Мария прочла «⟦начало⟧» и «⟦окончание⟧» без паузы."),
    ("T12", "two_ascii", "Он сказал \"⟦да⟧\" и позже \"⟦нет⟧\" в трубку."),
    ("T13", "unclosed_guillemet", "В черновике осталось «важное объявление без завершения."),
    ("T14", "unclosed_curly", "Судья прочёл “неполную цитату без завершения."),
    ("T15", "unclosed_ascii", "В файле осталось \"одно незавершённое сообщение."),
    ("T16", "unclosed_ascii_single", "В комментарии напечатали 'незавершённую заметку."),
    ("T17", "stray_closer", "Николай заметил » на свободной строке."),
    ("T18", "mismatch", "Системный журнал сохранил «сбой процесса” без пары."),
    ("T19", "apostrophe", "В справочнике встретилось O'Brien в качестве фамилии."),
    ("T20", "apostrophe_two", "Название Rock'n'Roll оставили без кавычек."),
    ("T21", "escaped_double", "Оператор набрал \\\"точно\\\" и сохранил код."),
    ("T22", "escaped_single", "Разработчик записал \\'false\\' и продолжил."),
    ("T23", "two_mixed", "На плакате было „⟦всё готово⟧“ и «⟦подписи нет⟧»."),
    ("T24", "orphan_after_closed", "Сначала указали «⟦верно⟧», а затем «сбой без конца."),
    ("T25", "unclosed_inner", "Олег записал «текст “без внутреннего закрытия» в журнал."),
    ("T26", "escaped_and_real", "Михаил набрал \\\"нет\\\", а вслух сказал «⟦да⟧»."),
    ("T27", "line_break", "В заметках указали «⟦первая глава\nвторая глава⟧»."),
    ("T28", "punct_inside", "Редактор выделил «⟦что, где?⟧» с пометкой."),
    ("T29", "adjacent", "В книге стояло «⟦слева⟧»«⟦справа⟧» рядом."),
    ("T30", "quote_at_start", "«⟦Проверка пройдена⟧» — написано на двери."),
    ("T31", "quote_at_end", "Ответ в конце сообщения: “⟦счёт оплачен⟧”"),
    ("T32", "quote_negation", "Не было проблем, хотя заявили «⟦ничего не выдали⟧»."),
    ("T33", "nested_double_ascii", "В отчёте написали “⟦слово ⟧'⟦важно⟧'⟦ встречается⟧”."),
    ("T34", "simple_curly_with_apostrophe", "Слово O'Brien заключили в «⟦замечание редактора⟧»."),
    ("T35", "escaped_before_real", "Программист набрал \\\"один\\\" и прочёл “⟦два⟧”."),
    ("T36", "punct_colon", "Инженер зачитал: „⟦кран закрыт; насос выключен⟧“."),
)

def remove_gold_markers(marked):
    chars=[]
    spans=[]
    opened=None
    for c in marked:
        if c==OPEN:
            if opened is not None: raise ValueError("nested gold markers forbidden")
            opened=len(chars)
        elif c==CLOSE:
            if opened is None or len(chars)==opened:
                raise ValueError("orphan/empty gold marker")
            spans.append((opened,len(chars)))
            opened=None
        else:
            chars.append(c)
    if opened is not None: raise ValueError("missing manual gold close")
    return "".join(chars),tuple(spans)

def train36():
    rows=[]
    for id_,category,marked in MANUAL_TRAIN_MARKUP:
        text,spans=remove_gold_markers(marked)
        rows.append(ManualQuote(id_,category,text,spans))
    assert len(rows)==36 and len({r.row_id for r in rows})==36
    assert len({r.text for r in rows})==36
    assert all(r.row_id.startswith("T") for r in rows)
    return tuple(rows)

def digest(rows=None):
    rows=rows if rows is not None else train36()
    data=[(r.row_id,r.category,r.text,r.gold_char_spans) for r in rows]
    return hashlib.sha256(json.dumps(data,sort_keys=True,ensure_ascii=False,
        separators=(",",":")).encode("utf-8")).hexdigest()

if __name__=="__main__":
    print(json.dumps({"stage":STAGE,"count":len(train36()),"sha256":digest()},
        sort_keys=True,ensure_ascii=False))
