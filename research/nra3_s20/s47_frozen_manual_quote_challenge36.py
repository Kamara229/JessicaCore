"""S47 PRECOMMITTED source-only handwritten adversarial quote boundary challenge.

This source is the ONLY source of S47 quote-span gold. Gold comes from
independently written ⟦...⟧ spans, NOT from S39's _inside_quotes,
S46 raw q[...,0], a language model, or a downstream model prediction.
Remove markup to obtain user-visible plain Unicode Russian text.
Designed after S46; it is a FROZEN ADVERSARIAL DIAGNOSTIC, NOT a
representative independent random sample or final acceptance EVAL.
S38 protected 32 are not imported, accessed or scored.
The scope policy is deliberate: incomplete/mismatched quotes do not
establish reliable quoted content, so unmatched tails have NO gold spans.
Escaped \\" markers are literal characters, not quotation delimiters.
No training or model selection on these S47 examples is permitted.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json

STAGE="NRA3_S47_SEPARATELY_PRECOMMITTED_MANUAL_QUOTE_BOUNDARY_CHALLENGE_36"
MARKUP_START="⟦"
MARKUP_END="⟧"

@dataclass(frozen=True)
class QuoteChallenge:
    case_id:str
    category:str
    text:str
    gold_character_spans:tuple[tuple[int,int],...]

# Handwritten labels/markup, NOT derived from the model's quote parser.
# A marker covers quoted CONTENT, never delimiter quotation marks.
MANUAL_MARKED_CASES=(
    ("Q01", "guillemet", "Артём прочёл запись «⟦совещание перенесено на пятницу⟧» и улыбнулся."),
    ("Q02", "curly_double", "На витрине написали “⟦открыто до полудня⟧”, хотя было темно."),
    ("Q03", "ascii_double", "Секретарь произнёс \"⟦письмо готово⟧\" и вышел."),
    ("Q04", "ascii_single", "Лектор подчеркнул '⟦важное уточнение⟧' на доске."),
    ("Q05", "german_style", "В протоколе стояло „⟦вопрос закрыт⟧“ без пояснений."),
    ("Q06", "curly_single", "Рецензент вставил ‘⟦редакция согласна⟧’ в комментарий."),
    ("Q07", "nested_curly", "В статье приводят «⟦Олег сказал: ⟧“⟦ответ завтра⟧”⟦ утром⟧» и спорят."),
    ("Q08", "nested_ascii", "Учёный напомнил «⟦тезис ⟧\"⟦доказательство⟧\"⟦ остаётся спорным⟧»."),
    ("Q09", "nested_mixed", "В блокноте было “⟦сначала ⟧‘⟦проверить⟧’⟦ детали⟧” рядом с датой."),
    ("Q10", "nested_german", "Автор процитировал «⟦обозначено как ⟧„⟦чужое письмо⟧“⟦ в каталоге⟧»."),
    ("Q11", "two_quotes", "Диспетчер повторил «⟦рейс отменён⟧» и «⟦путь закрыт⟧»."),
    ("Q12", "two_ascii", "Михаил написал \"⟦да⟧\", затем написал \"⟦нет⟧\" без подписи."),
    ("Q13", "adjacent_quotes", "Редактор соединил «⟦слева⟧»«⟦справа⟧» на одном листе."),
    ("Q14", "at_start", "«⟦Быстрая проверка завершена⟧» — говорится в инструкции."),
    ("Q15", "at_end", "Заказчик отметил в журнале: «⟦проверить схему завтра⟧»"),
    ("Q16", "punct_inside", "Библиотекарь увидел «⟦стоп, не открывать!⟧» и остановился."),
    ("Q17", "quote_vs_real_negation", "Эксперт не подтвердил факт, но процитировал «⟦не передавать документы⟧»."),
    ("Q18", "quoted_negative_pronoun", "Мария сказала «⟦никому нельзя⟧», но отправила бумаги курьеру."),
    ("Q19", "em_dash", "Пожарный записал: «⟦выход свободен — проходите⟧»; затем позвонил."),
    ("Q20", "line_break", "Стенографист сохранил «⟦первая строка\nвторая строка⟧» в документе."),
    ("Q21", "embedded_punctuation", "Куратор выделил “⟦два слова (проверить)⟧” на экране."),
    ("Q22", "nest_quote_around_comma", "Фраза «⟦сначала проверить, ⟧“⟦потом подписать⟧”⟦ и отправить⟧» осталась."),
    ("Q23", "unclosed_guillemet", "В конце листа напечатали «важное сообщение без закрывающей кавычки."),
    ("Q24", "unclosed_ascii", "В конце записи осталась \"незаконченная реплика без знака."),
    ("Q25", "unmatched_closer", "Участник заметил неожиданное » после слова проверка."),
    ("Q26", "mismatched_pair", "Сотрудник видел «ошибка в цитате” и пропустил строку."),
    ("Q27", "orphan_after_closed", "Ранее было «⟦согласовано⟧», затем появилась «оборванная запись."),
    ("Q28", "escaped_literal", "Павел ввёл \\\"не готово\\\" в терминале и нажал Enter."),
    ("Q29", "apostrophe_name", "В списке нашлось имя O'Brien без цитат."),
    ("Q30", "unmatched_single", "В документе встретилась одинокая ' перед словом тестирования."),
    ("Q31", "plain_no_quote", "Нина не отправила письмо получателю, но сохранила копию."),
    ("Q32", "mixed_outer_separate", "Выпуск включал „⟦факт первый⟧“ и «⟦факт второй⟧» без деталей."),
    ("Q33", "unclosed_curly", "Редактор записал “обрывок фразы без закрытия и ушёл."),
    ("Q34", "unicode_typographic_double", "На обложке оставили “⟦тихий вечер⟧” и рисунок."),
    ("Q35", "quote_words_after", "Внутри «⟦плана согласования⟧» не было даты, но слово согласование встречалось позже."),
    ("Q36", "escaped_and_real", "Он ввёл \\\"пусто\\\", а затем произнёс «⟦готово⟧» при свидетелях."),
)

def unmark(source):
    """Pure markup stripping; returns text plus manual character spans."""
    result=[]
    spans=[]
    opened=None
    for c in source:
        if c==MARKUP_START:
            if opened is not None: raise ValueError("nested manual gold markup")
            opened=len(result)
        elif c==MARKUP_END:
            if opened is None or opened==len(result):
                raise ValueError("empty or unmatched manual gold markup")
            spans.append((opened,len(result)))
            opened=None
        else:
            result.append(c)
    if opened is not None:
        raise ValueError("unclosed manual gold markup")
    return "".join(result),tuple(spans)

def challenge36():
    rows=[]
    for case_id,category,marked in MANUAL_MARKED_CASES:
        text,spans=unmark(marked)
        if not text.strip() or not case_id.startswith("Q"):
            raise ValueError("invalid frozen S47 manual challenge")
        rows.append(QuoteChallenge(case_id,category,text,spans))
    if len(rows)!=36 or len({r.case_id for r in rows})!=36 or len({r.text for r in rows})!=36:
        raise ValueError("S47 must contain exactly 36 disjoint manually marked rows")
    if not any(not r.gold_character_spans for r in rows):
        raise ValueError("S47 missing negative quote spans")
    if not any(len(r.gold_character_spans)>1 for r in rows):
        raise ValueError("S47 missing nested and multi-quote phenomena")
    return tuple(rows)

def sealed_source_digest():
    rows=challenge36()
    payload=[(r.case_id,r.category,r.text,r.gold_character_spans) for r in rows]
    return hashlib.sha256(json.dumps(payload,ensure_ascii=False,
        sort_keys=True,separators=(",",":")).encode("utf-8")).hexdigest()

if __name__=="__main__":
    print(json.dumps({"stage":STAGE,"count":len(challenge36()),
        "manual_gold_digest":sealed_source_digest()},
        sort_keys=True,ensure_ascii=False))
