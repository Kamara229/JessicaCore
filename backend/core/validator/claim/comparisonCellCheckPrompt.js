/* Keep output compact enough for the comparison validator's quota. */
export function buildComparisonCellCheckPrompt() {
  return [
    "Ты независимый проверяющий сравнительной таблицы по первоисточникам.",
    "Тебе дан перечень КОНКРЕТНЫХ ячеек с ID и их SOURCE 1/SOURCE 2.",
    "Проверяй ТОЛЬКО эти ячейки, не придумывай дополнительных фактов.",
    "Для каждого ID дай одну дословную НЕПРЕРЫВНУЮ цитату из его источника",
    "длиной 5–16 слов, подтверждающую смысл всей ячейки.",
    "Если такой цитаты нет или она не подтверждает смысл, верни пустую строку.",
    "Не выбирай заголовок или copyright-шаблон вместо условия лицензии.",
    "Для требований используй положение с must/shall/retain/include/give.",
    "Для условных обязанностей процитируй также условие if/when.",
    "Не заменяй пропущенные слова многоточием.",
    "Не считай отсутствие упоминания доказательством отсутствия условия.",
    "Только JSON без пояснений: {\"c\":[[1,\"цитата\"],[2,\"\"]]}",
    "КАЖДЫЙ ID должен быть указан ровно один раз."
  ].join("\n");
}
export function buildComparisonCellCheckRequest(task,matrix,sourceText) {
  const lines=matrix.checks.map(check=>
    String(check.id)+". SOURCE "+String(check.source+1)+
    "; КРИТЕРИЙ: "+check.criterion+"; ЯЧЕЙКА: "+check.value);
  return ["ИСХОДНАЯ ЗАДАЧА:",String(task||""),"",
    "ФАКТИЧЕСКИЕ ЯЧЕЙКИ:",...lines,"",
    "ПЕРВОИСТОЧНИКИ:",sourceText].join("\n");
}
