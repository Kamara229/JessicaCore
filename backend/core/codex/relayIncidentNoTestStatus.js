/**
 * Recognize a single precise "backend incident diagnosed / fix in PR / NOT
 * deployed / NO Jessica test" status in the already selected ChatGPT thread.
 *
 * A provider outage must not turn a plain verified no-test progress note into
 * another "interpret_failed" handoff. This module is intentionally NOT a
 * generic interpreter; it neither executes tests nor validates CI claims.
 */
export function codexIncidentNoTestStatus(message) {
  const source = typeof message === "string" ? message : "";
  if (!source || source.length > 1500) return null;
  const lines = source.trim().split(/\r?\n/u);
  let body;
  const marker = /^JESSICA_REPLY_([a-f\d]{8}-[a-f\d]{4}-[a-f\d]{4}-[a-f\d]{4}-[a-f\d]{12})$/iu.exec(lines[0] || "");
  if (marker) {
    if (lines.length !== 3 || lines.at(-1) !== "JESSICA_END_" + marker[1]) return null;
    body = lines[1];
  } else {
    if (lines.length !== 1 || /JESSICA_(?:REPLY|END)_/iu.test(source)) return null;
    body = lines[0];
  }
  if (!body || body.length > 240) return null;
  if (!/^Причина\s*:\s*backend\s+не\s+распознавал\s+формат\s+«Тест\s*:\s*…»\s+и\s+обращался\s+к\s+AI-интерпретатору\./iu.test(body))
    return null;
  if (!/Исправление\s+в\s+PR\s*#\d{1,6}\s*,\s*backend\s+CI\s+пройден\./iu.test(body))
    return null;
  if (!/Render\s+и\s+Supabase\s+проверены\./iu.test(body)) return null;
  if (!/Live\s+не\s+менял\./iu.test(body)) return null;
  if (!/Тест\s+Jessica\s+не\s+запускался\.?\s*$/iu.test(body)) return null;
  if (/(?:запусти|запускать|выполни|повтори|отправь|создай|удали|установи|deploy|merge|пароль|секрет|ключ)/iu.test(body))
    return null;
  return "Поняла: причина сбоя найдена, исправление пока только в исследуемом PR, а рабочая Jessica не менялась. Проверь полный CI и готовность исправления. Тест в обычном чате Jessica пока не запускаю; результата нет.";
}
