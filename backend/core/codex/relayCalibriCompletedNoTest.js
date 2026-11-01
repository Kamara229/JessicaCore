/**
 * Interpret a FINISHED CALIBRI measurement/status as a safe no-test
 * continuation, not a request to run a Jessica benchmark.
 *
 * Android normally strips JESSICA_REPLY/END markers; both original
 * matching envelope and extracted body are supported. This function
 * never verifies GitHub claims, runs tools, writes memories or accepts
 * a task; it only acknowledges and requests continuing research.
 */
export function completedCalibriNoTestHandoff(message) {
  if (typeof message !== "string" || !message.trim() || message.length > 2500) return null;
  const lines = message.split(/\r?\n/u).map(s => s.trim()).filter(Boolean);
  if (!lines.length) return null;
  const match = /^JESSICA_REPLY_([\da-f]{8}-[\da-f]{4}-[\da-f]{4}-[\da-f]{4}-[\da-f]{12})$/iu.exec(lines[0]);
  let raw;
  if (match) {
    if (lines.length < 3 || lines.at(-1) !== "JESSICA_END_" + match[1]) return null;
    raw = lines.slice(1, -1).join(" ");
  } else {
    if (/JESSICA_(?:REPLY|END)_/iu.test(message)) return null;
    raw = lines.join(" ");
  }
  const body = raw.replace(/\s+/gu, " ").trim();
  if (!body || body.length > 480 || /[«»]/u.test(body)) return null;
  const project = /^CALIBRI\s+v(\d{1,3})\s*[,.;:—–-]\s*PR\s*#(\d{1,6})\s*:/iu.exec(body);
  if (!project || Number(project[1]) < 23) return null;
  if (!/(?:\bPPL\b|\bNLL\b|бит\s*\/\s*вес|\b\d[\d\s]*\s*Б\b|весов|матриц|\bCI\b)/iu.test(body)) return null;
  if (!/(?:Следующ(?:ий|им)\s+(?:шаг|тест)|Далее\s*[—–:-])/iu.test(body)) return null;
  // "Тест Jessica не нужен" and "Jessica не нужна" signify a no-test
  // research continuation, including when the next research step is
  // described as a "тест" (e.g. split TRAIN windows).
  const noJessicaTest =
    /Тест\s+(?:(?:в\s+(?:обычном\s+)?чате\s+)?Jessica\s+)?не\s+(?:нужен|требуется|запрошен)|Jessica\s+не\s+нужна/iu.test(body);
  if (!noJessicaTest) return null;
  if (/(?:выполни|запусти|проведи|повтори|отправь)\s+(?:новый\s+)?(?:тест|задач)/iu.test(body)) return null;
  // An explicit instruction to execute a Jessica test or access secrets
  // must use the existing strict command path (or human review).
  if (/(?:выполни|запусти|отправь|проведи|повтори)\s+(?:новый\s+)?(?:тест\b|задач[уа]\b|в\s+обычн)|(?:парол|секрет|API[- ]?ключ|токен|удали|удалить|merge|deploy|разверни|установи)/iu.test(body)) return null;
  return "Поняла: CALIBRI v" + project[1] + ", PR #" + project[2] +
    " — передан исследовательский статус, а не результат теста Jessica. " +
    "Продолжай следующий этап через GitHub; тест в обычном чате Jessica не запускаю.";
}
