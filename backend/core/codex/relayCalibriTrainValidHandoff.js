/**
 * Provider-free, no-test acknowledgement of a precise CALIBRI TRAIN/VALID
 * diagnostic. It neither launches a Jessica test nor changes research data.
 */
export function calibriTrainValidNoTestHandoff(message) {
  if (typeof message !== "string" || !message.trim() || message.length > 1200) return null;
  const lines = message.trim().split(/\r?\n/u).map(s => s.trim()).filter(Boolean);
  const marker = /^JESSICA_REPLY_([\da-f]{8}-[\da-f]{4}-[\da-f]{4}-[\da-f]{4}-[\da-f]{12})$/iu.exec(lines[0] || "");
  let body;
  if (marker) {
    if (lines.length !== 3 || lines[2] !== "JESSICA_END_" + marker[1]) return null;
    body = lines[1];
  } else {
    if (lines.length !== 1 || /JESSICA_(?:REPLY|END)_/iu.test(message)) return null;
    body = lines[0];
  }
  if (!body || body.length > 240) return null;
  if (!/^Следующий\s+шаг\s*:\s*на\s+TRAIN\s*\/\s*VALID\s+собрать\s+ошибки\s+BoolQ\s*:/iu.test(body)) return null;
  if (!/\bID\b/u.test(body) || !/правильный\s+ответ/iu.test(body) || !/CB59\s*\/\s*CB88/iu.test(body)) return null;
  if (!/уверенность\s+и\s+расхождения/iu.test(body) || !/Определить\s+причины\s+деградации/iu.test(body)) return null;
  if (!/Замороженный\s+TEST\s+и\s+рабочую\s+Jessica\s+не\s+менять\.?$/iu.test(body)) return null;
  if (/(?:запусти|выполни|отправь|проведи|повтори)\s+(?:новый\s+)?(?:тест|задач)|(?:deploy|merge|установи|удали|секрет|парол|API[- ]?ключ)/iu.test(body)) return null;
  return "Поняла: на TRAIN/VALID нужно сравнить ошибки BoolQ у CB59 и CB88, включая уверенность и расхождения. Продолжай анализ через исследовательскую ветку. Замороженный TEST и рабочую Jessica не меняем; тест Jessica не запускаю.";
}
