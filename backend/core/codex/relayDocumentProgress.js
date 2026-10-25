/** Route an explicit long-document follow-up without consuming interpreter quota. */
export function documentAnalysisNextStep(message) {
  if (typeof message !== "string" || message.length > 2500) return null;
  const match=/^JESSICA_REPLY_([0-9a-f-]{36})\s*\n/iu.exec(message);
  if (!match || !message.trimEnd().endsWith("JESSICA_END_" + match[1])) return null;
  const body=message.slice(match[0].length,message.lastIndexOf("JESSICA_END_"))
    .replace(/\s+/gu," ").trim();
  if (!/тест\s+пока\s+не\s+(?:нужен|запрошен|повторять|запускать)/iu.test(body))
    return null;
  if (!/\b(?:CI|Render)\b/iu.test(body) || !/backend/iu.test(body) ||
      !/\b[0-9a-f]{7,40}\b/iu.test(body)) return null;
  const next=/следующий\s+шаг\s*[—–:-]\s*([^.!?]{8,160})/iu.exec(body);
  if (!next || !/пораздельн[а-яё]*\s+анализ[а-яё]*\s+всего\s+документ/iu.test(next[1]))
    return null;
  if (/(?:запусти|выполни|повтори|проведи)\s+тест|в\s+обычном\s+чате\s+отправь|\bAPK\b|секрет|парол|настройк|введи\s+ключ|удали\s+данные/iu.test(body))
    return null;
  return "Поняла: контроль цитат добавлен, но полный документ ещё не проанализирован. Проверь CI и Render указанного коммита, затем продолжай пораздельный анализ документа в backend и Supabase. Тест в обычном чате Jessica пока не нужен.";
}
