/** Allow a narrow class of ChatGPT backend progress notes without an AI call. */
export function acknowledgeBackendProgress(message) {
  const raw=typeof message==="string"?message:"";
  if (!raw || raw.length>10000) return null;
  const lines=raw.split(/\r?\n/).map(s=>s.trim())
    .filter(s=>s && !/^JESSICA_(?:REPLY|END)_[\da-f-]+$/iu.test(s));
  const body=lines.join(" ").replace(/\s+/gu," ").trim();
  const isProgress=/^(?:тест (?:пройден|частично пройден|прош[её]л|не пройден|не подтвержд[её]н|завершился (?:ошибкой|сбоем))|(?:исправил|обновил|доработал|вн[её]с)(?:\s|[.,:])|(?:проверка|исправление)\s+завершен[ао]|коммит(?:\s|[.,:])|Render\s+Live\s*:|CI\s+(?:пройд[её]н|заверш[её]н|выполняется))/iu.test(body);
  const next=/следующ(?:ий|им)\s+шаг\s*[:—–-]?\s*(.{3,240})/iu.exec(body);
  const standaloneStep=/^следующ(?:ий|им)\s+шаг\s*[:—–-]?\s+/iu.test(body);
  if ((!isProgress && !standaloneStep) || !next) return null;
  if (!/(?:CI|Render|GitHub|Supabase|backend|регресс|лог(?:и|ов)|депло|разв[её]рт|провер|Apache|TinyFish|web_fetch|первоисточник|загрузк)/iu.test(next[1]) ||
      /(?:APK|секрет|(?<!\p{L})ключ[а-яё]{0,5}(?!\p{L})|парол|настро(?:йк|ек)|владел|телефон|установ)/iu.test(next[1])) return null;
  const failedTest=/^тест (?:частично пройден|не пройден|не подтвержд[её]н|завершился (?:ошибкой|сбоем))/iu.test(body);
  return (failedTest?"Поняла: тест не пройден, исправление backend учтено. ":
      standaloneStep?"Поняла следующий шаг. ":"Поняла: предыдущий шаг backend завершён. ") +
      "Проверь указанный шаг через свои подключения GitHub, Render и Supabase и продолжай план. Тест в обычном чате Jessica пока не запрошен.";
}
