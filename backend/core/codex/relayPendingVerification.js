/** Safe no-model continuation for pending backend CI/Render verification. */
export function pendingBackendVerification(message) {
  if (typeof message !== "string" || message.length > 10000) return null;
  const body = message.split(/\r?\n/u)
    .map(x => x.trim()).filter(x => x &&
      !/^JESSICA_(?:REPLY|END)_[\da-f-]{36}$/iu.test(x))
    .join(" ").replace(/\s+/gu," ").trim();
  if (!/(?:^|[.!?]\s+)коммит\s+[0-9a-f]{7,40}(?:\b|\.)/iu.test(body)) return null;
  if (!/тест\s+(?:в\s+(?:обычном\s+)?чате(?:\s+Jessica)?\s+)?пока\s+не\s+(?:нужен|запрошен|повторять|запускать)/iu.test(body)) return null;
  if (!/(?:Render.{0,60}CI|CI.{0,60}Render).{0,100}(?:проверя(?:ется|ются)|ещ[её]\s+(?:провер|выполня)|ожида(?:ется|ют)|не\s+заверш|pending|in_progress)/iu.test(body)) return null;
  // A status handoff is never a license for device, account or secret changes.
  if (/(?:установ(?:и|ить)\s+APK|введи\s+(?:пароль|ключ)|смен(?:и|ить)\s+(?:настройки|пароль|ключ)|запусти\s+тест)/iu.test(body)) return null;
  return "Поняла: очередной шаг backend выполнен, но проверка ещё не завершена. Проверь CI и Render через свои подключения, сопоставь коммит с развёртыванием и продолжай план. Тест в обычном чате Jessica пока не нужен.";
}
