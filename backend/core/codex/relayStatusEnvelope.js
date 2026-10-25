/**
 * Deterministic status handoff for bounded ChatGPT relay envelopes.
 * AI quota is never needed to acknowledge a safe "no test" progress report.
 * Source text is not executed and no tasks, test chats or external tools run.
 */
export function deterministicStatusHandoff(message) {
    if (typeof message!=="string" || message.length>2500) return null;
    const match=/^JESSICA_REPLY_([a-f\d]{8}-[a-f\d]{4}-[a-f\d]{4}-[a-f\d]{4}-[a-f\d]{12})\s*\r?\n/iu.exec(message);
    if (!match) return null;
    const suffix="JESSICA_END_"+match[1];
    if (message.trimEnd().split(/\r?\n/u).at(-1)?.trim()!==suffix) return null;
    const endIndex=message.lastIndexOf(suffix);
    const body=message.slice(match[0].length,endIndex).replace(/\s+/gu," ").trim();
    if (!body || body.length>900) return null;
    // This path is intended only for positive technical progress, not plans,
    // arguments, quoted instructions or ambiguous test outcomes.
    const progressVerb="(?:Добавил|Исправил|Обновил|Доработал|Внедрил|Реализовал|Усилил|Проверил)";
    const directProgress=new RegExp(`^(?:${progressVerb}|Причина\\s+сбоя\\s+найдена)`,"iu").test(body);
    // Incident updates often start with a precise diagnosis rather than
    // the verb. They still require the later completed fix AND commit hash.
    const quotaIncident=new RegExp(
        `^Причина\\s+(?:HTTP\\s*)?429(?:\\s|\\s*[—–:-])[^.!?]{0,150}[.!?]\\s+${progressVerb}[^.!?]{0,160}\\([a-f\\d]{7,40}\\)`,
        "iu"
    ).test(body);
    if (!directProgress && !quotaIncident) return null;
    if (!/\([a-f\d]{7,40}\)/iu.test(body)) return null;
    if (!/(?:backend|регресс|CI|Render|Supabase)/iu.test(body)) return null;
    if (!/(?:тест\s+(?:в\s+(?:обычном\s+)?чате\s+Jessica\s+)?пока\s+не\s+(?:нужен|запрошен|повторять|запускать)|тест\s+пока\s+не\s+повторять)/iu.test(body))
        return null;
    // Refuse untrusted requests for privileged actions or implicit test execution.
    if (/(?:запусти|выполни|проведи|повтори|отправь|создай|открой)\s+(?:новый\s+)?(?:тест|чат)|(?:в\s+обычном\s+чате\s+отправь)|\bAPK\b|(?:парол|секрет|API[- ]?ключ|токен|настройк|удалить|удали|телефон|установить|установи)/iu.test(body))
        return null;
    const next=/(?:Следующий\s+шаг|Далее)\s*[:—–-]\s*([^.!?]{6,180})/iu.exec(body);
    if (!next) return null;
    const target=next[1].trim();
    if (!/^(?:провер(?:ить|ка|ять)|завершить\s+проверку|продолжить|перейти\s+к|сопоставить)\s+/iu.test(target))
        return null;
    if (!/(?:\bCI\b|Render|GitHub|Supabase|backend|регресс|разв[её]рт|анализ|документ|план|сопоставлен|цитат)/iu.test(target))
        return null;
    return "Поняла: предыдущий шаг backend отмечен, но тест Jessica не выполнялся. Проверь GitHub CI, Render и Supabase и продолжай текущий план. Тест в обычном чате Jessica пока не нужен.";
}
