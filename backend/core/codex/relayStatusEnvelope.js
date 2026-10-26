/**
 * Deterministic status handoff for bounded ChatGPT relay envelopes.
 * AI quota is never needed to acknowledge a safe "no test" progress report.
 * Source text is not executed and no tasks, test chats or external tools run.
 */
export function deterministicStatusHandoff(message) {
    if (typeof message!=="string" || message.length>2500) return null;
    // Android ChatReplyExtraction.read() deliberately strips both markers and
    // passes ONLY the bounded body to /api/codex/relay/interpret. Rejecting
    // that verified body here silently fell through to quota-bound Groq.
    // Also accept the original full envelope for other clients and tests.
    const match=/^JESSICA_REPLY_([a-f\d]{8}-[a-f\d]{4}-[a-f\d]{4}-[a-f\d]{4}-[a-f\d]{12})\s*\r?\n/iu.exec(message);
    let rawBody;
    if(match) {
        const suffix="JESSICA_END_"+match[1];
        if(message.trimEnd().split(/\r?\n/u).at(-1)?.trim()!==suffix)
            return null;
        rawBody=message.slice(match[0].length,message.lastIndexOf(suffix));
    } else {
        // Never accept a broken, incomplete, or nested marker as body-only.
        if(/JESSICA_[A-Z]+_/iu.test(message))return null;
        // The extracted body must carry a verified positive development,
        // not merely a self-declared "CI completed" assertion.
        if(!/(?:регресси(?:и|онные тесты)\s+прошли|backend[- ]тесты\s+прошли|Render\s+Live|Supabase\s+проверен)/iu.test(message))
            return null;
        rawBody=message;
    }
    const body=rawBody.replace(/\s+/gu," ").trim();
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
    // CI notes can start with the previous run, then a completed code fix.
    // The returned action is an acknowledgement, not a verified test result.
    const ciProgress=/^Предыдущий\s+(?:общий\s+)?CI\s+(?:успешен|пройден|заверш[её]н\s+успешно)\.\s+(?:Добавил|Исправил|Обновил|Доработал|Внедрил|Реализовал|Усилил)[^.!?;]{0,180}\([a-f\d]{7,40}\)/iu.test(body);
    if (!directProgress && !quotaIncident && !ciProgress) return null;
    if (!/\([a-f\d]{7,40}\)/iu.test(body)) return null;
    if (!/(?:backend|регресс|CI|Render|Supabase)/iu.test(body)) return null;
    if (!/(?:тест\s+(?:в\s+(?:обычном\s+)?чате\s+Jessica\s+)?пока\s+не\s+(?:нужен|запрошен|повторять|запускать)|тест\s+пока\s+не\s+повторять)/iu.test(body))
        return null;
    // Refuse untrusted requests for privileged actions or implicit test execution.
    if (/(?:запусти|выполни|проведи|повтори|отправь|создай|открой)\s+(?:новый\s+)?(?:тест|чат)|(?:в\s+обычном\s+чате\s+отправь)|\bAPK\b|(?:парол|секрет|API[- ]?ключ|токен|настройк|удалить|удали|телефон|установить|установи)/iu.test(body))
        return null;
    // Progress notes can omit "Далее". Recognize only source-bounded,
    // committed, confirmed no-test CI status; never run tests or tools.
    const firstSentence=body.split(/[.!?]/u,1)[0];
    const committedProgress=directProgress &&
        /^(?:Добавил|Исправил|Обновил|Доработал|Внедрил|Реализовал|Усилил|Проверил)(?:\s|[.,:])[^.!?;]{0,200}\([a-f\d]{7,40}\)/iu.test(firstSentence);
    const noNextProgress=committedProgress &&
        /(?:^|[.!?]\s+)(?:(?:Новый|Текущий|Общий)\s+(?:общий\s+)?CI|CI)\s+(?:ещ[её]\s+)?(?:ид[её]т|выполняется|в\s+процессе|запущен)(?:[.!?]|$)/iu.test(body) &&
        /(?:backend[- ]регресси(?:и|онные тесты)|регрессии)\s+прошли/iu.test(body) &&
        /Render\s+Live/iu.test(body) &&
        /Supabase\s+проверен/iu.test(body) &&
        /Тест\s+пока\s+не\s+нужен\.?$/iu.test(body);
    const next=/(?:Следующий\s+шаг|Далее)\s*[:—–-]\s*([^.!?]{6,180})/iu.exec(body);
    if (next) {
        const target=next[1].trim();
        // "Далее — анализ свыше 16 окон" is a safe future work item
        // rather than a Jessica test. A noun alone grants no broad action.
        const nounStep=/^(?:анализ|обработка|проверка|сборка|планирование)\s+/iu.test(target);
        if (!nounStep && !/^(?:провер(?:ить|ка|ять)|завершить\s+проверку|продолжить|перейти\s+к|сопоставить)\s+/iu.test(target))
            return null;
        const scope=/(?:\bCI\b|Render|GitHub|Supabase|backend|регресс|разв[её]рт|документ|раздел|план|сопоставлен|цитат|окон|страниц|файлов|\bresume\b)/iu.test(target);
        if (!scope && (nounStep || !/анализ/iu.test(target))) return null;
    } else if (!ciProgress && !noNextProgress) return null;
    else if (ciProgress && !noNextProgress && (
        !/(?:Новый|Текущий)\s+(?:общий\s+)?CI\s+(?:ещ[её]\s+)?(?:ид[её]т|выполняется|в\s+процессе|запущен)/iu.test(body) ||
        !/(?:backend[- ]регресси(?:и|онные тесты)|регрессии)\s+прошли/iu.test(body) ||
        !/Render\s+Live/iu.test(body) ||
        !/Supabase\s+проверен/iu.test(body) ||
        !/Тест\s+пока\s+не\s+нужен\.?$/iu.test(body)))
        return null;
    return "Поняла: предыдущий шаг backend отмечен, но тест Jessica не выполнялся. Проверь GitHub CI, Render и Supabase и продолжай текущий план. Тест в обычном чате Jessica пока не нужен.";
}
