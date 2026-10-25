/**
 * Recognise a verified release-status handoff without relying on Groq.
 * This grants no access to external tools or Jessica test chats.
 */
export function releaseCiStatusHandoff(message) {
    if (typeof message !== "string" || message.length > 2500) return null;
    const match = /^JESSICA_REPLY_([0-9a-f]{8}-[0-9a-f-]{27,28})\s*\n/iu.exec(message);
    if (!match) return null;
    const ending = "JESSICA_END_" + match[1];
    if (message.trimEnd().split(/\r?\n/).at(-1)?.trim() !== ending) return null;
    const body = message.slice(match[0].length, message.lastIndexOf(ending))
        .replace(/\s+/gu, " ").trim();
    // Require an actual bounded commit status and a specific, safe next step.
    if (!/\([0-9a-f]{7,40}\)/iu.test(body)) return null;
    if (!/Render\s+Live/iu.test(body) ||
        !/(?:backend[- ]тесты\s+прошли|регресси(?:и|онные тесты)\s+прошли|backend[- ]регрессии\s+прошли)/iu.test(body))
        return null;
    if (!/тест\s+пока\s+не\s+(?:нужен|запрошен|повторять|запускать)/iu.test(body)) return null;
    const next = /следующий\s+шаг\s*[—–:-]\s*([^.!?]{8,160})/iu.exec(body);
    if (!next || !/^проверить\s+(?:завершение|статус|результат(?:ы)?|прохождение)\s+(?:общего\s+|нового\s+|release\s+|phone\s+|обоих\s+)?CI(?:\b|\s|$)/iu.test(next[1])) return null;
    if (/(?:запусти|выполни|повтори|проведи)\s+тест|в\s+обычном\s+чате\s+отправь|\bAPK\b|парол|секрет|ключ\s+доступ|настройк|удал(?:и|ить)\s+данны/iu.test(body)) return null;
    return "Поняла: пораздельный анализ backend внедрён, но общий CI ещё требует проверки. Проверь финальный статус CI и Render через подключения GitHub и Render, затем продолжай план. Тест в обычном чате Jessica пока не нужен.";
}
