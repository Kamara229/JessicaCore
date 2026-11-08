/** Provider-free CALIBRI 2.0 research relay for short owner-safe status notes.
 *
 * ChatGPT owns all research, code, and GitHub test execution. This narrowly
 * recognizes a passive note naming a numbered stage; it does not interpret a
 * user instruction as permission to run a Jessica test or modify LIVE.
 * Unknown, invalid, or adversarial messages still follow the old interpreter.
 */
export function calibri2ResearchPassiveHandoff(message) {
    if (typeof message !== "string" || !message.trim() || message.length > 1400)
        return null;
    const lines=message.trim().split(/\r?\n/u).map(s=>s.trim()).filter(Boolean);
    const open=/^JESSICA_REPLY_([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$/iu.exec(lines[0]||"");
    let body;
    if (open) {
        if (lines.length < 3 || lines.at(-1)!=="JESSICA_END_"+open[1])
            return null;
        body=lines.slice(1,-1).join(" ");
    } else {
        if (/JESSICA_(?:REPLY|END)_/iu.test(message)) return null;
        body=lines.join(" ");
    }
    body=body.replace(/\s+/gu," ").trim();
    if (!body || body.length > 700) return null;
    if (!/^Продолжаем\s+(?:исследование|работу\s+над)\s+CALIBRI\s+2\.0[.!\s]/iu.test(body))
        return null;
    const stage=/\bS(\d{1,3})\b/iu.exec(body)?.[1];
    if (!stage || Number(stage) < 1 || !/\bGitHub\b/iu.test(body))
        return null;
    if (!/Следующ(?:ий|ая)\s+шаг\s*[—–:\-]?\s*(?:дождаться|проверить|сверить|изучить)(?=\s|[.!?,;]|$)/iu.test(body))
        return null;
    if (!/Разработку\s+и\s+тесты\s+вед[её]т\s+ChatGPT/iu.test(body) ||
        !/(?:Jessica|Джессика)\s+только\s+переда[её]т\s+сообщения,\s*без\s+изменений\s+LIVE[.!]?$/iu.test(body))
        return null;
    // Explicit ordinary-chat tests must never be hidden by this shortcut.
    if (/(?:запусти|выполни|проведи|повтори|создай|открой)\s+(?:новый\s+|следующий\s+|этот\s+|один\s+)?(?:тест(?:ы)?|тестовый\s+чат|задач[уа])/iu.test(body) ||
        /(?:парол|секрет|API[- ]?ключ|токен\s+(?:доступа|авторизации)|установи|удали|deploy|merge|https?:\/\/|[«»\x60])/iu.test(body) ||
        /(?:план|исследование)\s+(?:полностью\s+)?заверш[её]н/iu.test(body))
        return null;
    return {
        action:"reply",task:"",continueThread:false,
        reply:"CALIBRI 2.0, S"+stage+
              ": новый тест Jessica не запрашивался и не запускался. "+
              "Код и проверки GitHub ведёт ChatGPT. "+
              "Передаю новые сообщения владельца без выдуманных результатов."
    };
}
