/** Provider-free CALIBRI status: "after GitHub CI completes, check 96 answers".
 *
 * This is a future verification owned by ChatGPT/GitHub, NOT a textual
 * ordinary-chat test for Jessica. Prevent the real S26 90-second timeout
 * caused by interpreting "после завершения CI проверить..." as a test.
 * Keep explicit ordinary-chat requests eligible through the normal route.
 */
export function externalResearchCiHandoff(message) {
    if (typeof message !== "string" || !message.trim() || message.length > 1400) return null;
    const lines=message.trim().split(/\r?\n/u).map(x=>x.trim()).filter(Boolean);
    const envelope=/^JESSICA_REPLY_([\da-f]{8}-[\da-f]{4}-[\da-f]{4}-[\da-f]{4}-[\da-f]{12})$/iu.exec(lines[0]||"");
    let body;
    if (envelope) {
        if (lines.length<3 || lines.at(-1)!=="JESSICA_END_"+envelope[1])return null;
        body=lines.slice(1,-1).join(" ");
    } else {
        if (/JESSICA_(?:REPLY|END)_/iu.test(message)) return null;
        body=lines.join(" ");
    }
    body=body.replace(/\s+/gu," ").trim();
    if (body.length>850) return null;
    const match=/(?:^|[^\p{L}\p{N}])S(\d{1,3})(?=[^\p{L}\p{N}]|$)/iu.exec(body);
    if (!match || Number(match[1])<1) return null;
    if (!/\bGitHub\b/iu.test(body) || !/\bCI\b/iu.test(body) ||
        !/Следующ(?:ий|ая)\s+шаг\s*[—–:-]\s*после\s+завершения\s+CI\s+проверить\s+96\s+(?:новых\s+)?двухсловных\s+ответов/iu.test(body) ||
        !/\bbpp\b/iu.test(body) || !/\bCPU\b/iu.test(body) ||
        !/ChatGPT/iu.test(body) ||
        !/(?:Jessica|Джессика)\s+(?:пока\s+)?только\s+переда[её]т\s+сообщения\s*[.!]?$/iu.test(body))
        return null;
    // Explicit Jessica test actions, user secrets and arbitrary trailing
    // material cannot be swallowed by the passive status shortcut.
    if (/(?:создай|открой|запусти|выполни|проведи|повтори|отправь)\s+(?:новый\s+|один\s+|следующий\s+|этот\s+)?(?:тест|тестовый\s+чат|задач[уа])/iu.test(body) ||
        /(?:в\s+обычном\s+чате|через\s+обычный\s+чат|Точная\s+(?:задача|формулировка)|парол|секрет|API[- ]?ключ|токен\s+(?:доступа|авторизац)|удали|установи|deploy|merge|https?:\/\/|[\x60«»])/iu.test(body))
        return null;
    return {
        action:"reply",task:"",continueThread:false,
        reply:"CALIBRI S"+match[1]+": проверка 96 ответов, bpp и CPU "+
              "после завершения GitHub CI принадлежит ChatGPT. "+
              "Тест в обычном чате Jessica не запрашивался и не запускался."
    };
}