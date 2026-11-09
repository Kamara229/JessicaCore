/**
 * Quota-free continuation for short, explicitly non-executing S-numbered
 * CALIBRI research updates. Never calls a provider, runs Jessica tests, or
 * claims to have verified an external GitHub result.
 *
 * The phone normally extracts just the JESSICA_REPLY/END body; also accept
 * one fully matched envelope for older clients. Bad markers fail closed.
 */
export function continuousCalibrIResearchHandoff(message) {
    if (typeof message !== "string" || !message.trim() || message.length > 1500)
        return null;
    const lines = message.trim().split(/\r?\n/u).map(s=>s.trim()).filter(Boolean);
    const opening = /^JESSICA_REPLY_([a-f\d]{8}-[a-f\d]{4}-[a-f\d]{4}-[a-f\d]{4}-[a-f\d]{12})$/iu.exec(lines[0] || "");
    let body;
    if (opening) {
        if (lines.length < 3 || lines.at(-1) !== "JESSICA_END_" + opening[1]) return null;
        body = lines.slice(1,-1).join(" ");
    } else {
        if (/JESSICA_(?:REPLY|END)_/iu.test(message)) return null;
        body = lines.join(" ");
    }
    body = body.replace(/\s+/gu," ").trim();
    if (!body || body.length > 750) return null;
    const stage = /^(?:CALIBRI\s+)?S(\d{1,3})(?=[:.,;\s—–-])/iu.exec(body)?.[1];
    if (!stage || Number(stage) < 1 || Number(stage) > 999) return null;
    if (!/(?:TRAIN|CPU|GPU|GitHub|Green\s+Tunnel|\bbpp\b|\bEOS\b)/iu.test(body))
        return null;
    if (!/Следующ(?:ий|ая)\s+шаг\s*[—–:-]\s*\S/iu.test(body))
        return null;
    // Only clearly delegated no-Jessica-test progress may bypass the model.
    const noJessicaTest =
        /LIVE\s+не\s+менять/iu.test(body) ||
        /(?:Jessica|Джессик[ауе])\s+не\s+(?:тестировать|запуска(?:ть|й|ет)\s+тест)/iu.test(body) ||
        /(?:Тест(?:ы)?\s+(?:в\s+(?:обычном\s+)?чате\s+)?(?:Jessica|Джессик[аи]))\s+не\s+(?:нужен|нужны|запрошен|требуется|запуска(?:ть|й|ет))/iu.test(body) ||
        /(?:HTTP\s*429|сбой)\s+(?:Jessica|Джессик[аи])\s+не\s+(?:является|считать)\s+тестом/iu.test(body);
    if (!noJessicaTest) return null;
    // Explicit test execution, credentials, deployment and completion override
    // any passive language, even in the same short message.
    if (/(?:выполни|запусти|проведи|повтори|отправь)\s+(?:один\s+|этот\s+|новый\s+|следующий\s+)?(?:тест|тестовую\s+задачу)/iu.test(body) ||
        /(?:создай|открой)\s+(?:новый\s+)?тестовый\s+чат/iu.test(body) ||
        /(?:парол|секрет|API[- ]?ключ|токен\s+(?:доступа|авторизац)|\bAPK\b|\bdeploy\b|\bmerge\b|https?:\/\/|[«»`])/iu.test(body) ||
        /(?:план|исследование)\s+(?:полностью\s+)?заверш[её]н/iu.test(body))
        return null;
    return {
        action:"reply",task:"",continueThread:false,
        reply:"CALIBRI S"+stage+
            ": сообщение о текущем исследовании принято. Тест в обычном чате Jessica не запускала. "+
            "Продолжай проверки через GitHub и сообщай следующий шаг; остаюсь в этом чате."
    };
}
