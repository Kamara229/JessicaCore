/**
 * A deterministic non-executing CALIBRI chat handoff.
 *
 * The owner chose Jessica as a literal chat relay ONLY. Short CALIBRI
 * progress/idle messages must not fall through to quota-limited AI
 * interpretation and be reported as failed tests. No GitHub/Render/
 * Supabase access, new test, automatic retry, or claim of completed QA.
 *
 * Deliberately constrained: source must start with CALIBRI version,
 * explicitly prohibit Jessica tests/delegate them to ChatGPT, and
 * contain no executable instructions or credential requests.
 */
export function calibriPassiveRelayHandoff(message) {
    if (typeof message !== "string" || !message.trim() || message.length > 1400)
        return null;
    const lines=message.trim().split(/\r?\n/u).map(x=>x.trim()).filter(Boolean);
    const first=lines[0] || "";
    const marker=/^JESSICA_REPLY_([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$/iu.exec(first);
    let body;
    if (marker) {
        if (lines.length<3 || lines.at(-1)!=="JESSICA_END_"+marker[1])
            return null;
        body=lines.slice(1,-1).join(" ");
    } else {
        if (/JESSICA_(?:REPLY|END)_/iu.test(message)) return null;
        body=lines.join(" ");
    }
    body=body.replace(/\s+/gu," ").trim();
    if (body.length>720) return null;

    const version=/^(?:Продолжаем\s+)?CALIBRI\s+v(\d{2,3})(?=\b|[.:,;—–-])/iu.exec(body)?.[1];
    if (!version || Number(version)<23) return null;
    if (!/(?:ChatGPT|GitHub|Кодекс|Джессик|Jessica|тест|подтверждени|сообщени)/iu.test(body))
        return null;

    const explicitlyDelegated=
        /(?:тест(?:ы|ирование)?|исследование|разработку|работу\s+с\s+GitHub)[^.?!]{0,95}(?:вед[её]т|выполняет|проводит)\s+ChatGPT/iu.test(body) ||
        /ChatGPT\s+самостоятельно\s+(?:контролирует|вед[её]т|проверяет|выполняет)[^.?!]{0,80}/iu.test(body) ||
        /(?:Джессик[аеуы]?|Jessica)[^.?!]{0,75}тест(?:ы)?\s+не\s+запуска(?:й|ет|ю|ть)/iu.test(body) ||
        /(?:Тест(?:ы)?|Ничего)\s+не\s+запускай/iu.test(body) ||
        /(?:Не\s+отправляй\s+(?:повторных\s+)?подтверждений|Подтверждения\s+не\s+требуются|Ожидай\s+следующего\s+сообщения)/iu.test(body) ||
        /Тесты\s+и\s+разработку\s+вед[её]т\s+ChatGPT/iu.test(body);
    if (!explicitlyDelegated) return null;

    // A requested task, a secret, a URL, or source code is never
    // reclassified as a passive status. Existing explicit-test
    // extraction runs earlier in the interpreter.
    if (/(?:https?:\/\/|[«»`]|(?:парол|секрет|API[- ]?ключ)|токен\s+(?:доступа|авторизации|GitHub)|(?:запусти|выполни|проведи|повтори|отправь)\s+(?:новый\s+|следующий\s+|этот\s+|один\s+)?тест\b|(?:создай|открой)\s+(?:новый\s+)?тестовый\s+чат\b|(?:deploy|merge)\b)/iu.test(body))
        return null;
    if (/\b(?:план\s+(?:полностью\s+)?заверш[её]н|заверши\s+сессию)\b/iu.test(body))
        return null;
    return {
        action:"reply",task:"",continueThread:false,
        reply:"CALIBRI v"+version+
            ": новый тест Jessica не запрошен и не запускался. " +
            "Разработку и проверки GitHub ведёт ChatGPT. " +
            "Передаю только новые сообщения владельца без повторных подтверждений."
    };
}
