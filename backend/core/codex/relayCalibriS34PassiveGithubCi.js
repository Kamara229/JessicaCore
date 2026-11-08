/**
 * Passive CALIBRI S34 CI-status messages are instructions for ChatGPT's
 * GitHub research, NOT requests for ordinary-chat Jessica tests.
 * Exact 2026-11-08 S34 message failed the existing S28+ classifier because
 * it used "Jessica не тестировать, LIVE не менять" rather than ending with
 * "Jessica тесты не запускает"; it fell through to quota-bound provider.
 * Fail-closed. No IO, model inference, tests, or production changes.
 */
export function passiveS34GithubCiNoTest(message) {
    if (typeof message !== "string" || !message.trim() || message.length > 1400)
        return null;
    const lines = message.trim().split(/\r?\n/u).map(s => s.trim()).filter(Boolean);
    const start = /^JESSICA_REPLY_([\da-f]{8}-[\da-f]{4}-[\da-f]{4}-[\da-f]{4}-[\da-f]{12})$/iu.exec(lines[0] || "");
    let body;
    if (start) {
        if (lines.length < 3 ||
            lines.at(-1).toLowerCase() !== ("JESSICA_END_" + start[1]).toLowerCase())
            return null;
        body = lines.slice(1, -1).join(" ");
    } else {
        if (/JESSICA_(?:REPLY|END)_/iu.test(message)) return null;
        body = lines.join(" ");
    }
    body = body.replace(/\s+/gu, " ").trim();
    if (!body || body.length > 850) return null;

    // A real S34 JSON report was sent as a sandbox attachment, not as a
    // GitHub-CI instruction. Without this narrowly scoped path the passive
    // report hits the quota-limited model and incorrectly yields an incident.
    // This is acknowledgement ONLY; it does not inspect, execute or test.
    const s34TrainReport =
        /^JSON\\s+S34\\s*:/iu.test(body) &&
        /\\[скачать\\s+отч[её]т\\]\\(sandbox:\\/mnt\\/data\\/S34_actual_36M_FP32_TRAIN_report\\.json\\)/iu.test(body) &&
        /\\b240\\b/u.test(body) && /\\b214\\b/u.test(body) &&
        /\\b26\\b/u.test(body) && /\\b80\\b/u.test(body) &&
        /(?:80\\s+шагов\\s+отклонены)/iu.test(body) &&
        /\\bLIVE\\s+не\\s+меня(?:ть|й)\\b/iu.test(body) &&
        !/(?:создай|открой|запусти|выполни|проведи|повтори|протестируй)\\s+(?:(?:новый|один|этот|текстовый|следующий)\\s+)?(?:тест(?:ы)?|тестовый\\s+чат)/iu.test(body) &&
        !/(?:https?:\\/\\/|парол|секрет|API[-\\s]?ключ|токен\\s+доступа|deploy|merge|удали|установи\\s+(?:APK|пакет|приложение)|[«»\\x60])/iu.test(body);
    if (s34TrainReport) return {
        action: "reply", task: "", continueThread: false,
        reply: "S34 JSON-отчёт получен: 80 шагов отклонены, новые веса не улучшились. " +
            "Анализ причин отклонения — задача ChatGPT через GitHub. " +
            "Jessica тест не запускает, LIVE не меняет."
    };

    if (!/^CALIBRI\s+S34\s*[:—–-]\s*Следующий\s+шаг\s*[—–:-]\s*[^.!?]+/iu.test(body))
        return null;
    if (!/\bGitHub\b/iu.test(body) ||
        !/\bPR\s*#?417\b/iu.test(body) ||
        !/\b(?:CI|workflow|Actions)\b/iu.test(body) ||
        !/(?:research[-\s]?ветк|провер|триггер|запуск)/iu.test(body))
        return null;

    // Require BOTH explicit boundaries so a genuine chat test is not masked.
    if (!/(?:(?:Jessica|Джессика)\s+(?:тесты?\s+не\s+запуска(?:ет|й|ть)|не\s+тестировать)|Тесты\s+не\s+запускай)/iu.test(body) ||
        !/\bLIVE\s+не\s+меня(?:ть|й)/iu.test(body))
        return null;

    if (/(?:создай|открой|запусти|выполни|проведи|повтори|протестируй)\s+(?:(?:новый|один|этот|текстовый|следующий)\s+)?(?:тест(?:ы)?|тестовый\s+чат)/iu.test(body) ||
        /(?:тест\s+через\s+(?:обычный\s+)?чат|тест\s+в\s+обычном\s+чате|точная\s+(?:формулировка|задача)|секрет|парол|API[-\s]?ключ|токен\s+(?:доступа|авторизации)|удали|установи\s+(?:APK|пакет|приложение)|deploy|merge|https?:\/\/|[«»\x60])/iu.test(body) ||
        /(?:план|исследование)\s+(?:полностью\s+)?заверш[её]н/iu.test(body))
        return null;

    return {
        action: "reply", task: "", continueThread: false,
        reply: "CALIBRI S34: проверка GitHub CI и PR #417 — задача ChatGPT, " +
            "а не текстовый тест Jessica. Тест не запускался. " +
            "Новых измерений не утверждаю, LIVE не меняю."
    };
}
