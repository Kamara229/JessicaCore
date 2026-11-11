/**
 * Exact, bounded CALIBRI status/next-step handoff with explicit NO-Jessica-test
 * intent. The selected ChatGPT conversation owns research and GitHub Actions.
 *
 * This facade only classifies a read-only handoff. It never runs a benchmark,
 * touches source code, verifies CI, or claims a test has passed.
 *
 * The Android client normally passes only the validated message body. Other
 * clients may pass the matching JESSICA_REPLY/END envelope.
 */
export function calibriNoTestNextStep(message) {
    if (typeof message !== "string" || !message.trim() || message.length > 1700)
        return null;

    const lines = message.trim().split(/\r?\n/u).map(s => s.trim()).filter(Boolean);
    const open = /^JESSICA_REPLY_([\da-f]{8}-[\da-f]{4}-[\da-f]{4}-[\da-f]{4}-[\da-f]{12})$/iu.exec(lines[0] || "");
    let raw;
    if (open) {
        if (lines.length < 3 || lines.at(-1) !== "JESSICA_END_" + open[1])
            return null;
        raw = lines.slice(1, -1).join(" ");
    } else {
        if (/JESSICA_(?:REPLY|END)_/iu.test(message)) return null;
        raw = lines.join(" ");
    }
    const body = raw.replace(/\s+/gu, " ").trim();
    if (!body || body.length > 750) return null;

    // Explicitly restrict this shortcut to compact, stand-alone CALIBRI
    // progress or a suggested experiment, not arbitrary quoted instructions.
    const direct = /^(?:Следующ(?:ий|ая)\s+(?:тест|шаг|проверка|эксперимент)\s+)?CALIBRI\s+v(\d{2,3})(?=\b|[,.:;—–-])/iu.exec(body);
    // Mixed incident report (e.g. PR #267 fixed + CALIBRI next step) must not
    // fall through to quota-bound Groq simply because CALIBRI is not word 1.
    // Require an exact bounded incident shape rather than arbitrary prefix text.
    const mixedIncident = /^Сбой\s+(?:Codex|Кодекс)\s+исправлен\s+в\s+GitHub\s*:\s*PR\s*#\d{1,6}\s+объедин[её]н,\s+регрессии\s+прошли\./iu.test(body);
    const nested = mixedIncident && /Следующ(?:ий|ая)\s+(?:тест|шаг|проверка|эксперимент)\s*[—–:-]\s*CALIBRI\s+v(\d{2,3})(?=\b|[,.:;—–-])/iu.exec(body);
    const version = direct?.[1] ?? (nested || [])[1];
    if (!version || Number(version) < 23) return null;
    if (!/(?:\b(?:GitHub|PR|EOS|Green|CPU|FP32|CB59|NLL|KL)\b|нейровес|сло[её]в|эксперимент|исследован|коррекц|сравнен|провер)/iu.test(body))
        return null;

    // The caller, not Jessica, must explicitly own all testing.
    const noJessicaTest =
        /(?:Джессика|Jessica)\s*,?\s*тесты?\s+не\s+запуска(?:й|ет|ть|ю)/iu.test(body) ||
        /тест(?:ы)?\s+(?:в\s+(?:сво[её]м|обычном)\s+чате\s+)?Jessica\s+не\s+(?:запуска(?:й|ет|ть|ю)|нужен|требуется)/iu.test(body) ||
        /Запуск\s+(?:теста\s+)?выполняю\s+я\s+через\s+GitHub/iu.test(body) ||
        // Owner's CALIBRI-only no-test relay: short "тесты не запускай".
        // Require a PR and a next-step heading to prevent generic bypass.
        (/(?:^|[\s.,;!?])тест(?:ы)?\s+не\s+запускай(?:те)?(?=$|[\s.,;!?])/iu.test(body) &&
            /\bPR\s*#\d{1,6}\b/iu.test(body) &&
            /Следующ(?:ий|ая)\s+(?:шаг|проверка)/iu.test(body)) ||
        // "Jessica/LIVE не менять" is a strictly no-action CALIBRI notice.
        // Require PR and next step to avoid classifying arbitrary text as progress.
        (/Jessica\s*\/\s*LIVE\s+не\s+(?:менять|трогать)(?:[.!?,;]|$)/iu.test(body) &&
            /\bPR\s*#\d{1,6}\b/iu.test(body) &&
            /Следующ(?:ий|ая)\s+(?:шаг|проверка)/iu.test(body));
    if (!noJessicaTest) return null;

    // Do not turn an ambiguous/conflicting instruction into consent to test.
    // Explicit ordinary-chat test requests use the existing strict path.
    // Credential TOKEN requests are unsafe, but LLM token terminology
    // ('первом токене', 'токенов') is ordinary CALIBRI research, not a secret.
    if (/(?:создай|открой)\s+(?:новый\s+)?тестовый\s+чат\s+(?:Jessica|Джессик)/iu.test(body) ||
        /(?:выполни|запусти|проведи|отправь|повтори|создай|открой)\s+(?:(?:один|этот|новый|следующий)\s+)?(?:тест(?:ы)?|тестовый\s+чат|задач[уа])\s+(?:в|через|для)\s+(?:обычн|сво[её]м|Jessica|Джессик)/iu.test(body) ||
        /(?:Точная\s+(?:формулировка|задача|инструкция)|https?:\/\/|[«»]|(?:парол|секрет|API[- ]?ключ|(?:токен(?:а|ы|ов)?\s+(?:доступа|авторизац\w*|GitHub|API|Render|Supabase)|(?:выдай|покажи|пришли|отправь|раскрой|используй)\s+(?:мне\s+)?токен(?=$|[\s.,;!?]))|установи|удали|удалить|deploy|merge))/iu.test(body) ||
        body.includes(String.fromCharCode(96)))
        return null;

    return "Поняла: CALIBRI v" + version +
        " — следующий исследовательский шаг выполняет ChatGPT через GitHub. " +
        "В обычном чате Jessica тесты не запускаю. Продолжай исследование " +
        "в выбранном чате по фактическим результатам.";
}
