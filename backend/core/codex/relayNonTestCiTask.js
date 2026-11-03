/**
 * Recognize one very narrow, explicitly non-Jessica-test task handed through
 * the selected ChatGPT thread, even when external model interpretation fails.
 *
 * A task for ChatGPT to inspect Github Actions is NEVER a Jessica chat test.
 * This handler cannot run tests, create branches, call providers or deploy.
 */
export function nonTestCiTaskHandoff(message) {
    if (typeof message !== "string" || message.length > 1800) return null;
    const trimmed = message.trim();
    const wrapped = /^JESSICA_REPLY_([0-9a-f-]{36})\s*\r?\n/iu.exec(trimmed);
    let body = trimmed;
    if (wrapped) {
        if (!/^[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}$/iu.test(wrapped[1]))
            return null;
        const suffix="JESSICA_END_"+wrapped[1];
        if (trimmed.split(/\r?\n/u).at(-1).trim()!==suffix) return null;
        body=trimmed.slice(wrapped[0].length,trimmed.lastIndexOf(suffix)).trim();
    } else if (/JESSICA_(?:REPLY|END)_/iu.test(trimmed)) return null;
    if (body.length>500) return null;
    // Exact prior-turn recovery case, not an invitation to execute a task.
    const match=/^Дословная\s+задача\s*:\s*[«"]\s*(анализ\s+результатов\s+v\d{1,3}|проверка\s+(?:статуса|результатов)\s+(?:GitHub\s+Actions|CI)(?:\s+#\d{3,14})?)\s*[»"]\.?\s*Это\s+проверка\s+результатов\s+GitHub\s+Actions\s*,\s*а\s+не\s+текстовый\s+тест\s+в\s+чате\s+Jessica\.\s*Запуск\s+теста\s+Jessica\s+не\s+требуется\.?$/iu.exec(body);
    if (!match) return null;
    return {
        action:"reply", task:"", continueThread:false,
        reply:"Поняла: «"+match[1]+"» — задача ChatGPT по проверке GitHub Actions, не тест Jessica. Продолжай проверку CI и исследование CALIBRI через GitHub; тест Jessica не запускаю."
    };
}
