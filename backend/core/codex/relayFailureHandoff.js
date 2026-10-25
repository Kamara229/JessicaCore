/**
 * A model outage is a failed interpretation, NOT a completed Jessica task.
 * Hand the failure to the same selected ChatGPT conversation so its assistant
 * can inspect GitHub/Render/Supabase. No test, retry, or learning write here.
 */
const providerUnavailable = error => {
    const detail = String(error?.message || error || "");
    return /\b429\b|rate[_ -]?limit|quota|tokens? per day|resource[_ -]?exhausted|daily[_ -]?token/i.test(detail);
};
export function providerFailureHandoff(error) {
    if (!providerUnavailable(error)) return null;
    return {
        action: "reply",
        task: "",
        continueThread: false,
        reply: "Кодекс Jessica: анализ последнего ответа ChatGPT не выполнен — провайдер AI вернул ограничение запросов (HTTP 429/квота). Запрошенный тест не запускался и не повторялся; результата теста нет. Передаю технический сбой в этот же чат. Проверь backend, Render и Supabase своими подключениями, исправь причину и продолжи план. Не считай это успешным тестом."
    };
}
