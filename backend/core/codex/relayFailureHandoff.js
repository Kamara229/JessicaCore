/**
 * A model outage is a failed interpretation, NOT a completed Jessica task.
 * Hand the failure to the same selected ChatGPT conversation so its assistant
 * can inspect GitHub/Render/Supabase. No test, retry, or learning write here.
 */
const providerUnavailable = error => {
    const detail = String(error?.message || error || "");
    return /\b429\b|rate[_ -]?limit|quota|tokens? per day|resource[_ -]?exhausted|daily[_ -]?token/i.test(detail);
};
export function interpretationFailureHandoff(error, researchMode = false) {
    const quota = providerUnavailable(error);
    if (researchMode) return {
        action: "reply",
        task: "",
        continueThread: false,
        reply: "Кодекс Jessica: анализ этого этапа CALIBRI не выполнен — " +
            (quota ? "AI-провайдер достиг лимита." : "сервис интерпретации недоступен.") +
            " Тест не запускался; эксперимент и результаты не подтверждены мной. " +
            "Продолжай исследование CALIBRI через свои инструменты в отдельной research-ветке: " +
            "следующий небольшой эксперимент с реальным размером, качеством и временем вычисления. " +
            "Не меняй main, Render, Supabase и APK."
    };
    return {
        action: "reply",
        task: "",
        continueThread: false,
        reply: "Кодекс Jessica: анализ последнего сообщения ChatGPT не выполнен — " +
            (quota ? "ограничение запросов AI-провайдера (HTTP 429/квота)." :
                "сервис интерпретации вернул ошибку или не дал однозначного решения.") +
            " Тест из этого сообщения не запускался, результата нет, автоматического повтора не было." +
            " Передаю технический сбой в выбранный чат. Проверь backend, Render и Supabase" +
            " через свои подключения, устрани причину и продолжи план. Это НЕ успешный тест."
    };
}
