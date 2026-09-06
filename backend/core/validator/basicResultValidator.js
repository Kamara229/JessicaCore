/*
 * JESSICA BASIC RESULT VALIDATOR
 *
 * Простые технические проверки результата без AI.
 */

export function validateBasicResult(
    taskRunResult,
    answerResult
) {

    if (
        !answerResult ||
        answerResult.success !== true
    ) {

        return {
            valid: false,
            shouldRetry: true,
            needsClarification: false,
            reason: "Не удалось сформировать итоговый ответ"
        };

    }


    if (
        typeof answerResult.text !== "string" ||
        !answerResult.text.trim()
    ) {

        return {
            valid: false,
            shouldRetry: true,
            needsClarification: false,
            reason: "Итоговый ответ пустой"
        };

    }


    if (
        taskRunResult &&
        taskRunResult.success === false
    ) {

        const needsClarification =
            taskRunResult.needsClarification === true;


        return {
            valid: false,
            shouldRetry: !needsClarification,
            needsClarification,
            reason:
                taskRunResult.text ||
                "План выполнен с ошибкой"
        };

    }


    return {
        valid: true,
        shouldRetry: false,
        needsClarification: false,
        reason: "Базовые технические проверки пройдены"
    };

}
