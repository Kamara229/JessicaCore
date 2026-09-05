import OpenAI from "openai";


/*
 * =========================================================
 * JESSICA RESULT VALIDATOR
 * =========================================================
 *
 * Validator получает:
 *
 * - исходную задачу;
 * - план;
 * - результаты инструментов;
 * - финальный ответ.
 *
 * И проверяет:
 *
 * 1. выполнена ли задача;
 * 2. хватает ли данных;
 * 3. не противоречит ли ответ инструментам;
 * 4. нужна ли повторная попытка.
 *
 * В Jessica 4.0 Validator сможет запускать
 * автоматическое перепланирование.
 */


const groq =
    process.env.GROQ_API_KEY
        ? new OpenAI({
            apiKey:
                process.env.GROQ_API_KEY,

            baseURL:
                "https://api.groq.com/openai/v1"
        })
        : null;


/*
 * =========================================================
 * JSON CLEANUP
 * =========================================================
 */


function cleanJsonText(
    text
) {

    return text
        .replace(
            /```json/gi,
            ""
        )
        .replace(
            /```/g,
            ""
        )
        .trim();

}


/*
 * =========================================================
 * BASIC CHECK
 * =========================================================
 *
 * Простые технические проверки
 * выполняются без дополнительного AI.
 */


function basicValidation(
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

            reason:
                "Не удалось сформировать итоговый ответ"
        };

    }


    if (
        typeof answerResult.text !== "string" ||
        !answerResult.text.trim()
    ) {

        return {
            valid: false,

            shouldRetry: true,

            reason:
                "Итоговый ответ пустой"
        };

    }


    if (
        taskRunResult &&
        taskRunResult.success === false
    ) {

        return {
            valid: false,

            shouldRetry:
                taskRunResult.needsClarification !== true,

            needsClarification:
                taskRunResult.needsClarification === true,

            reason:
                taskRunResult.text ||
                "План выполнен с ошибкой"
        };

    }


    return {
        valid: true
    };

}


/*
 * =========================================================
 * AI VALIDATION
 * =========================================================
 */


export async function validateResult(
    task,
    plan,
    taskRunResult,
    answerResult
) {

    const basic =
        basicValidation(
            taskRunResult,
            answerResult
        );


    if (
        basic.valid !== true
    ) {

        return {
            success: true,

            valid: false,

            shouldRetry:
                basic.shouldRetry === true,

            needsClarification:
                basic.needsClarification === true,

            reason:
                basic.reason
        };

    }


    /*
     * Если ответ пришёл напрямую
     * из успешно выполненного инструмента,
     * в первой версии считаем его надёжным.
     *
     * Например current_time.
     */
    if (
        answerResult.source === "tool"
    ) {

        return {
            success: true,

            valid: true,

            shouldRetry: false,

            needsClarification: false,

            reason:
                "Ответ получен напрямую от успешно выполненного инструмента"
        };

    }


    /*
     * Если Groq недоступен,
     * не ломаем успешно сформированный ответ.
     */
    if (!groq) {

        return {
            success: true,

            valid: true,

            shouldRetry: false,

            needsClarification: false,

            reason:
                "AI Validator недоступен, базовая проверка пройдена"
        };

    }


    try {

        const response =
            await groq.responses.create({

                model:
                    "openai/gpt-oss-20b",

                instructions:
                    (
                        "Ты — Validator системы Jessica Core. " +

                        "Ты НЕ отвечаешь пользователю. " +
                        "Ты проверяешь качество уже полученного ответа. " +

                        "Определи, решает ли итоговый ответ исходную задачу. " +

                        "Если использовались инструменты, проверь, " +
                        "что ответ не противоречит их результатам. " +

                        "Не требуй лишних подробностей, если пользователь задал простой вопрос. " +

                        "Не отклоняй хороший ответ только потому, что его можно было бы сделать подробнее. " +

                        "shouldRetry=true ставь только тогда, когда повторная попытка " +
                        "реально может улучшить результат. " +

                        "needsClarification=true ставь только тогда, когда задача " +
                        "невыполнима без уточнения от пользователя. " +

                        "Верни ТОЛЬКО JSON без markdown. " +

                        "Формат: " +

                        JSON.stringify({
                            valid:
                                true,
                            shouldRetry:
                                false,
                            needsClarification:
                                false,
                            reason:
                                "краткая причина"
                        })
                    ),

                input:
                    (
                        `ИСХОДНАЯ ЗАДАЧА:\n` +
                        `${task}\n\n` +

                        `ПЛАН:\n` +
                        `${JSON.stringify(plan, null, 2)}\n\n` +

                        `РЕЗУЛЬТАТ ВЫПОЛНЕНИЯ ПЛАНА:\n` +
                        `${JSON.stringify(taskRunResult, null, 2)}\n\n` +

                        `ИТОГОВЫЙ ОТВЕТ:\n` +
                        `${answerResult.text}`
                    ),

                reasoning: {
                    effort:
                        "low"
                }

            });


        const raw =
            response.output_text
                ?.trim();


        if (!raw) {

            return {
                success: true,

                valid: true,

                shouldRetry: false,

                needsClarification: false,

                reason:
                    "Validator не вернул результат, базовая проверка пройдена"
            };

        }


        let validation;


        try {

            validation =
                JSON.parse(
                    cleanJsonText(
                        raw
                    )
                );

        } catch {

            console.error(
                "Validator invalid JSON:",
                raw
            );


            return {
                success: true,

                valid: true,

                shouldRetry: false,

                needsClarification: false,

                reason:
                    "Validator вернул некорректный JSON, базовая проверка пройдена"
            };

        }


        return {
            success: true,

            valid:
                validation.valid === true,

            shouldRetry:
                validation.shouldRetry === true,

            needsClarification:
                validation.needsClarification === true,

            reason:
                typeof validation.reason === "string"
                    ? validation.reason
                    : ""
        };


    } catch (error) {

        console.error(
            "Validator error:",
            error
        );


        return {
            success: true,

            valid: true,

            shouldRetry: false,

            needsClarification: false,

            reason:
                "Validator недоступен, базовая проверка пройдена"
        };

    }

}
