import OpenAI from "openai";


/*
 * =========================================================
 * JESSICA ANSWER COMPOSER
 * =========================================================
 *
 * Answer Composer получает:
 *
 * - исходную задачу пользователя;
 * - план Planner;
 * - реальные результаты инструментов.
 *
 * И формирует финальный ответ пользователю.
 *
 * Он НЕ должен придумывать данные,
 * которых нет в результатах инструментов.
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
 * FORMAT TOOL RESULTS
 * =========================================================
 */


function formatToolResults(
    results
) {

    if (
        !Array.isArray(results) ||
        results.length === 0
    ) {

        return "Инструменты не использовались.";

    }


    return results
        .map(
            (
                result,
                index
            ) => {

                return JSON.stringify(
                    {
                        step:
                            index + 1,

                        tool:
                            result.tool,

                        arguments:
                            result.arguments,

                        success:
                            result.success,

                        text:
                            result.text,

                        data:
                            result.data
                    },
                    null,
                    2
                );

            }
        )
        .join(
            "\n\n"
        );

}


/*
 * =========================================================
 * DIRECT TOOL ANSWER
 * =========================================================
 *
 * Если задача состоит из одного инструмента
 * и инструмент уже вернул хороший готовый текст,
 * лишний вызов AI не нужен.
 */


function tryDirectAnswer(
    taskRunResult
) {

    if (
        !taskRunResult ||
        !Array.isArray(
            taskRunResult.results
        )
    ) {

        return null;

    }


    if (
        taskRunResult.results.length !== 1
    ) {

        return null;

    }


    const result =
        taskRunResult.results[0];


    if (
        result.success !== true
    ) {

        return null;

    }


    if (
        typeof result.text !== "string" ||
        !result.text.trim()
    ) {

        return null;

    }


    return result.text.trim();

}


/*
 * =========================================================
 * COMPOSE ANSWER
 * =========================================================
 */


export async function composeAnswer(
    task,
    plan,
    taskRunResult
) {

    /*
     * Сначала пробуем вернуть прямой
     * результат инструмента.
     *
     * Например current_time уже сам возвращает:
     *
     * "Сейчас в Dubai: 14:35:22..."
     *
     * Нет смысла снова отправлять это в AI.
     */
    const directAnswer =
        tryDirectAnswer(
            taskRunResult
        );


    if (directAnswer) {

        return {
            success: true,

            text:
                directAnswer,

            source:
                "tool"
        };

    }


    if (!groq) {

        return {
            success: false,

            text:
                "Groq Answer Composer не настроен"
        };

    }


    try {

        const toolResults =
            formatToolResults(
                taskRunResult?.results || []
            );


        const response =
            await groq.responses.create({

                model:
                    "openai/gpt-oss-20b",

                instructions:
                    (
                        "Ты — Answer Composer системы Jessica Core. " +

                        "Твоя задача — сформировать готовый ответ пользователю " +
                        "на основании исходной задачи, плана и результатов инструментов. " +

                        "Не описывай внутреннюю архитектуру Jessica. " +
                        "Не показывай Planner, JSON или технические названия инструментов, " +
                        "если пользователь сам об этом не спрашивал. " +

                        "Если инструменты предоставили фактические данные, " +
                        "считай их основным источником истины. " +

                        "Не изменяй точные числа, даты, время, URL и другие данные инструмента. " +

                        "Не придумывай сведения, которых нет в результатах инструментов. " +

                        "Если инструменты не использовались, реши обычную интеллектуальную задачу самостоятельно. " +

                        "Отвечай на языке пользователя. " +

                        "Для простого вопроса отвечай кратко. " +
                        "Для сложного — настолько подробно, насколько необходимо. " +

                        "Не используй markdown-таблицы с символами |, " +
                        "пока Android Jessica не умеет нормально их отображать. " +

                        "Не выдавай скрытые рассуждения. " +
                        "Дай только полезный конечный ответ."
                    ),

                input:
                    (
                        `ИСХОДНАЯ ЗАДАЧА:\n` +
                        `${task}\n\n` +

                        `ПЛАН:\n` +
                        `${JSON.stringify(plan, null, 2)}\n\n` +

                        `РЕЗУЛЬТАТЫ ИНСТРУМЕНТОВ:\n` +
                        `${toolResults}`
                    ),

                reasoning: {
                    effort:
                        "medium"
                }

            });


        const answer =
            response.output_text
                ?.trim();


        if (!answer) {

            return {
                success: false,

                text:
                    "Answer Composer вернул пустой ответ"
            };

        }


        return {
            success: true,

            text:
                answer,

            source:
                "groq"
        };


    } catch (error) {

        console.error(
            "Answer Composer error:",
            error
        );


        return {
            success: false,

            status:
                error?.status || 0,

            text:
                "Не удалось сформировать итоговый ответ"
        };

    }

}
