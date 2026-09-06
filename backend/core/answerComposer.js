import OpenAI from "openai";


/*
 * =========================================================
 * JESSICA ANSWER COMPOSER
 * =========================================================
 *
 * Answer Composer получает:
 *
 * - исходную задачу;
 * - план;
 * - реальные результаты инструментов.
 *
 * Он формирует ТОЛЬКО конечный ответ.
 *
 * ВАЖНО:
 *
 * Answer Composer:
 *
 * - НЕ планирует;
 * - НЕ выбирает инструменты;
 * - НЕ вызывает инструменты;
 * - НЕ выполняет интернет-поиск;
 * - НЕ должен повторно решать routing.
 *
 * Все действия к этому моменту уже выполнены
 * Planner + TaskRunner.
 */


/*
 * =========================================================
 * AI CLIENT
 * =========================================================
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
 * CONFIG
 * =========================================================
 */


const MAX_TOOL_CONTEXT =
    50000;


/*
 * =========================================================
 * FORMAT TOOL RESULTS
 * =========================================================
 */


function formatToolResults(
    results
) {

    if (
        !Array.isArray(
            results
        ) ||
        results.length === 0
    ) {

        return "Инструменты не использовались.";

    }


    const formatted =
        results
            .map(
                (
                    result,
                    index
                ) => {

                    return JSON.stringify(
                        {
                            step:
                                index + 1,

                            id:
                                result.id || null,

                            tool:
                                result.tool,

                            arguments:
                                result.arguments || {},

                            success:
                                result.success === true,

                            text:
                                result.text || "",

                            data:
                                result.data ?? null
                        },
                        null,
                        2
                    );

                }
            )
            .join(
                "\n\n"
            );


    /*
     * Защита от слишком большого контекста.
     *
     * web_fetch может вернуть крупную страницу.
     */
    if (
        formatted.length >
        MAX_TOOL_CONTEXT
    ) {

        return (
            formatted.slice(
                0,
                MAX_TOOL_CONTEXT
            ) +
            "\n\n[Часть технического контекста сокращена из-за объёма]"
        );

    }


    return formatted;

}


/*
 * =========================================================
 * DIRECT TOOL ANSWER
 * =========================================================
 *
 * Некоторые инструменты сами возвращают
 * полноценный конечный ответ.
 *
 * Например current_time.
 *
 * В таком случае дополнительный AI-вызов
 * не нужен.
 *
 *
 * web_search и web_fetch при успехе
 * возвращают text="",
 * поэтому сюда не попадут.
 */


/*
 * Инструменты, чей text действительно
 * является готовым пользовательским ответом.
 *
 * Это НЕ маршрутизация задач.
 *
 * Это техническое свойство результата tool.
 */
const DIRECT_ANSWER_TOOLS =
    new Set([
        "current_time"
    ]);


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
        !DIRECT_ANSWER_TOOLS.has(
            result.tool
        )
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
 * BUILD COMPOSER INPUT
 * =========================================================
 */


function buildComposerInput(
    task,
    plan,
    taskRunResult
) {

    const toolResults =
        formatToolResults(
            taskRunResult?.results || []
        );


    return (
        `ИСХОДНАЯ ЗАДАЧА ПОЛЬЗОВАТЕЛЯ:\n` +
        `${task}\n\n` +

        `ВЫПОЛНЕННЫЙ ПЛАН:\n` +
        `${JSON.stringify(plan, null, 2)}\n\n` +

        `ФАКТИЧЕСКИЕ РЕЗУЛЬТАТЫ ВЫПОЛНЕНИЯ:\n` +
        `${toolResults}`
    );

}


/*
 * =========================================================
 * SYSTEM PROMPT
 * =========================================================
 */


const COMPOSER_INSTRUCTIONS =
    [
        "Ты — Answer Composer системы Jessica Core.",

        "Твоя единственная задача — написать конечный ответ пользователю.",

        "",

        "Ты находишься ПОСЛЕ Planner и TaskRunner.",
        "Все необходимые инструменты уже были выполнены.",

        "",

        "ЗАПРЕЩЕНО:",
        "- вызывать инструменты;",
        "- просить вызвать инструмент;",
        "- создавать tool call;",
        "- выполнять повторный интернет-поиск;",
        "- менять маршрут задачи;",
        "- придумывать результаты инструментов;",
        "- утверждать, что было выполнено действие, которого нет в результатах.",

        "",

        "Если были получены результаты инструментов,",
        "используй их как основной фактический контекст.",

        "Если был прочитан веб-источник,",
        "основывай утверждения на содержимом этого источника.",

        "Если данных недостаточно для уверенного ответа,",
        "прямо укажи, чего именно не хватает.",

        "Не заполняй пробелы догадками.",

        "",

        "Если инструменты не использовались,",
        "реши обычную интеллектуальную задачу самостоятельно.",

        "",

        "Не показывай пользователю внутренний JSON,",
        "Planner, TaskRunner, названия внутренних этапов",
        "или техническую архитектуру Jessica.",

        "",

        "Отвечай на языке пользователя.",

        "Для простого вопроса отвечай кратко.",
        "Для сложного вопроса дай достаточное объяснение.",

        "",

        "Пока интерфейс Jessica отображает обычный текст.",
        "Поэтому не используй Markdown-таблицы.",
        "Избегай LaTeX и сложной математической разметки.",
        "Формулы по возможности записывай обычным текстом.",

        "",

        "Верни только готовый пользовательский ответ."
    ].join(
        "\n"
    );


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
     * -----------------------------------------------------
     * DIRECT TOOL RESULT
     * -----------------------------------------------------
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


    /*
     * -----------------------------------------------------
     * AI CONFIG
     * -----------------------------------------------------
     */


    if (!groq) {

        return {
            success: false,

            text:
                "Groq Answer Composer не настроен"
        };

    }


    /*
     * -----------------------------------------------------
     * COMPOSE
     * -----------------------------------------------------
     */


    try {

        const input =
            buildComposerInput(
                task,
                plan,
                taskRunResult
            );


        /*
         * ВАЖНО:
         *
         * Используем chat.completions,
         * а не Responses API.
         *
         * Tools сюда вообще НЕ передаются.
         *
         * Поэтому Answer Composer
         * физически отделён от Tool Registry.
         */


        const response =
            await groq.chat.completions.create({

                model:
                    "openai/gpt-oss-20b",

                messages: [
                    {
                        role:
                            "system",

                        content:
                            COMPOSER_INSTRUCTIONS
                    },
                    {
                        role:
                            "user",

                        content:
                            input
                    }
                ],

                temperature:
                    0.2

            });


        const answer =
            response.choices?.[0]
                ?.message
                ?.content
                ?.trim();


        if (!answer) {

            return {
                success: false,

                text:
                    "Answer Composer вернул пустой ответ"
            };

        }


        /*
         * Дополнительная защита:
         *
         * Composer не должен возвращать
         * структурированный tool-call вместо текста.
         */


        const message =
            response.choices?.[0]
                ?.message;


        if (
            Array.isArray(
                message?.tool_calls
            ) &&
            message.tool_calls.length > 0
        ) {

            console.error(
                "Answer Composer attempted tool call:",
                message.tool_calls
            );


            return {
                success: false,

                text:
                    "Answer Composer попытался выполнить недопустимое действие"
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
