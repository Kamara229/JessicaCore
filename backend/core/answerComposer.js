import OpenAI from "openai";

import {
    executeAIWithRetry
} from "../ai/aiRetry.js";

import {
    buildComposerContext
} from "./composer/composerContext.js";


/*
 * =========================================================
 * JESSICA ANSWER COMPOSER
 * =========================================================
 *
 * Answer Composer получает только подготовленный
 * фактический контекст и формирует конечный ответ.
 *
 * Он НЕ должен видеть внутреннюю механику:
 *
 * - Tool Registry;
 * - arguments инструментов;
 * - шаги Planner;
 * - $from;
 * - routing.
 *
 * Подготовкой безопасного контекста занимается:
 *
 * core/composer/composerContext.js
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


const COMPOSER_MODEL =
    "openai/gpt-oss-20b";


/*
 * =========================================================
 * DIRECT TOOL ANSWER
 * =========================================================
 *
 * Некоторые инструменты уже возвращают
 * полноценный пользовательский ответ.
 *
 * Для них AI Composer не нужен.
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
 * SYSTEM PROMPT
 * =========================================================
 */


const COMPOSER_INSTRUCTIONS =
    [
        "Ты — Answer Composer системы Jessica Core.",

        "",
        "Твоя единственная задача — сформировать конечный ответ пользователю.",

        "",
        "Все действия по получению данных уже завершены.",
        "Перед тобой находится только подготовленный фактический контекст.",

        "",
        "Ты НЕ являешься агентом выполнения действий.",

        "",
        "ЗАПРЕЩЕНО:",
        "- вызывать инструменты;",
        "- создавать tool call;",
        "- предлагать системе вызвать инструмент;",
        "- выполнять поиск;",
        "- пытаться открыть URL;",
        "- менять маршрут решения;",
        "- придумывать отсутствующие данные;",
        "- дополнять источник фактами из собственной памяти, если задача требует подтверждения источником.",

        "",
        "Используй только предоставленные фактические данные.",

        "",
        "Если режим доказательств source_content,",
        "существенные фактические утверждения должны следовать из содержимого предоставленного источника.",

        "",
        "Если данных недостаточно для ответа,",
        "прямо скажи, какой информации недостаточно.",
        "Не пытайся самостоятельно её искать.",

        "",
        "Если задача не требует внешних данных",
        "и фактический контекст отсутствует,",
        "можно решить обычную интеллектуальную задачу самостоятельно.",

        "",
        "Не показывай пользователю:",
        "- внутреннюю архитектуру Jessica;",
        "- Planner;",
        "- Validator;",
        "- названия внутренних этапов;",
        "- технический JSON.",

        "",
        "Отвечай на языке пользователя.",

        "",
        "Для простого вопроса отвечай кратко.",
        "Для сложного вопроса дай достаточное объяснение.",

        "",
        "Интерфейс Jessica отображает обычный текст.",
        "Не используй Markdown-разметку.",
        "Не используй Markdown-таблицы.",
        "Не используй символы ** для выделения.",
        "Не используй заголовки с #.",
        "Избегай LaTeX.",
        "Формулы по возможности записывай обычным текстом.",

        "",
        "Верни только готовый пользовательский ответ."
    ].join(
        "\n"
    );


/*
 * =========================================================
 * AI REQUEST
 * =========================================================
 */


async function requestAnswer(
    input
) {

    return await executeAIWithRetry(
        async () => {

            return await groq
                .chat
                .completions
                .create({

                    model:
                        COMPOSER_MODEL,

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

        },
        {
            label:
                "Answer Composer"
        }
    );

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
     * =====================================================
     * 1. DIRECT TOOL RESULT
     * =====================================================
     */


    const directAnswer =
        tryDirectAnswer(
            taskRunResult
        );


    if (directAnswer) {

        return {
            success:
                true,

            text:
                directAnswer,

            source:
                "tool"
        };

    }


    /*
     * =====================================================
     * 2. AI CONFIG
     * =====================================================
     */


    if (!groq) {

        return {
            success:
                false,

            text:
                "Groq Answer Composer не настроен"
        };

    }


    /*
     * =====================================================
     * 3. BUILD SAFE CONTEXT
     * =====================================================
     *
     * ВАЖНО:
     *
     * Сюда больше НЕ передаётся JSON плана
     * и техническая структура результатов tools.
     */


    let input;


    try {

        input =
            buildComposerContext(
                task,
                plan,
                taskRunResult
            );

    } catch (error) {

        console.error(
            "Answer Composer context error:",
            error
        );


        return {
            success:
                false,

            text:
                "Не удалось подготовить данные для итогового ответа"
        };

    }


    /*
     * =====================================================
     * 4. COMPOSE
     * =====================================================
     */


    try {

        const response =
            await requestAnswer(
                input
            );


        const message =
            response
                ?.choices
                ?.[0]
                ?.message;


        /*
         * Дополнительная защита.
         *
         * Даже несмотря на отсутствие tools
         * Composer не должен пытаться
         * создавать tool call.
         */


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
                success:
                    false,

                text:
                    "Answer Composer попытался выполнить недопустимое действие"
            };

        }


        const answer =
            message
                ?.content
                ?.trim();


        if (!answer) {

            return {
                success:
                    false,

                text:
                    "Answer Composer вернул пустой ответ"
            };

        }


        /*
         * =================================================
         * 5. SUCCESS
         * =================================================
         */


        return {
            success:
                true,

            text:
                answer,

            source:
                "groq"
        };


    } catch (error) {

        console.error(
            "Answer Composer final error:",
            error
        );


        return {
            success:
                false,

            status:
                error?.status || 0,

            retryable:
                error?.status === 429,

            text:
                "Не удалось сформировать итоговый ответ"
        };

    }

}
