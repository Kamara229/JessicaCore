/*
 * =========================================================
 * JESSICA PLANNER CLIENT v2
 * =========================================================
 *
 * AI-клиент Planner.
 *
 * Ответственность:
 *
 * - выбор модели Planner;
 * - параметры генерации;
 * - JSON output mode;
 * - проверка входных messages;
 * - отправка запроса в Groq;
 * - возврат сырого ответа Planner.
 *
 * НЕ отвечает за:
 *
 * - построение prompt;
 * - PlanningContext;
 * - parsing JSON;
 * - нормализацию плана;
 * - validation;
 * - retry;
 * - выполнение плана.
 *
 * =========================================================
 */


import {
    getGroqClient
} from "./groqClient.js";


/*
 * =========================================================
 * CONFIG
 * =========================================================
 */


const PLANNER_MODEL =
    "openai/gpt-oss-20b";


/*
 * План — компактный JSON.
 *
 * 800 токенов оставляют запас для:
 *
 * - многошагового плана;
 * - arguments;
 * - $from зависимостей;
 * - evidence;
 * - reasoningSummary.
 *
 * При этом лимит меньше прежних 1200,
 * что снижает максимальный token budget
 * одного Planner-запроса.
 */


const PLANNER_MAX_COMPLETION_TOKENS =
    800;


/*
 * Planner занимается маршрутизацией,
 * поэтому высокий reasoning effort
 * здесь не требуется.
 */


const PLANNER_REASONING_EFFORT =
    "low";


/*
 * =========================================================
 * MESSAGE VALIDATION
 * =========================================================
 */


function validateMessages(
    messages
) {

    if (
        !Array.isArray(messages)
    ) {

        throw new Error(
            "Planner messages должны быть массивом"
        );

    }


    if (
        messages.length === 0
    ) {

        throw new Error(
            "Planner messages пустые"
        );

    }


    for (
        let index = 0;
        index < messages.length;
        index++
    ) {

        const message =
            messages[index];


        if (
            !message ||
            typeof message !== "object"
        ) {

            throw new Error(
                `Planner message ${index + 1} некорректен`
            );

        }


        if (
            typeof message.role !== "string" ||
            !message.role.trim()
        ) {

            throw new Error(
                `Planner message ${index + 1} не содержит role`
            );

        }


        if (
            typeof message.content !== "string" ||
            !message.content.trim()
        ) {

            throw new Error(
                `Planner message ${index + 1} не содержит content`
            );

        }

    }

}


/*
 * =========================================================
 * REQUEST OPTIONS
 * =========================================================
 */


function buildPlannerRequestOptions(
    messages
) {

    return {

        model:
            PLANNER_MODEL,


        /*
         * Planner должен быть максимально
         * детерминированным.
         */

        temperature:
            0,


        /*
         * Для маршрутизации достаточно
         * минимального reasoning effort.
         */

        reasoning_effort:
            PLANNER_REASONING_EFFORT,


        /*
         * Внутреннее reasoning модели
         * в ответ API не включаем.
         */

        include_reasoning:
            false,


        /*
         * Ограничиваем максимальный
         * размер генерируемого плана.
         */

        max_completion_tokens:
            PLANNER_MAX_COMPLETION_TOKENS,


        /*
         * Planner обязан вернуть JSON.
         *
         * Полная JSON Schema здесь
         * намеренно не используется:
         * arguments инструментов динамические.
         */

        response_format: {

            type:
                "json_object"

        },


        messages

    };

}


/*
 * =========================================================
 * PLANNER CHAT
 * =========================================================
 */


export async function plannerChat(
    messages
) {

    validateMessages(
        messages
    );


    const groq =
        getGroqClient();


    const requestOptions =
        buildPlannerRequestOptions(
            messages
        );


    return await groq
        .chat
        .completions
        .create(
            requestOptions
        );

}


/*
 * =========================================================
 * MODEL INFO
 * =========================================================
 */


export function getPlannerModel() {

    return PLANNER_MODEL;

}


/*
 * =========================================================
 * OUTPUT LIMIT INFO
 * =========================================================
 */


export function getPlannerMaxCompletionTokens() {

    return PLANNER_MAX_COMPLETION_TOKENS;

}
