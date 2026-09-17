/*
 * =========================================================
 * JESSICA PLANNER CLIENT
 * =========================================================
 *
 * Единый AI-клиент Planner.
 *
 *
 * Отвечает только за:
 *
 * - выбор модели Planner;
 * - общие параметры модели;
 * - JSON output mode;
 * - отправку сообщений;
 * - возврат ответа модели.
 *
 *
 * НЕ отвечает за:
 *
 * - построение prompt;
 * - parsing JSON;
 * - нормализацию плана;
 * - validation;
 * - retry;
 * - execution.
 *
 * =========================================================
 */


import {
    getGroqClient
} from "./groqClient.js";


/*
 * =========================================================
 * MODEL
 * =========================================================
 */


const PLANNER_MODEL =
    "openai/gpt-oss-20b";


/*
 * =========================================================
 * OUTPUT LIMIT
 * =========================================================
 *
 * План является компактным JSON.
 *
 * Лимит нужен, чтобы Planner не генерировал
 * чрезмерно большие ответы.
 *
 * При этом оставляем достаточный запас
 * для многошаговых планов.
 *
 * =========================================================
 */


const PLANNER_MAX_COMPLETION_TOKENS =
    1200;


/*
 * =========================================================
 * VALIDATE MESSAGES
 * =========================================================
 */


function validateMessages(
    messages
) {

    if (
        !Array.isArray(
            messages
        )
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


    return await groq
        .chat
        .completions
        .create({

            /*
             * Planner model.
             */

            model:
                PLANNER_MODEL,


            /*
             * Минимальная случайность.
             */

            temperature:
                0,


            /*
             * GPT-OSS reasoning.
             *
             * Planner должен рассуждать,
             * но для маршрутизации нам
             * достаточно low.
             */

            reasoning_effort:
                "low",


            /*
             * Не возвращаем reasoning
             * в API response.
             */

            include_reasoning:
                false,


            /*
             * Ограничение размера плана.
             */

            max_completion_tokens:
                PLANNER_MAX_COMPLETION_TOKENS,


            /*
             * =================================================
             * JSON OBJECT MODE
             * =================================================
             *
             * Критично:
             *
             * модель теперь обязана вернуть
             * синтаксически корректный JSON.
             *
             * Semantic correctness дальше всё равно
             * проверяет planValidator.
             *
             * JSON Schema здесь намеренно не используем,
             * потому что arguments инструментов являются
             * динамическими.
             */

            response_format: {

                type:
                    "json_object"

            },


            messages

        });

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
