/*
 * =========================================================
 * JESSICA AI RESULT REQUEST
 * =========================================================
 *
 * Выполняет AI-запрос для Result Validator.
 *
 *
 * Отвечает только за:
 *
 * - получение готовых messages;
 * - вызов validatorChat();
 * - AI retry через executeAIWithRetry();
 * - возврат raw AI response.
 *
 *
 * НЕ отвечает за:
 *
 * - построение prompt;
 * - compaction input;
 * - parsing JSON;
 * - normalization outcomeType;
 * - semantic decisions.
 *
 * =========================================================
 */


import {
    validatorChat
} from "../../../ai/validatorClient.js";

import {
    executeAIWithRetry
} from "../../../ai/aiRetry.js";


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
        ) ||
        messages.length === 0
    ) {

        throw new Error(
            "AI Result Validator messages пустые или некорректные"
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
                `AI Result Validator message ${index + 1} некорректен`
            );

        }


        if (
            typeof message.role !== "string" ||
            !message.role.trim()
        ) {

            throw new Error(
                `AI Result Validator message ${index + 1} не содержит role`
            );

        }


        if (
            typeof message.content !== "string" ||
            !message.content.trim()
        ) {

            throw new Error(
                `AI Result Validator message ${index + 1} не содержит content`
            );

        }

    }

}


/*
 * =========================================================
 * REQUEST
 * =========================================================
 */


export async function requestAIResultValidation(
    messages
) {

    validateMessages(
        messages
    );


    return await executeAIWithRetry(

        async () => {

            return await validatorChat(
                messages
            );

        },

        {

            label:
                "AI Validator"

        }

    );

}
