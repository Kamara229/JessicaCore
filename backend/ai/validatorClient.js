/*
 * =========================================================
 * JESSICA VALIDATOR CLIENT
 * =========================================================
 *
 * Единый AI-клиент для Validator-модулей Jessica.
 *
 *
 * Используется:
 *
 * - Source Content Validator;
 * - Claim Evidence Validator;
 * - AI Semantic Validator;
 * - будущие Quality Validators.
 *
 *
 * Отвечает только за:
 *
 * - выбор модели;
 * - общие параметры AI Validator;
 * - ограничение размера ответа;
 * - отправку сообщений;
 * - возврат ответа модели.
 *
 *
 * НЕ отвечает за:
 *
 * - построение prompt;
 * - retry;
 * - parsing;
 * - бизнес-логику проверки;
 * - принятие решения valid / invalid.
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


const VALIDATOR_MODEL =
    "qwen/qwen3.8-27b";


/*
 * =========================================================
 * OUTPUT LIMIT
 * =========================================================
 *
 * У текущего Groq tier есть отдельный OTPM limit.
 *
 * Validator возвращает небольшой JSON,
 * поэтому длинная генерация ему не нужна.
 *
 * 300 токенов достаточно для:
 *
 * - valid;
 * - shouldRetry;
 * - needsClarification;
 * - reason;
 * - небольшого списка claims.
 *
 * =========================================================
 */


const VALIDATOR_MAX_COMPLETION_TOKENS =
    300;


/*
 * =========================================================
 * VALIDATE INPUT
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
            "Validator messages должны быть массивом"
        );

    }


    if (
        messages.length === 0
    ) {

        throw new Error(
            "Validator messages пустые"
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
                `Validator message ${index + 1} некорректен`
            );

        }


        if (
            typeof message.role !== "string" ||
            !message.role.trim()
        ) {

            throw new Error(
                `Validator message ${index + 1} не содержит role`
            );

        }


        if (
            typeof message.content !== "string" ||
            !message.content.trim()
        ) {

            throw new Error(
                `Validator message ${index + 1} не содержит content`
            );

        }

    }

}


/*
 * =========================================================
 * VALIDATOR CHAT
 * =========================================================
 */


export async function validatorChat(
    messages
) {


    validateMessages(
        messages
    );


    const groq =
        getGroqClient();


    /*
     * reasoning_effort = none
     *
     * Validator не должен тратить output tokens
     * на скрытое рассуждение.
     *
     *
     * response_format = json_object
     *
     * Все текущие Validator-модули ожидают JSON.
     *
     *
     * max_completion_tokens = 300
     *
     * Защищает от OTPM 429 и от слишком
     * многословных ответов модели.
     */


    return await groq
        .chat
        .completions
        .create({

            model:
                VALIDATOR_MODEL,


            temperature:
                0,


            reasoning_effort:
                "none",


            reasoning_format:
                "hidden",


            max_completion_tokens:
                VALIDATOR_MAX_COMPLETION_TOKENS,


            response_format: {

                type:
                    "json_object"

            },


            messages

        });

}


/*
 * =========================================================
 * CONFIG INFO
 * =========================================================
 */


export function getValidatorModel() {

    return VALIDATOR_MODEL;

}


/*
 * =========================================================
 * OUTPUT LIMIT INFO
 * =========================================================
 */


export function getValidatorMaxCompletionTokens() {

    return VALIDATOR_MAX_COMPLETION_TOKENS;

}
