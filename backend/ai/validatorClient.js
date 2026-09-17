/*
 * =========================================================
 * JESSICA VALIDATOR CLIENT
 * =========================================================
 *
 * Единый AI клиент для всех проверочных модулей Jessica.
 *
 *
 * Используется:
 *
 * - Claim Evidence Validator
 * - AI Validator
 * - будущие Quality Validators
 *
 *
 * Отвечает только за:
 *
 * - выбор модели Validator;
 * - отправку сообщений в AI;
 * - возврат ответа модели.
 *
 *
 * НЕ отвечает за:
 *
 * - создание prompt;
 * - бизнес-логику проверки;
 * - retry;
 * - parsing JSON;
 * - принятие решений.
 *
 *
 * Архитектура:
 *
 * Validator
 *       ↓
 * validatorClient
 *       ↓
 * Groq
 *
 * =========================================================
 */


import {
    getGroqClient
} from "./groqClient.js";



/*
 * =========================================================
 * MODEL CONFIG
 * =========================================================
 */


const VALIDATOR_MODEL =
    "qwen/qwen3.8-27b";



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



    return await groq
        .chat
        .completions
        .create({

            model:
                VALIDATOR_MODEL,


            temperature:
                0,


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
