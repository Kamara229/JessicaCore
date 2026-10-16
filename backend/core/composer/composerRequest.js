/*
 * =========================================================
 * JESSICA COMPOSER REQUEST v3
 * =========================================================
 *
 * Координатор запроса Answer Composer.
 *
 *
 * AI transport:
 *
 * backend/ai/composerClient.js
 *
 *
 * Этот файл:
 *
 * - получает Composer prompt;
 * - проверяет input;
 * - вызывает AI client.
 *
 *
 * НЕ:
 *
 * - создаёт OpenAI client;
 * - управляет retry;
 * - выполняет tools;
 * - валидирует факты.
 *
 * =========================================================
 */


import {
    isComposerClientAvailable,
    requestComposerAI,
    getComposerModel
} from "../../ai/composerClient.js";


import {
    getComposerInstructions
} from "./composerPrompt.js";


/*
 * =========================================================
 * AVAILABILITY
 * =========================================================
 */


export function isComposerAvailable()
{

    return isComposerClientAvailable();

}


/*
 * =========================================================
 * VALIDATE INPUT
 * =========================================================
 */


function validateComposerInput(
    input
) {

    if(
        typeof input !== "string"
        ||
        !input.trim()
    ){

        throw new Error(
            "Answer Composer получил пустой контекст"
        );

    }

}


/*
 * =========================================================
 * REQUEST
 * =========================================================
 */


export async function requestComposerAnswer(
    input
) {

    validateComposerInput(
        input
    );


    if(
        !isComposerAvailable()
    ){

        throw new Error(
            "Groq Answer Composer не настроен"
        );

    }


    return await requestComposerAI({

        instructions:

            getComposerInstructions(),

        input

    });

}


/*
 * =========================================================
 * MODEL INFO
 * =========================================================
 */


export {
    getComposerModel
};
