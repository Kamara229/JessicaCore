/*
 * =========================================================
 * JESSICA COMPOSER REQUEST v3
 * =========================================================
 *
 * Координатор запроса Answer Composer.
 *
 *
 * Flow:
 *
 * Composer Context
 *      ↓
 * Composer Prompt
 *      ↓
 * Composer AI Client
 *
 *
 * AI transport:
 *
 * backend/ai/composerClient.js
 *
 *
 * НЕ:
 *
 * - создаёт OpenAI client;
 * - знает Groq baseURL;
 * - управляет technical retry;
 * - выполняет tools;
 * - валидирует факты;
 * - определяет semantic outcome.
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
    input,
    options = {}
) {

    /*
     * =====================================================
     * INPUT
     * =====================================================
     */


    validateComposerInput(
        input
    );


    /*
     * =====================================================
     * AVAILABILITY
     * =====================================================
     */


    if(
        !isComposerAvailable()
    ){

        throw new Error(
            "Groq Answer Composer не настроен"
        );

    }


    /*
     * =====================================================
     * AI CLIENT
     * =====================================================
     */


    return await requestComposerAI({

        instructions:

            getComposerInstructions(),

        input,

        pairedSources: options?.pairedSources === true

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
